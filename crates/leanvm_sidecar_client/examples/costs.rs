//! Op cost against size, for any backend that speaks the protocol:
//! costs <channel> <signatures-file> <out.csv> [ops] [timed-runs]
//!
//! `ops` is a comma-separated subset of `aggregate,extend,recurse,verify` (default: all). Every point
//! runs once untimed, then `timed-runs` times (default 5, or 3 when the untimed run took over 3 s).
//! A row's `ms` is the time from `send` to the response. Failures are rows with `ms` = FAIL.
//! Signatures come from the file, as disjoint slices: the claims of 16:1 children must not overlap.

use leanvm_sidecar_client::{
    Client, FLAG_KEEP_PROOF, FLAG_USE_KEPT, FLAG_VERIFY_CHILDREN, OP_AGGREGATE, OP_EXTEND, OP_RECURSE, OP_VERIFY,
    Response, ST_OK,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const RESP_CAP: usize = 16 << 20;
const AGGREGATE_SIZES: [usize; 19] =
    [1, 16, 32, 64, 100, 128, 150, 200, 256, 300, 384, 500, 512, 600, 750, 1000, 1024, 1200, 1500];
const KEPT_SIZES: [usize; 4] = [150, 300, 750, 1350];
const NEW_SIZES: [usize; 5] = [1, 50, 150, 300, 750];
const CHILD_COUNTS: [usize; 7] = [2, 3, 4, 6, 8, 12, 16];
const LEAF: usize = 1500;
const SMALL: usize = 100;

/// A `proof | claims` body, split, each part with its length prefix.
#[derive(Clone)]
struct Proved {
    proof: Vec<u8>,
    claims: Vec<u8>,
}

impl Proved {
    fn parse(body: &[u8]) -> Self {
        let len = u64::from_le_bytes(body[..8].try_into().unwrap()) as usize;
        let (proof, claims) = body.split_at(8 + len);
        Self {
            proof: proof.to_vec(),
            claims: claims.to_vec(),
        }
    }

    /// The proof's size without its length prefix.
    fn proof_bytes(&self) -> usize {
        self.proof.len() - 8
    }

    /// `claims | proof`, the encoding of a child and of a `VERIFY` payload.
    fn child(&self) -> Vec<u8> {
        [self.claims.as_slice(), &self.proof].concat()
    }
}

struct Bench {
    client: Client,
    signatures: Vec<u8>,
    record: usize,
    csv: File,
    timed_runs: Option<usize>,
}

/// One point of the table; the fields that do not apply are 0.
#[derive(Clone, Copy, Default)]
struct Point {
    rate: u32,
    n_sigs: usize,
    children: usize,
    child_sigs: usize,
    kept_sigs: usize,
    new_sigs: usize,
    verify_children: bool,
}

fn call(client: &mut Client, op: u32, flags: u32, rate: u32, payload: &[u8]) -> (Response, f64) {
    let started = Instant::now();
    let id = client.send(op, flags, rate, payload, RESP_CAP).expect("no room in the channel");
    let response = loop {
        match client.try_recv() {
            Some(response) => break response,
            None => std::hint::spin_loop(),
        }
    };
    let ms = started.elapsed().as_secs_f64() * 1e3;
    assert_eq!(response.id, id);
    (response, ms)
}

impl Bench {
    /// The `count` signatures from `first`, as an `AGGREGATE` payload.
    fn signatures(&self, first: usize, count: usize) -> Vec<u8> {
        let records = &self.signatures[first * self.record..(first + count) * self.record];
        [(count as u32).to_le_bytes().as_slice(), records].concat()
    }

    /// Prove `count` signatures from `first`, untimed.
    fn leaf(&mut self, first: usize, count: usize, rate: u32) -> Proved {
        let payload = self.signatures(first, count);
        let (response, _) = call(&mut self.client, OP_AGGREGATE, 0, rate, &payload);
        assert_eq!(response.status, ST_OK, "{}", String::from_utf8_lossy(&response.body));
        Proved::parse(&response.body)
    }

    fn recurse_payload(children: &[Proved]) -> Vec<u8> {
        let mut payload = (children.len() as u32).to_le_bytes().to_vec();
        for child in children {
            payload.extend(child.child());
        }
        payload
    }

    fn row(&mut self, op: &str, point: Point, run: usize, ms: Option<f64>, proof_bytes: usize) {
        let ms = ms.map_or("FAIL".to_string(), |ms| format!("{ms:.3}"));
        let line = format!(
            "{op},{},{},{},{},{},{},{},{run},{ms},{proof_bytes}\n",
            point.rate,
            point.n_sigs,
            point.children,
            point.child_sigs,
            point.kept_sigs,
            point.new_sigs,
            u8::from(point.verify_children),
        );
        self.csv.write_all(line.as_bytes()).unwrap();
        self.csv.flush().unwrap();
    }

    /// Time `request` once untimed and then repeatedly. It returns the response and its time in
    /// milliseconds, and may do untimed setup first. Returns the last response's proof.
    fn measure(
        &mut self,
        op: &str,
        point: Point,
        input_proof_bytes: usize,
        mut request: impl FnMut(&mut Self) -> (Response, f64),
    ) -> Option<Proved> {
        let (response, first_ms) = request(self);
        let label = format!("{op} {:?}", (point.rate, point.n_sigs, point.children, point.kept_sigs, point.new_sigs));
        if response.status != ST_OK {
            let text = String::from_utf8_lossy(&response.body);
            println!("{label}: FAILED status {} detail {}: {text}", response.status, response.detail);
            self.row(op, point, 1, None, 0);
            return None;
        }
        let runs = self.timed_runs.unwrap_or(if first_ms > 3000.0 { 3 } else { 5 });
        let mut times = Vec::new();
        let mut last = response;
        for run in 1..=runs {
            let (response, ms) = request(self);
            if response.status != ST_OK {
                println!("{label}: FAILED in run {run}: status {}", response.status);
                self.row(op, point, run, None, 0);
                return None;
            }
            let bytes = if op == "VERIFY" {
                input_proof_bytes
            } else {
                Proved::parse(&response.body).proof_bytes()
            };
            self.row(op, point, run, Some(ms), bytes);
            times.push(ms);
            last = response;
        }
        times.sort_by(f64::total_cmp);
        println!("{label}: median {:.1} ms, min {:.1} ms (discarded {first_ms:.1})", times[times.len() / 2], times[0]);
        (op != "VERIFY").then(|| Proved::parse(&last.body))
    }

    fn aggregate(&mut self) {
        for rate in [1, 2] {
            for n in AGGREGATE_SIZES {
                let payload = self.signatures(0, n);
                let point = Point { rate, n_sigs: n, ..Point::default() };
                self.measure("AGGREGATE", point, 0, |b| call(&mut b.client, OP_AGGREGATE, 0, rate, &payload));
            }
        }
    }

    fn extend(&mut self) {
        for rate in [1, 2] {
            for k in KEPT_SIZES {
                let kept_payload = self.signatures(0, k);
                for n in NEW_SIZES {
                    let mut payload = 0u32.to_le_bytes().to_vec();
                    payload.extend(self.signatures(k, n));
                    let point = Point { rate, kept_sigs: k, new_sigs: n, ..Point::default() };
                    self.measure("EXTEND", point, 0, |b| {
                        // The kept proof is consumed by each request: rebuild it, untimed.
                        let (kept, _) = call(&mut b.client, OP_AGGREGATE, FLAG_KEEP_PROOF, rate, &kept_payload);
                        assert_eq!(kept.status, ST_OK);
                        call(&mut b.client, OP_EXTEND, FLAG_USE_KEPT, rate, &payload)
                    });
                }
            }
        }
    }

    fn recurse(&mut self) {
        let big: Vec<Proved> = (0..16).map(|i| self.leaf(i * LEAF, LEAF, 2)).collect();
        let small: Vec<Proved> = (0..16).map(|i| self.leaf(i * SMALL, SMALL, 2)).collect();
        for rate in [1, 2] {
            for c in CHILD_COUNTS {
                let payload = Self::recurse_payload(&big[..c]);
                let point = Point { rate, children: c, child_sigs: LEAF, ..Point::default() };
                self.measure("RECURSE", point, 0, |b| call(&mut b.client, OP_RECURSE, 0, rate, &payload));
            }
            for c in [2, 4, 16] {
                let payload = Self::recurse_payload(&small[..c]);
                let point = Point { rate, children: c, child_sigs: SMALL, ..Point::default() };
                self.measure("RECURSE", point, 0, |b| call(&mut b.client, OP_RECURSE, 0, rate, &payload));
            }
            for c in [2, 16] {
                let payload = Self::recurse_payload(&big[..c]);
                let point = Point { rate, children: c, child_sigs: LEAF, verify_children: true, ..Point::default() };
                self.measure("RECURSE", point, 0, |b| {
                    call(&mut b.client, OP_RECURSE, FLAG_VERIFY_CHILDREN, rate, &payload)
                });
            }
        }
    }

    fn verify(&mut self) {
        let big: Vec<Proved> = (0..16).map(|i| self.leaf(i * LEAF, LEAF, 2)).collect();
        for rate in [1, 2] {
            let leaf = self.leaf(0, LEAF, rate);
            let payload = leaf.child();
            let point = Point { rate, n_sigs: LEAF, ..Point::default() };
            self.measure("VERIFY", point, leaf.proof_bytes(), |b| call(&mut b.client, OP_VERIFY, 0, 0, &payload));

            let payload = Self::recurse_payload(&big);
            let (response, _) = call(&mut self.client, OP_RECURSE, 0, rate, &payload);
            assert_eq!(response.status, ST_OK, "{}", String::from_utf8_lossy(&response.body));
            let root = Proved::parse(&response.body);
            let payload = root.child();
            let point = Point { rate, children: 16, child_sigs: LEAF, ..Point::default() };
            self.measure("VERIFY", point, root.proof_bytes(), |b| call(&mut b.client, OP_VERIFY, 0, 0, &payload));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: costs <channel> <signatures-file> <out.csv> [ops] [timed-runs]");
        std::process::exit(2);
    }
    let ops = args.get(4).map_or("aggregate,extend,recurse,verify", String::as_str).to_string();
    let client = Client::open(args[1].as_ref()).expect("open channel");
    let claim = client.pub_key_size() + client.message_len();
    let record = claim + client.sig_size();
    let signatures = std::fs::read(&args[2]).expect("read signatures");
    assert!(signatures.len() >= 2100 * record && signatures.len() >= 16 * LEAF * record, "signature file too short");
    let mut csv = File::create(&args[3]).expect("create csv");
    csv.write_all(b"op,rate,n_sigs,children,child_sigs,kept_sigs,new_sigs,verify_children,run,ms,proof_bytes\n")
        .unwrap();
    let timed_runs = args.get(5).map(|runs| runs.parse().expect("timed runs"));
    let mut bench = Bench { client, signatures, record, csv, timed_runs };
    for op in ops.split(',') {
        match op {
            "aggregate" => bench.aggregate(),
            "extend" => bench.extend(),
            "recurse" => bench.recurse(),
            "verify" => bench.verify(),
            other => panic!("unknown op {other}"),
        }
    }
}
