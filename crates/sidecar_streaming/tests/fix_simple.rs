//! `greedy_simple` and `once_at` use no cost estimate, and do what their docs say.
use sidecar_streaming::strategies::Spec;
use sidecar_streaming::strategy::{Ctx, Strategy};
use sidecar_streaming::types::*;

/// A context whose `cost` and `proof` fail the test. It records what the strategy does.
struct Mock {
    now: Micros,
    slot: SlotInfo,
    proved: Vec<Job>,
    published: Vec<ProofId>,
    timers: Vec<Micros>,
}

impl Mock {
    fn new(deadline: Micros) -> Mock {
        let slot = SlotInfo { shape: Shape::Leaf, deadline, final_rate: 2, expected_inputs: 1500, caps: Caps::default() };
        Mock { now: 0, slot, proved: Vec::new(), published: Vec::new(), timers: Vec::new() }
    }
}

impl Ctx for Mock {
    fn now(&self) -> Micros {
        self.now
    }
    fn slot(&self) -> &SlotInfo {
        &self.slot
    }
    fn proof(&self, _: ProofId) -> &ProofInfo {
        panic!("the strategy asked for a proof")
    }
    fn lane_idle(&self, _: usize) -> bool {
        true
    }
    fn prove(&mut self, _: usize, job: Job) -> JobId {
        self.proved.push(job);
        JobId(self.proved.len() as u32 - 1)
    }
    fn verify(&mut self, _: usize, _: ProofId) -> JobId {
        panic!("the strategy verified")
    }
    fn abort(&mut self, _: JobId) -> bool {
        false
    }
    fn publish(&mut self, proof: ProofId) {
        self.published.push(proof);
    }
    fn set_timer(&mut self, at: Micros, _: u64) {
        self.timers.push(at);
    }
    fn cost(&self, _: &JobShape) -> Micros {
        panic!("the strategy asked for a cost estimate")
    }
}

fn make(spec: &str) -> Box<dyn Strategy> {
    Spec::parse(spec).unwrap().make(None).unwrap()
}

fn proved(id: u32) -> Event {
    Event::Proved { job: JobId(0), lane: 0, result: Ok(ProofId(id)) }
}

#[test]
fn greedy_simple_batches_all_queued_votes_when_idle() {
    let mut s = make("greedy_simple");
    let mut c = Mock::new(4_000_000);
    s.on_event(Event::SlotStart, &mut c);
    // The first vote starts a job at once. The next three wait for the prover.
    s.on_event(Event::Signature(SigId(1)), &mut c);
    for i in 2..5 {
        s.on_event(Event::Signature(SigId(i)), &mut c);
    }
    assert_eq!(c.proved.len(), 1);
    assert_eq!((c.proved[0].signatures.len(), c.proved[0].children.len(), c.proved[0].rate), (1, 0, 2));
    // Idle again: an EXTEND of the kept proof takes all three, at the final rate.
    c.now = 1_000_000;
    s.on_event(proved(7), &mut c);
    assert_eq!(c.proved.len(), 2);
    assert_eq!((c.proved[1].signatures.len(), c.proved[1].children.clone(), c.proved[1].rate), (3, vec![ProofId(7)], 2));
    // Nothing is published by the strategy, and no timer is set.
    assert!(c.published.is_empty() && c.timers.is_empty());
}

#[test]
fn greedy_simple_starts_no_job_after_the_deadline() {
    let mut s = make("greedy_simple");
    let mut c = Mock::new(4_000_000);
    s.on_event(Event::Signature(SigId(1)), &mut c);
    s.on_event(Event::Signature(SigId(2)), &mut c);
    c.now = 4_010_000;
    s.on_event(proved(3), &mut c);
    assert_eq!(c.proved.len(), 1);
}

#[test]
fn greedy_simple_puts_a_failed_job_back() {
    let mut s = make("greedy_simple");
    let mut c = Mock::new(4_000_000);
    s.on_event(Event::Signature(SigId(1)), &mut c);
    s.on_event(Event::Signature(SigId(2)), &mut c);
    c.now = 100_000;
    s.on_event(Event::Proved { job: JobId(0), lane: 0, result: Err(Failure::Rejected("x".into())) }, &mut c);
    assert_eq!(c.proved.len(), 2);
    assert_eq!(c.proved[1].signatures, vec![SigId(1), SigId(2)]);
    assert!(c.proved[1].children.is_empty());
}

#[test]
fn once_at_proves_once_at_the_lead_time() {
    let mut s = make("once_at");
    let mut c = Mock::new(6_000_000);
    s.on_event(Event::SlotStart, &mut c);
    assert_eq!(c.timers, vec![6_000_000 - 2_200_000]);
    for i in 0..5 {
        s.on_event(Event::Signature(SigId(i)), &mut c);
    }
    assert!(c.proved.is_empty());
    c.now = 3_800_000;
    s.on_event(Event::Timer(1), &mut c);
    assert_eq!(c.proved.len(), 1);
    assert_eq!((c.proved[0].signatures.len(), c.proved[0].children.len(), c.proved[0].rate), (5, 0, 2));
    // Votes after the start are left out, and a second timer starts nothing.
    s.on_event(Event::Signature(SigId(9)), &mut c);
    s.on_event(Event::Timer(1), &mut c);
    assert_eq!(c.proved.len(), 1);
    s.on_event(proved(4), &mut c);
    assert_eq!(c.published, vec![ProofId(4)]);
}

#[test]
fn once_at_takes_its_lead_from_the_spec() {
    let mut s = make("once_at:lead_ms=1500");
    let mut c = Mock::new(4_000_000);
    s.on_event(Event::SlotStart, &mut c);
    assert_eq!(c.timers, vec![2_500_000]);
    assert_eq!(s.name(), "once_at:lead_ms=1500");
}
