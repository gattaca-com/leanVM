//! The experimental executor runs the aggregation guest's child loop as parallel segments, however few children it has:
//! a node over eight children has the stable prover's proof, and a forged child inside a segment is refused by both
//! provers.

use aggregation::{AggregateClaims, AggregateError, AggregateProof, OutputClaims, signers_cache};
use experimental_lean_vm_core::ExperimentalProver;
use isa::{Program, Prover};
use stable_lean_vm_core::StableProver;
use std::cell::Cell;
use std::marker::PhantomData;
use verifier::E;

type Node = (AggregateClaims, AggregateProof);

/// Children of the node, each a leaf of `SIGNERS` signers over disjoint signers.
const CHILDREN: usize = 8;
const SIGNERS: usize = 6;
/// The child whose witness the forgery changes: in the middle, where the child loop runs it inside a segment.
const FORGED_CHILD: usize = 5;

/// A node on `P` at the lowest rate, publishing every claim it covers.
fn node<P: Prover>(children: &[Node], xmss: Vec<signers_cache::LeanXmssSigner>) -> Result<Node, AggregateError> {
    P::init();
    aggregation::aggregate::<P>(children, xmss, Vec::new(), None, OutputClaims::All, 1)
        .map(|(claims, proof, _)| (claims, proof))
}

/// `CHILDREN` leaves, proved by the experimental prover: both provers give a leaf the same proof.
fn leaves() -> Vec<Node> {
    let signers = signers_cache::get_signers::<ExperimentalProver>(CHILDREN * SIGNERS);
    signers
        .chunks(SIGNERS)
        .map(|chunk| node::<ExperimentalProver>(&[], chunk.to_vec()).expect("a leaf"))
        .collect()
}

thread_local! {
    /// The stream and the cell of its entries that [`Forger`] changes in [`FORGED_CHILD`]'s.
    static FORGERY: Cell<Option<(&'static str, usize)>> = const { Cell::new(None) };
}

/// Prover `P`, handed a node's witness with one cell of a child changed after the host verified the child, as a
/// malicious host could.
struct Forger<P>(PhantomData<P>);

impl<P: Prover> Prover for Forger<P> {
    fn init() {
        P::init();
    }

    fn prove(
        program: &Program,
        witness: &[(String, Vec<Vec<E>>)],
        public_input: [E; 2],
        log_inv_rate: usize,
    ) -> Result<(verifier::Proof, isa::Stats), String> {
        let (stream, cell) = FORGERY.with(Cell::get).expect("a forgery is set");
        let mut forged = witness.to_vec();
        let (_, entries) = forged
            .iter_mut()
            .find(|(name, _)| name == stream)
            .expect("the node pops the stream");
        let per_child = entries.len() / CHILDREN;
        entries[FORGED_CHILD * per_child][cell] += E::new(1, 0);
        P::prove(program, &forged, public_input, log_inv_rate)
    }

    fn map_collect<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        P::map_collect(n, f)
    }

    fn combine_rows(rows: &[u64], weights: &[E]) -> Vec<E> {
        P::combine_rows(rows, weights)
    }

    fn dot_rows(rows: &[u64], weights: &[E]) -> Vec<E> {
        P::dot_rows(rows, weights)
    }
}

/// Invariant: a node over eight children has the same proof from both provers, byte for byte, and it verifies. The
/// experimental executor runs the child loop as segments, the stable one serially.
#[test]
fn a_node_over_eight_children_is_identical() {
    let children = leaves();
    let experimental = node::<ExperimentalProver>(&children, Vec::new()).expect("an experimental node");
    let stable = node::<StableProver>(&children, Vec::new()).expect("a stable node");
    assert!(experimental.1.verify(&experimental.0), "the node verifies");
    assert!(
        experimental.1.to_bytes() == stable.1.to_bytes(),
        "the two provers' nodes differ"
    );
}

/// Invariant: a forged child inside a segment of the child loop fails the run, and the node is not proved, by either
/// prover. The forged cell is a child's commitment, a row of its opening, or a node of its cap.
#[test]
fn a_forged_child_in_a_segment_is_refused() {
    let children = leaves();
    for (what, stream, cell) in [
        ("a child's commitment", "child_signers", 0),
        ("an opened row", "merkle_leaf_rows", 0),
        ("a cap node", "merkle_caps", 4),
    ] {
        FORGERY.with(|forgery| forgery.set(Some((stream, cell))));
        assert_eq!(
            node::<Forger<ExperimentalProver>>(&children, Vec::new()).err(),
            Some(AggregateError::ProvingFailed),
            "the experimental prover refuses {what}"
        );
        assert_eq!(
            node::<Forger<StableProver>>(&children, Vec::new()).err(),
            Some(AggregateError::ProvingFailed),
            "the stable prover refuses {what}"
        );
    }
}
