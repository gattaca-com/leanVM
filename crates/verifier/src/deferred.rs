//! The claims an aggregate defers to its outermost verifier, on polynomials fixed by the protocol (doc/leanvm, "Deferred evaluation claims").
//!
//! The stacked bytecode is the first, evaluated by [`crate::Bytecode::evaluate`]. The other three are flock's matrices.

use crate::field::E;
use crate::haraka::{HARAKA_K_LOG, HARAKA_ROWS_LOG, haraka_row_values};
use crate::multilinear::{eq_kernel, multilinear_eval};
use crate::{VerifierError, require};

/// The multilinear extensions of `A0`, `B0` and `C0` at `point`: the row coordinates, then the column coordinates.
///
/// The same forward walk of the Haraka circuit as the lincheck's terminal check, against eq weights on both sides.
pub fn matrix_values(point: &[E]) -> Result<[E; 3], VerifierError> {
    require(
        point.len() == HARAKA_ROWS_LOG + HARAKA_K_LOG,
        VerifierError::MultilinearSize,
    )?;
    let (row, column) = point.split_at(HARAKA_ROWS_LOG);
    let [a, b, c] = haraka_row_values(&eq_kernel(column));
    Ok([
        multilinear_eval(&a, row)?,
        multilinear_eval(&b, row)?,
        multilinear_eval(&c, row)?,
    ])
}
