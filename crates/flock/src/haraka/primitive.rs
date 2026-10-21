//! The Haraka-512 primitive, shared with every native hash: [`primitives::hash`].

pub use primitives::hash::{
    AES_PER_ROUND, IN_LEN, MIX, MIX_COLUMNS, OUT_LEN, ROUNDS, SBOX_PER_CALL, STATES, affine_column, haraka512,
    haraka512_witness, out_source, rc_bytes, rc_index, sbox_index,
};
