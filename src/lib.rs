//! Lib target for reusing the core outside the binary (probe 0088 —
//! node-invariant exception: the bin target compiles from the same inputs,
//! main.rs untouched; gate — invariant harness + bit-exact bench nodes).
//! Minimal closure for external movegen/make-unmake.

pub mod board;
pub mod nnue;
pub mod zobrist;
