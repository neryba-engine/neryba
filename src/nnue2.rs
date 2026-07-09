//! probe 0035 — threat-input feature set (896 = 768 base + 128 threat).
//!
//! Stage 1: ONLY the index extractor + parity gate against python-chess
//! (cross-source control: `attacked()` here vs `is_attacked_by` there must
//! match bit-for-bit). Inference/accumulator is Stage 3, not here yet.
//!
//! Index space (fixed, identical to Python `features.py`):
//!   base   0..767 : `nnue::feat` = 64*(6*c+p)+s (mirrored for Black)
//!   threat 768..895: `768 + 2*s' + r`, s' — mirrored square (like base),
//!                    r=0 square attacked by the perspective (us),
//!                    r=1 square attacked by the opponent.

use crate::board::*;
use crate::nnue::feat;

/// Full sorted set of indices (base+threat) in perspective
/// `persp` (0=White, 1=Black).
pub fn feature_indices(b: &Board, persp: u8) -> Vec<usize> {
    let mut idx = Vec::with_capacity(64);

    // base 768 — the same piece features as net-1
    let mut occ = b.occ[0] | b.occ[1];
    while occ != 0 {
        let sq = occ.trailing_zeros() as u8;
        occ &= occ - 1;
        let p = b.sq[sq as usize];
        idx.push(feat(persp, pcolor(p), ptype(p) - 1, sq));
    }

    // threat 128 — per square: attacked by us / by the opponent
    let opp = 1 - persp;
    for sq in 0..64u8 {
        let sp = if persp == WHITE { sq } else { sq ^ 56 } as usize;
        if b.attacked(sq, persp) {
            idx.push(768 + 2 * sp);
        }
        if b.attacked(sq, opp) {
            idx.push(768 + 2 * sp + 1);
        }
    }

    idx.sort_unstable();
    idx
}
