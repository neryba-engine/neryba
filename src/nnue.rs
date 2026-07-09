//! NNUE inference (probe 0029, gate M2-A). Weights — net from 0028
//! (nets/out/run-20260706-neryba1), embedded in the binary. Arch
//! perspective (768 -> 128)x2 -> 1, SCReLU, SCALE 400.
//!
//! The math MIRRORS `research/0028-nnue-train1/net_eval.py` (the reference
//! that validated the 0028 gates) — including i32 wrapping (like numpy int32)
//! and floor division (like python `//`), so the parity gate is bit-for-bit.
//! First version is full recompute (incremental accumulator = separate
//! NPS probe after GREEN).

use crate::board::*;
use std::num::Wrapping;
use std::sync::OnceLock;

// probe 0040 (capacity lever): HIDDEN/net selected by cargo feature. Default
// (no feature) = net-1 128, production UNTOUCHED. Features nnue_h512/nnue_h1024
// are A/B binaries.
#[cfg(all(not(feature = "nnue_h512"), not(feature = "nnue_h1024")))]
pub const HIDDEN: usize = 128;
#[cfg(all(not(feature = "nnue_h512"), not(feature = "nnue_h1024")))]
const RAW: &[u8] = include_bytes!("nets/neryba1.bin");

#[cfg(feature = "nnue_h512")]
pub const HIDDEN: usize = 512;
#[cfg(feature = "nnue_h512")]
const RAW: &[u8] = include_bytes!("nets/neryba_h512.bin");

#[cfg(feature = "nnue_h1024")]
pub const HIDDEN: usize = 1024;
#[cfg(feature = "nnue_h1024")]
const RAW: &[u8] = include_bytes!("nets/neryba_h1024.bin");

const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

struct Net {
    fw: Vec<[i32; HIDDEN]>, // 768 features × 128
    fb: [i32; HIDDEN],
    ow: [i32; 2 * HIDDEN],
    ob: i32,
}

fn i16_at(i: usize) -> i32 {
    // little-endian i16 -> i32 (like np.fromfile int16)
    let lo = RAW[2 * i] as u16;
    let hi = RAW[2 * i + 1] as u16;
    (lo | (hi << 8)) as i16 as i32
}

fn net() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| {
        let mut fw = vec![[0i32; HIDDEN]; 768];
        let mut idx = 0;
        for f in fw.iter_mut() {
            for h in f.iter_mut() {
                *h = i16_at(idx);
                idx += 1;
            }
        }
        let mut fb = [0i32; HIDDEN];
        for h in fb.iter_mut() {
            *h = i16_at(idx);
            idx += 1;
        }
        let mut ow = [0i32; 2 * HIDDEN];
        for w in ow.iter_mut() {
            *w = i16_at(idx);
            idx += 1;
        }
        let ob = i16_at(idx);
        Net { fw, fb, ow, ob }
    })
}

/// Chess768 feature index in perspective `persp` (0=White, 1=Black).
/// c=0 if the piece belongs to the perspective; square is mirrored for Black.
#[inline]
pub fn feat(persp: u8, pcol: u8, pt0: u8, sq: u8) -> usize {
    let c = if pcol == persp { 0 } else { 1 };
    let s = if persp == WHITE { sq } else { sq ^ 56 };
    64 * (6 * c as usize + pt0 as usize) + s as usize
}

/// probe 0032: color-indexed accumulators (White-persp, Black-persp),
/// stm-independent (null-move does not touch them). acc[p][h] = fb[h] + Σ fw[feat(p,…)].
pub type Acc = [[i32; HIDDEN]; 2];

/// fresh accumulator from the board (oracle for debug_assert + initialization).
pub fn refresh(sq: &[u8; 64], occ_all: u64) -> Acc {
    let n = net();
    let mut acc = [n.fb, n.fb];
    let mut bb = occ_all;
    while bb != 0 {
        let s = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let p = sq[s as usize];
        add_piece(&mut acc, pcolor(p), ptype(p) - 1, s);
    }
    acc
}

/// deltas for make/unmake: add/remove a piece in BOTH perspectives.
#[inline]
pub fn add_piece(acc: &mut Acc, pcol: u8, pt0: u8, sq: u8) {
    let n = net();
    for persp in 0..2u8 {
        let row = &n.fw[feat(persp, pcol, pt0, sq)];
        let a = &mut acc[persp as usize];
        for h in 0..HIDDEN {
            a[h] = a[h].wrapping_add(row[h]);
        }
    }
}

#[inline]
pub fn sub_piece(acc: &mut Acc, pcol: u8, pt0: u8, sq: u8) {
    let n = net();
    for persp in 0..2u8 {
        let row = &n.fw[feat(persp, pcol, pt0, sq)];
        let a = &mut acc[persp as usize];
        for h in 0..HIDDEN {
            a[h] = a[h].wrapping_sub(row[h]);
        }
    }
}

/// output layer from ready accumulators → stm-cp (same math as `evaluate`).
pub fn eval_acc(acc: &Acc, stm: u8) -> i32 {
    let n = net();
    let us = &acc[stm as usize];
    let them = &acc[(1 - stm) as usize];
    reduce_out(us, them, n)
}

#[inline]
fn screlu(x: Wrapping<i32>) -> Wrapping<i32> {
    let y = x.0.clamp(0, QA);
    Wrapping(y) * Wrapping(y)
}

// output layer. Default (net-1, HIDDEN=128): i32 wrapping (mirror of numpy
// int32 in net_eval.py; the sum does not overflow at 128 — production byte-for-byte).
#[cfg(all(not(feature = "nnue_h512"), not(feature = "nnue_h1024")))]
#[inline]
fn reduce_out(us: &[i32; HIDDEN], them: &[i32; HIDDEN], n: &Net) -> i32 {
    let mut out = Wrapping(0i32);
    for h in 0..HIDDEN {
        out += screlu(Wrapping(us[h])) * Wrapping(n.ow[h]);
        out += screlu(Wrapping(them[h])) * Wrapping(n.ow[HIDDEN + h]);
    }
    let mut o = out.0.div_euclid(QA);
    o += n.ob;
    (Wrapping(o) * Wrapping(SCALE)).0.div_euclid(QA * QB)
}

// probe 0040: at HIDDEN≥512 the screlu·ow sum overflows i32 (silent wrap →
// garbage eval). i64 accumulator; parity is against Python bigint (net_eval,
// no wrap). Reduction math is identical.
#[cfg(any(feature = "nnue_h512", feature = "nnue_h1024"))]
#[inline]
fn reduce_out(us: &[i32; HIDDEN], them: &[i32; HIDDEN], n: &Net) -> i32 {
    let mut out: i64 = 0;
    for h in 0..HIDDEN {
        out += screlu(Wrapping(us[h])).0 as i64 * n.ow[h] as i64;
        out += screlu(Wrapping(them[h])).0 as i64 * n.ow[HIDDEN + h] as i64;
    }
    let mut o = out.div_euclid(QA as i64);
    o += n.ob as i64;
    ((o * SCALE as i64).div_euclid((QA * QB) as i64)) as i32
}

/// stm-relative cp (like net_eval.py `eval`).
pub fn evaluate(b: &Board) -> i32 {
    let n = net();
    let stm = b.stm;
    let other = 1 - stm;

    // i32 accumulator: fb + Σfw over ≤32 pieces does not overflow for any
    // HIDDEN (i16 weights × ≤32 terms ≈ 1M ≪ i32). Wrapping is unnecessary here.
    let mut us = [0i32; HIDDEN];
    let mut them = [0i32; HIDDEN];
    for (h, v) in us.iter_mut().enumerate() {
        *v = n.fb[h];
    }
    for (h, v) in them.iter_mut().enumerate() {
        *v = n.fb[h];
    }

    let mut occ = b.occ[0] | b.occ[1];
    while occ != 0 {
        let sq = occ.trailing_zeros() as u8;
        occ &= occ - 1;
        let p = b.sq[sq as usize];
        let pt0 = ptype(p) - 1; // 0..5, like python piece_type-1
        let pcol = pcolor(p);
        let iu = feat(stm, pcol, pt0, sq);
        let it = feat(other, pcol, pt0, sq);
        for h in 0..HIDDEN {
            us[h] += n.fw[iu][h];
            them[h] += n.fw[it][h];
        }
    }

    reduce_out(&us, &them, n)
}

/// demo self-check: startpos ≈ 0, +queen >> 0 (sanity check, not a gate)
pub fn demo() {
    let start = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let noq = Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let s = evaluate(&start);
    let n = evaluate(&noq);
    assert!(s.abs() < 200, "startpos not ≈0: {s}");
    assert!(n > 300, "queen advantage not visible: {n}");
    println!("nnue demo OK: start={s} +queen={n}");
}
