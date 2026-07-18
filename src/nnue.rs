//! NNUE inference (probe 0029, gate M2-A). Default weights — net-0063
//! (probe 0063: 8 output buckets over net-2 flywheel weights, promoted
//! to default 2026-07-17), embedded in the binary. Arch:
//! perspective (768 -> 128)x2 -> 1, SCReLU, SCALE 400.
//!
//! The math MIRRORS the reference evaluator that validated the 0028
//! gates — including i32 wrapping (like numpy int32) and floor division
//! (like python `//`), so the parity gate is bit-exact.

use crate::board::*;
use std::num::Wrapping;
use std::sync::OnceLock;

// probe 0040 (capacity lever): HIDDEN/net selected by cargo feature.
// Default (no feature) = net-0063 (output buckets, probe 0063 GREEN +10.9
// deploy gate vs net-2; promoted default-on 2026-07-17, stacked with the
// 0093 i16 layout). Net-2 lives in git tag prod-20260714; net-1 in
// prod-20260710; rollback = checkout the tag.
// Features nnue_h512/nnue_h1024 are A/B binaries.
#[cfg(all(not(feature = "nnue_h512"), not(feature = "nnue_h1024"), not(feature = "nnue_aug0041"), not(feature = "nnue_kbuckets"), not(feature = "nnue_l2"), not(feature = "nnue_i8")))]
pub const HIDDEN: usize = 128;
#[cfg(all(not(feature = "nnue_h512"), not(feature = "nnue_h1024"), not(feature = "nnue_aug0041"), not(feature = "nnue_kbuckets"), not(feature = "nnue_l2"), not(feature = "nnue_i8")))]
const RAW: &[u8] = include_bytes!("nets/neryba0063.bin");

// probe 0095 — i8 requant of net-0063 (1/64 grid): fw/fb/ob rint(/4), ow as is.
#[cfg(feature = "nnue_i8")]
pub const HIDDEN: usize = 128;
#[cfg(feature = "nnue_i8")]
const RAW: &[u8] = include_bytes!("nets/neryba0095.bin");

// probe 0094 — prong C: two-layer output head (256->16->1, 8 buckets). HIDDEN=128.
#[cfg(feature = "nnue_l2")]
pub const HIDDEN: usize = 128;
#[cfg(feature = "nnue_l2")]
const RAW: &[u8] = include_bytes!("nets/neryba0094.bin");
#[cfg(feature = "nnue_l2")]
const L1: usize = 16;

// probe 0041 — Syzygy-augmented net: same arch and HIDDEN, only the weights differ.
#[cfg(feature = "nnue_aug0041")]
pub const HIDDEN: usize = 128;
#[cfg(feature = "nnue_aug0041")]
const RAW: &[u8] = include_bytes!("nets/neryba_aug0041.bin");

// probe 0090 — king input buckets (prong B), STACKED on 0063: 8 output
// buckets + 10 king input buckets (ChessBucketsMirrored). HIDDEN=128.
#[cfg(feature = "nnue_kbuckets")]
pub const HIDDEN: usize = 128;
#[cfg(feature = "nnue_kbuckets")]
const RAW: &[u8] = include_bytes!("nets/neryba0090.bin");

#[cfg(feature = "nnue_h512")]
pub const HIDDEN: usize = 512;
#[cfg(feature = "nnue_h512")]
const RAW: &[u8] = include_bytes!("nets/neryba_h512.bin");

#[cfg(feature = "nnue_h1024")]
pub const HIDDEN: usize = 1024;
#[cfg(feature = "nnue_h1024")]
const RAW: &[u8] = include_bytes!("nets/neryba_h1024.bin");

// probe 0063: number of output buckets (phase-conditioned output,
// MaterialCount<8>). Default 8 = net-0063 (promoted 2026-07-17); 1 only for
// the legacy single-bucket arms (h512/h1024/aug0041 — their bins carry no
// bucket weights).
#[cfg(any(feature = "nnue_h512", feature = "nnue_h1024", feature = "nnue_aug0041"))]
const NUM_BUCKETS: usize = 1;
#[cfg(not(any(feature = "nnue_h512", feature = "nnue_h1024", feature = "nnue_aug0041")))]
const NUM_BUCKETS: usize = 8;

// probe 0090: number of king INPUT buckets (ChessBucketsMirrored). Default 1
// = a single 768 feature block (byte-identical); nnue_kbuckets -> 10.
#[cfg(feature = "nnue_kbuckets")]
const NUM_KB: usize = 10;
#[cfg(not(feature = "nnue_kbuckets"))]
const NUM_KB: usize = 1;

// probe 0090: king square -> bucket (64 expansion of the 32-layout via file
// fold [0,1,2,3,3,2,1,0]; d/e mirror). Exact mirror of bullet
// ChessBucketsMirrored. NUM_KB=1 -> unused (bucket always 0).
#[cfg(feature = "nnue_kbuckets")]
const BUCKETS64: [usize; 64] = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 4, 5, 5, 5, 5, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
];

// probe 0095: the i8 arm lives on the 1/64 grid -> QA=64 (same reduction formula).
#[cfg(not(feature = "nnue_i8"))]
const QA: i32 = 255;
#[cfg(feature = "nnue_i8")]
const QA: i32 = 64;
const QB: i32 = 64;
const SCALE: i32 = 400;

// probe 0095: fw/fb storage type — i8 on the i8 arm (half the bytes,
// saddw widening).
#[cfg(not(feature = "nnue_i8"))]
type FwInt = i16;
#[cfg(feature = "nnue_i8")]
type FwInt = i8;

struct Net {
    // probe 0093: fw/fb stored as i16 (the native xQA quant) — 8 NEON lanes
    // per op instead of 4 and half the traffic; fw 393KB->196KB. Correctness:
    // wrapping arithmetic is modular, and the final acc fits i16
    // (headroom scan: max|acc| 6593 << 32767) -> bit-exact vs i32.
    // probe 0095: on the i8 arm FwInt=i8 (fw 98KB), acc stays i16.
    fw: Vec<[FwInt; HIDDEN]>, // 768 features x HIDDEN
    fb: [FwInt; HIDDEN],
    #[cfg(not(feature = "nnue_l2"))]
    ow: [[i32; 2 * HIDDEN]; NUM_BUCKETS], // probe 0063: phase buckets (NUM_BUCKETS=1 default = single output)
    // probe 0094: two-layer head — L1 256->16 (bucket-major rows) + L2 16->1.
    #[cfg(feature = "nnue_l2")]
    l1w: Vec<[i16; 2 * HIDDEN]>, // NUM_BUCKETS*L1 rows
    #[cfg(feature = "nnue_l2")]
    l1b: [[i32; L1]; NUM_BUCKETS],
    #[cfg(feature = "nnue_l2")]
    l2w: [[i32; L1]; NUM_BUCKETS],
    ob: [i32; NUM_BUCKETS],
}

fn i16_at(i: usize) -> i16 {
    // little-endian i16 (like np.fromfile int16)
    let lo = RAW[2 * i] as u16;
    let hi = RAW[2 * i + 1] as u16;
    (lo | (hi << 8)) as i16
}

fn net() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| {
        // probe 0090: king input buckets -> 768*NUM_KB feature blocks (NUM_KB=1 default).
        #[cfg(not(feature = "nnue_i8"))]
        let (fw, fb, mut idx) = {
            let mut fw = vec![[0i16; HIDDEN]; 768 * NUM_KB];
            let mut idx = 0;
            for f in fw.iter_mut() {
                for h in f.iter_mut() {
                    *h = i16_at(idx);
                    idx += 1;
                }
            }
            let mut fb = [0i16; HIDDEN];
            for h in fb.iter_mut() {
                *h = i16_at(idx);
                idx += 1;
            }
            (fw, fb, idx)
        };
        // probe 0095: fw/fb are i8 bytes; ow/ob follow as i16 from the shifted index.
        #[cfg(feature = "nnue_i8")]
        let (fw, fb, mut idx) = {
            let mut fw = vec![[0i8; HIDDEN]; 768 * NUM_KB];
            let mut b = 0usize;
            for f in fw.iter_mut() {
                for h in f.iter_mut() {
                    *h = RAW[b] as i8;
                    b += 1;
                }
            }
            let mut fb = [0i8; HIDDEN];
            for h in fb.iter_mut() {
                *h = RAW[b] as i8;
                b += 1;
            }
            debug_assert!(b % 2 == 0);
            (fw, fb, b / 2)
        };
        // probe 0063: l1w saved bucket-major (bullet .transpose()) -> each
        // bucket is 2*HIDDEN contiguous. NUM_BUCKETS=1 -> 256 values as in
        // net-2 (byte-identical).
        #[cfg(not(feature = "nnue_l2"))]
        let mut ow = [[0i32; 2 * HIDDEN]; NUM_BUCKETS];
        #[cfg(not(feature = "nnue_l2"))]
        for bucket in ow.iter_mut() {
            for w in bucket.iter_mut() {
                *w = i16_at(idx) as i32;
                idx += 1;
            }
        }
        // probe 0094: L1 rows bucket-major (bucket*L1+j), then l1b, l2w, ob.
        #[cfg(feature = "nnue_l2")]
        let mut l1w = vec![[0i16; 2 * HIDDEN]; NUM_BUCKETS * L1];
        #[cfg(feature = "nnue_l2")]
        for row in l1w.iter_mut() {
            for w in row.iter_mut() {
                *w = i16_at(idx);
                idx += 1;
            }
        }
        #[cfg(feature = "nnue_l2")]
        let mut l1b = [[0i32; L1]; NUM_BUCKETS];
        #[cfg(feature = "nnue_l2")]
        for bucket in l1b.iter_mut() {
            for b in bucket.iter_mut() {
                *b = i16_at(idx) as i32;
                idx += 1;
            }
        }
        #[cfg(feature = "nnue_l2")]
        let mut l2w = [[0i32; L1]; NUM_BUCKETS];
        #[cfg(feature = "nnue_l2")]
        for bucket in l2w.iter_mut() {
            for w in bucket.iter_mut() {
                *w = i16_at(idx) as i32;
                idx += 1;
            }
        }
        let mut ob = [0i32; NUM_BUCKETS];
        for o in ob.iter_mut() {
            *o = i16_at(idx) as i32;
            idx += 1;
        }
        #[cfg(not(feature = "nnue_l2"))]
        return Net { fw, fb, ow, ob };
        #[cfg(feature = "nnue_l2")]
        return Net { fw, fb, l1w, l1b, l2w, ob };
    })
}

/// Chess768 feature index in perspective `persp` (0=White, 1=Black).
/// c=0 if the piece belongs to the perspective; square mirrored for Black.
#[inline]
pub fn feat(persp: u8, pcol: u8, pt0: u8, sq: u8) -> usize {
    let c = if pcol == persp { 0 } else { 1 };
    let s = if persp == WHITE { sq } else { sq ^ 56 };
    64 * (6 * c as usize + pt0 as usize) + s as usize
}

/// probe 0090: oriented king square of a perspective (white=raw, black=^56).
#[inline]
fn ksq_or(persp: u8, kings: [u8; 2]) -> u8 {
    if persp == WHITE { kings[0] } else { kings[1] ^ 56 }
}

/// probe 0090: king input bucket + file mirror (bullet ChessBucketsMirrored).
/// `base` = feat() 0..768. NUM_KB=1 -> returns base (byte-identical to net-2/0063).
/// nnue_kbuckets: 768*bucket + (base ^ flip), flip=7 (file mirror) for king on e-h.
#[inline]
fn kb_index(base: usize, ksq_oriented: u8) -> usize {
    #[cfg(feature = "nnue_kbuckets")]
    {
        let bucket = BUCKETS64[ksq_oriented as usize];
        let flip = if ksq_oriented % 8 > 3 { 7 } else { 0 };
        768 * bucket + (base ^ flip)
    }
    #[cfg(not(feature = "nnue_kbuckets"))]
    {
        let _ = ksq_oriented;
        base
    }
}

/// probe 0032: color-indexed accumulators (White-persp, Black-persp),
/// stm-independent (null move does not touch them). acc[p][h] = fb[h] + sum fw[feat(p,..)].
/// probe 0093: i16 — wrapping is modular, so it equals the i32 truth as long
/// as the final value fits i16; the guarantee is inductive (margin assert
/// below + the headroom scan).
pub type Acc = [[i16; HIDDEN]; 2];

/// probe 0093: inductive proof of no i16 wrap in debug builds.
/// |acc| <= 29491 after EVERY delta + max|fw row| = 505 (headroom scan) ->
/// slipping past the assert through a wrap is impossible (needs a delta >= 6554).
#[cfg(debug_assertions)]
fn assert_headroom(a: &[i16; HIDDEN]) {
    for &v in a {
        debug_assert!(v.unsigned_abs() <= 29491, "i16 acc headroom breached: {v}");
    }
}

/// fresh accumulator from the board (oracle for debug_assert + init).
pub fn refresh(sq: &[u8; 64], occ_all: u64, kings: [u8; 2]) -> Acc {
    let n = net();
    let mut acc = [[0i16; HIDDEN]; 2];
    for p in 0..2 {
        for h in 0..HIDDEN {
            acc[p][h] = n.fb[h] as i16; // probe 0095: FwInt->i16 (no-op on i16 arms)
        }
    }
    let mut bb = occ_all;
    while bb != 0 {
        let s = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let p = sq[s as usize];
        add_piece(&mut acc, pcolor(p), ptype(p) - 1, s, kings);
    }
    acc
}

/// deltas for make/unmake: add/remove a piece in BOTH perspectives.
/// `kings` = [white_ksq, black_ksq] for the king input bucket (probe 0090; NUM_KB=1 -> ignored).
#[inline]
pub fn add_piece(acc: &mut Acc, pcol: u8, pt0: u8, sq: u8, kings: [u8; 2]) {
    let n = net();
    for persp in 0..2u8 {
        let row = &n.fw[kb_index(feat(persp, pcol, pt0, sq), ksq_or(persp, kings))];
        let a = &mut acc[persp as usize];
        for h in 0..HIDDEN {
            a[h] = a[h].wrapping_add(row[h] as i16);
        }
        #[cfg(debug_assertions)]
        assert_headroom(a);
    }
}

#[inline]
pub fn sub_piece(acc: &mut Acc, pcol: u8, pt0: u8, sq: u8, kings: [u8; 2]) {
    let n = net();
    for persp in 0..2u8 {
        let row = &n.fw[kb_index(feat(persp, pcol, pt0, sq), ksq_or(persp, kings))];
        let a = &mut acc[persp as usize];
        for h in 0..HIDDEN {
            a[h] = a[h].wrapping_sub(row[h] as i16);
        }
        #[cfg(debug_assertions)]
        assert_headroom(a);
    }
}

/// probe 0063: phase bucket by piece count (MaterialCount<NUM_BUCKETS>).
/// divisor = ceil(32/N) — exact bullet `game/outputs.rs` formula. NUM_BUCKETS=1
/// -> always 0 (single-bucket arms h512/h1024/aug0041 — byte-identical to the old path).
#[inline]
fn bucket_of(piece_count: u32) -> usize {
    let divisor = (32 + NUM_BUCKETS - 1) / NUM_BUCKETS; // = 32.div_ceil(N)
    (piece_count as usize).saturating_sub(2) / divisor
}

/// output layer from ready accumulators -> stm-cp (same math as `evaluate`).
/// `piece_count` = popcount(occ) selecting the phase bucket (probe 0063).
pub fn eval_acc(acc: &Acc, stm: u8, piece_count: u32) -> i32 {
    let n = net();
    let us = &acc[stm as usize];
    let them = &acc[(1 - stm) as usize];
    reduce_out(us, them, n, bucket_of(piece_count))
}

#[inline]
fn screlu(x: Wrapping<i32>) -> Wrapping<i32> {
    let y = x.0.clamp(0, QA);
    Wrapping(y) * Wrapping(y)
}

// output layer. Default (HIDDEN=128): i32 wrapping (mirror of numpy int32 in
// the reference evaluator; the sum does not overflow at 128 — prod byte-exact).
#[cfg(all(not(feature = "nnue_h512"), not(feature = "nnue_h1024"), not(feature = "nnue_l2")))]
#[inline]
fn reduce_out(us: &[i16; HIDDEN], them: &[i16; HIDDEN], n: &Net, bucket: usize) -> i32 {
    let ow = &n.ow[bucket];
    let mut out = Wrapping(0i32);
    for h in 0..HIDDEN {
        // probe 0093: widen i16->i32 at read; the math is unchanged.
        out += screlu(Wrapping(us[h] as i32)) * Wrapping(ow[h]);
        out += screlu(Wrapping(them[h] as i32)) * Wrapping(ow[HIDDEN + h]);
    }
    let mut o = out.0.div_euclid(QA);
    o += n.ob[bucket];
    (Wrapping(o) * Wrapping(SCALE)).0.div_euclid(QA * QB)
}

// probe 0040: at HIDDEN>=512 the screlu*ow sum overflows i32 (silent wrap ->
// garbage eval). i64 accumulator; parity vs Python bigint (no wrap).
// Reduction math is identical.
#[cfg(any(feature = "nnue_h512", feature = "nnue_h1024"))]
#[inline]
fn reduce_out(us: &[i16; HIDDEN], them: &[i16; HIDDEN], n: &Net, bucket: usize) -> i32 {
    let ow = &n.ow[bucket];
    let mut out: i64 = 0;
    for h in 0..HIDDEN {
        out += screlu(Wrapping(us[h] as i32)).0 as i64 * ow[h] as i64;
        out += screlu(Wrapping(them[h] as i32)).0 as i64 * ow[HIDDEN + h] as i64;
    }
    let mut o = out.div_euclid(QA as i64);
    o += n.ob[bucket] as i64;
    ((o * SCALE as i64).div_euclid((QA * QB) as i64)) as i32
}

// probe 0094: two-layer head. SCReLU activations of the concat
// (clamp^2 <= 65025 -> i32 buffer), L1: 16 neurons x 256 MAC (bucket select),
// SCReLU again, L2: 16->1. Quant normalization mirrors the bullet export
// (shifts pinned at training time; the NPS pre-gate does not depend on the
// exact shifts — the compute shape is the same).
#[cfg(feature = "nnue_l2")]
#[inline]
fn reduce_out(us: &[i16; HIDDEN], them: &[i16; HIDDEN], n: &Net, bucket: usize) -> i32 {
    let mut act = [0i32; 2 * HIDDEN];
    for h in 0..HIDDEN {
        let u = (us[h] as i32).clamp(0, QA);
        let t = (them[h] as i32).clamp(0, QA);
        act[h] = u * u;
        act[HIDDEN + h] = t * t;
    }
    let mut out = Wrapping(0i32);
    for j in 0..L1 {
        let row = &n.l1w[bucket * L1 + j];
        let mut s = Wrapping(0i32);
        for k in 0..2 * HIDDEN {
            s += Wrapping(act[k]) * Wrapping(row[k] as i32);
        }
        let a = (s.0.div_euclid(QA) + n.l1b[bucket][j]).clamp(0, QA);
        out += Wrapping(a * a) * Wrapping(n.l2w[bucket][j]);
    }
    let mut o = out.0.div_euclid(QA);
    o += n.ob[bucket];
    (Wrapping(o) * Wrapping(SCALE)).0.div_euclid(QA * QB)
}

/// stm-relative cp. probe 0093: full recompute = refresh + eval_acc
/// (the same i16 path as the incremental one — single code path).
pub fn evaluate(b: &Board) -> i32 {
    let occ_all = b.occ[0] | b.occ[1];
    let acc = refresh(&b.sq, occ_all, b.king);
    eval_acc(&acc, b.stm, occ_all.count_ones())
}

/// demo self-check: startpos ~ 0, +queen >> 0 (sanity, not a gate)
pub fn demo() {
    let start = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let noq = Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let s = evaluate(&start);
    let n = evaluate(&noq);
    assert!(s.abs() < 200, "startpos not ~0: {s}");
    assert!(n > 300, "queen advantage not visible: {n}");
    println!("nnue demo OK: start={s} +queen={n}");
}
