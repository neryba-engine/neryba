//! Tapered evaluation — 1:1 port of sparring v3 (backend/engines/v3/evaluation.py).
//! Traceability: ADR-0004 baseline (tapered HCE) + PREREG 0008 (semantic parity).
//!
//! PeSTO tables are laid out a8=0..h1=63 (as in v3): WHITE squares index with
//! sq^56, BLACK plain. Score is centipawns from White's perspective.

use crate::board::*;
use crate::nnue;

fn nnue_on() -> bool {
    // probe 0029 GREEN (+309 Elo): NNUE default-on; the NERYBA_HCE=1 knob
    // brings back PeSTO (A/B, datagen on HCE labels)
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NERYBA_HCE").is_err())
}

pub const MATE: i32 = 100_000;
pub const MATE_THRESHOLD: i32 = MATE - 1000;

/// Search-side material (MVV-LVA ordering scale) — v2/v3 values.
pub const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 0]; // idx by ptype

const PHASE_INC: [i32; 7] = [0, 0, 1, 1, 2, 4, 0];
const TOTAL_PHASE: i32 = 24;

const MG_VALUE: [i32; 7] = [0, 82, 337, 365, 477, 1025, 0];
const EG_VALUE: [i32; 7] = [0, 94, 281, 297, 512, 936, 0];

#[rustfmt::skip]
const MG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const MG_KNIGHT: [i32; 64] = [
   -167, -89, -34, -49,  61, -97, -15,-107,
    -73, -41,  72,  36,  23,  62,   7, -17,
    -47,  60,  37,  65,  84, 129,  73,  44,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -13,   4,  16,  13,  28,  19,  21,  -8,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
   -105, -21, -58, -33, -17, -28, -19, -23,
];
#[rustfmt::skip]
const MG_BISHOP: [i32; 64] = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
];
#[rustfmt::skip]
const MG_ROOK: [i32; 64] = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
];
#[rustfmt::skip]
const MG_QUEEN: [i32; 64] = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
];
#[rustfmt::skip]
const MG_KING: [i32; 64] = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
];
#[rustfmt::skip]
const EG_PAWN: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
];
#[rustfmt::skip]
const EG_KNIGHT: [i32; 64] = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
];
#[rustfmt::skip]
const EG_BISHOP: [i32; 64] = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
];
#[rustfmt::skip]
const EG_ROOK: [i32; 64] = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
];
#[rustfmt::skip]
const EG_QUEEN: [i32; 64] = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
];
#[rustfmt::skip]
const EG_KING: [i32; 64] = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
];

const fn pst(pt: u8, mg: bool) -> &'static [i32; 64] {
    match (pt, mg) {
        (PAWN, true) => &MG_PAWN,
        (KNIGHT, true) => &MG_KNIGHT,
        (BISHOP, true) => &MG_BISHOP,
        (ROOK, true) => &MG_ROOK,
        (QUEEN, true) => &MG_QUEEN,
        (KING, true) => &MG_KING,
        (PAWN, false) => &EG_PAWN,
        (KNIGHT, false) => &EG_KNIGHT,
        (BISHOP, false) => &EG_BISHOP,
        (ROOK, false) => &EG_ROOK,
        (QUEEN, false) => &EG_QUEEN,
        _ => &EG_KING,
    }
}

// pawn-structure / king-safety constants (v3 values, verbatim)
const ISO_MG: i32 = -12;
const ISO_EG: i32 = -8;
const DBL_MG: i32 = -8;
const DBL_EG: i32 = -18;
const PASSED_MG: [i32; 8] = [0, 4, 8, 15, 30, 55, 90, 0];
const PASSED_EG: [i32; 8] = [0, 12, 24, 45, 75, 120, 180, 0];
const SHIELD_PEN: i32 = -12;
const OPEN_FILE_PEN: i32 = -22;
const SEMI_OPEN_PEN: i32 = -11;
const TROPISM_Q: i32 = -4;
const TROPISM_R: i32 = -2;

const MOB_OFFSET: [i32; 7] = [0, 0, 4, 6, 7, 13, 0];
const MOB_MG: [i32; 7] = [0, 0, 4, 4, 2, 1, 0];
const MOB_EG: [i32; 7] = [0, 0, 4, 5, 4, 2, 0];

pub fn evaluate(b: &Board) -> i32 {
    // probe 0029 (env NERYBA_NNUE, A/B): NNUE inference instead of tapered HCE.
    // nnue returns stm-cp → convert to White-cp (this function's convention).
    // The flag is cached once per process (per-node env read would be costly).
    if nnue_on() {
        // probe 0032: accumulator is already up to date (make/unmake) → output layer only
        let stm = nnue::eval_acc(&b.acc, b.stm);
        return if b.stm == WHITE { stm } else { -stm };
    }

    // probe 0022 (env NERYBA_MOPUP2, A/B): mop-up against a bare king,
    // OWN derivation of the CPW idea (drive the king to the edge/corner of the
    // bishop's color, bring the kings closer) — no borrowed tables/constants
    // (ADR-0004 extension)
    if let Some((c1, c2)) = mopup2_coeffs() {
        if let Some(s) = mopup2_eval(b, c1, c2) {
            return s;
        }
    }

    let mut mg = 0i32;
    let mut eg = 0i32;
    let mut phase = 0i32;

    // material + PST: iterate only occupied squares via bitboards (0012)
    let mut bb_all = b.occ[0] | b.occ[1];
    while bb_all != 0 {
        let s = bb_all.trailing_zeros() as u8;
        bb_all &= bb_all - 1;
        let p = b.sq[s as usize];
        let pt = ptype(p);
        phase += PHASE_INC[pt as usize];
        if pcolor(p) == WHITE {
            let idx = (s ^ 56) as usize;
            mg += MG_VALUE[pt as usize] + pst(pt, true)[idx];
            eg += EG_VALUE[pt as usize] + pst(pt, false)[idx];
        } else {
            let idx = s as usize;
            mg -= MG_VALUE[pt as usize] + pst(pt, true)[idx];
            eg -= EG_VALUE[pt as usize] + pst(pt, false)[idx];
        }
    }
    let wp = b.pieces[PAWN as usize] & b.occ[WHITE as usize];
    let bp = b.pieces[PAWN as usize] & b.occ[BLACK as usize];

    let (ps_mg, ps_eg) = pawn_structure(wp, bp);
    let (mb_mg, mb_eg) = mobility(b);
    mg += ps_mg + mb_mg;
    eg += ps_eg + mb_eg;
    mg += king_safety(b, WHITE) - king_safety(b, BLACK);

    if phase > TOTAL_PHASE {
        phase = TOTAL_PHASE;
    }
    // Python uses floor division; Rust `/` truncates -> div_euclid for parity.
    (mg * phase + eg * (TOTAL_PHASE - phase)).div_euclid(TOTAL_PHASE)
}

/// probe 0022 (PREREG research/0022-mopup-own): coefficients (C1, C2) from env
/// NERYBA_MOPUP2 ("C1,C2" or "1" = post-tune default); None = feature off.
fn mopup2_coeffs() -> Option<(i32, i32)> {
    use std::sync::OnceLock;
    static C: OnceLock<Option<(i32, i32)>> = OnceLock::new();
    *C.get_or_init(|| match std::env::var("NERYBA_MOPUP2") {
        Err(_) => None,
        Ok(v) if v == "0" => None,
        Ok(v) => {
            if let Some((a, b)) = v.split_once(',') {
                Some((a.trim().parse().unwrap_or(30), b.trim().parse().unwrap_or(8)))
            } else {
                Some((30, 8)) // default; fixed in the VERDICT after the tuning grid
            }
        }
    })
}

/// Manhattan distance to center (min over d4/e4/d5/e5) — own table derived
/// from the distance definition; corner = 6. (Reuses the 0016-mopup derivation.)
#[rustfmt::skip]
const CMD: [i32; 64] = [
    6, 5, 4, 3, 3, 4, 5, 6,
    5, 4, 3, 2, 2, 3, 4, 5,
    4, 3, 2, 1, 1, 2, 3, 4,
    3, 2, 1, 0, 0, 1, 2, 3,
    3, 2, 1, 0, 0, 1, 2, 3,
    4, 3, 2, 1, 1, 2, 3, 4,
    5, 4, 3, 2, 2, 3, 4, 5,
    6, 5, 4, 3, 3, 4, 5, 6,
];

#[inline]
fn manhattan(a: i32, b: i32) -> i32 {
    (a % 8 - b % 8).abs() + (a / 8 - b / 8).abs()
}

/// "Known win" base: above any normal eval, « MATE_THRESHOLD.
const MOPUP2_WIN_BASE: i32 = 1200;

fn mopup2_eval(b: &Board, c1: i32, c2: i32) -> Option<i32> {
    for us in [WHITE, BLACK] {
        let them = us ^ 1;
        if b.occ[them as usize].count_ones() != 1 {
            continue;
        }
        let ours = b.occ[us as usize];
        let pawns = (b.pieces[PAWN as usize] & ours).count_ones();
        let knights = (b.pieces[KNIGHT as usize] & ours).count_ones();
        let bishops = b.pieces[BISHOP as usize] & ours;
        let rooks = (b.pieces[ROOK as usize] & ours).count_ones();
        let queens = (b.pieces[QUEEN as usize] & ours).count_ones();
        // 0x55AA…: light squares (a1 dark, b1 light — python-chess layout)
        let two_color_bishops = bishops & 0x55AA_55AA_55AA_55AA != 0
            && bishops & !0x55AA_55AA_55AA_55AA != 0;
        let kbn = pawns == 0 && rooks == 0 && queens == 0
            && bishops.count_ones() == 1 && knights >= 1;
        let mating = pawns > 0 || rooks > 0 || queens > 0
            || (bishops != 0 && knights > 0) || two_color_bishops || knights >= 3;
        if !mating {
            return None; // bare king present but no mating material — normal eval
        }
        let ok = (b.pieces[KING as usize] & ours).trailing_zeros() as i32;
        let tk = (b.pieces[KING as usize] & b.occ[them as usize]).trailing_zeros() as i32;
        let push = if kbn {
            // corner of the bishop's color: light -> a8(56)/h1(7), dark -> a1(0)/h8(63)
            let bsq = bishops.trailing_zeros() as i32;
            let light = (bsq % 8 + bsq / 8) % 2 == 1;
            let (ca, cb) = if light { (56, 7) } else { (0, 63) };
            7 - manhattan(tk, ca).min(manhattan(tk, cb)).min(7)
        } else {
            CMD[tk as usize]
        };
        let mut material = 0i32;
        for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN] {
            material += (b.pieces[pt as usize] & ours).count_ones() as i32
                * EG_VALUE[pt as usize];
        }
        let score = MOPUP2_WIN_BASE + material + c1 * push + c2 * (14 - manhattan(ok, tk));
        return Some(if us == WHITE { score } else { -score });
    }
    None
}

const FILE_A: u64 = 0x0101_0101_0101_0101;

#[inline]
fn file_mask(f: usize) -> u64 {
    FILE_A << f
}

#[inline]
fn adj_files_mask(f: usize) -> u64 {
    let mut m = 0;
    if f > 0 {
        m |= FILE_A << (f - 1);
    }
    if f < 7 {
        m |= FILE_A << (f + 1);
    }
    m
}

/// enemy pawns strictly ahead of s (own+adjacent files) — passed-pawn mask
#[inline]
fn front_span(s: u8, white: bool) -> u64 {
    let f = (s & 7) as usize;
    let r = s >> 3;
    let files = file_mask(f) | adj_files_mask(f);
    if white {
        // ranks strictly above r
        if r >= 7 { 0 } else { files & (!0u64 << ((r as u64 + 1) * 8)) }
    } else if r == 0 {
        0
    } else {
        files & ((1u64 << (r as u64 * 8)) - 1)
    }
}

fn pawn_structure(wp: u64, bp: u64) -> (i32, i32) {
    let mut mg = 0;
    let mut eg = 0;

    let mut bb = wp;
    while bb != 0 {
        let s = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let f = (s & 7) as usize;
        if wp & adj_files_mask(f) == 0 {
            mg += ISO_MG;
            eg += ISO_EG;
        }
        if bp & front_span(s, true) == 0 {
            let r = (s >> 3) as usize;
            mg += PASSED_MG[r];
            eg += PASSED_EG[r];
        }
    }
    let mut bb = bp;
    while bb != 0 {
        let s = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let f = (s & 7) as usize;
        if bp & adj_files_mask(f) == 0 {
            mg -= ISO_MG;
            eg -= ISO_EG;
        }
        if wp & front_span(s, false) == 0 {
            let rel = 7 - (s >> 3) as usize;
            mg -= PASSED_MG[rel];
            eg -= PASSED_EG[rel];
        }
    }
    for f in 0..8 {
        let wc = (wp & file_mask(f)).count_ones() as i32;
        if wc > 1 {
            mg += DBL_MG * (wc - 1);
            eg += DBL_EG * (wc - 1);
        }
        let bc = (bp & file_mask(f)).count_ones() as i32;
        if bc > 1 {
            mg -= DBL_MG * (bc - 1);
            eg -= DBL_EG * (bc - 1);
        }
    }
    (mg, eg)
}

/// Mobility via ray-table attacks (probe 0010): identical VALUES to the old
/// per-step walk — attacks include the first blocker, & !own excludes own.
/// 0012: iterate only minor/major piece bitboards instead of all 64 squares.
fn mobility(b: &Board) -> (i32, i32) {
    let occ_all = b.occ[0] | b.occ[1];
    let mut mg = 0;
    let mut eg = 0;
    let movers = b.pieces[KNIGHT as usize] | b.pieces[BISHOP as usize]
        | b.pieces[ROOK as usize] | b.pieces[QUEEN as usize];
    let mut bb = movers;
    while bb != 0 {
        let s = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let p = b.sq[s as usize];
        let pt = ptype(p);
        let c = pcolor(p);
        let att = match pt {
            KNIGHT => KNIGHT_ATT_PUB[s as usize],
            BISHOP => slider_attacks(s, occ_all, true),
            ROOK => slider_attacks(s, occ_all, false),
            _ => slider_attacks(s, occ_all, true) | slider_attacks(s, occ_all, false),
        };
        let m = (att & !b.occ[c as usize]).count_ones() as i32 - MOB_OFFSET[pt as usize];
        let sign = if c == WHITE { 1 } else { -1 };
        mg += sign * MOB_MG[pt as usize] * m;
        eg += sign * MOB_EG[pt as usize] * m;
    }
    (mg, eg)
}

fn king_safety(b: &Board, color: u8) -> i32 {
    let kbb = b.pieces[KING as usize] & b.occ[color as usize];
    if kbb == 0 {
        return 0;
    }
    let ksq = kbb.trailing_zeros() as u8;
    let f = (ksq & 7) as i8;
    let rank = (ksq >> 3) as i8;
    let mut score = 0i32;

    let own_pawns = b.pieces[PAWN as usize] & b.occ[color as usize];
    let all_pawns = b.pieces[PAWN as usize];

    // pawn shield: own pawns on king file ±1, two ranks in front (mask op)
    let shield_files = file_mask(f as usize) | adj_files_mask(f as usize);
    let mut front = 0u64;
    let rr: [i8; 2] = if color == WHITE { [rank + 1, rank + 2] } else { [rank - 1, rank - 2] };
    for r in rr {
        if (0..8).contains(&r) {
            front |= 0xFFu64 << (r as u64 * 8);
        }
    }
    let shield = (own_pawns & shield_files & front).count_ones() as i32;
    score += SHIELD_PEN * (3 - shield.min(3));

    // open / semi-open files around the king
    for ff in [f - 1, f, f + 1] {
        if (0..8).contains(&ff) {
            let fm = file_mask(ff as usize);
            if own_pawns & fm == 0 {
                score += if all_pawns & fm == 0 { OPEN_FILE_PEN } else { SEMI_OPEN_PEN };
            }
        }
    }

    // tropism: enemy queens/rooks near our king (iterate only Q/R bitboards)
    let enemy_occ = b.occ[(1 - color) as usize];
    for (pt, w) in [(QUEEN, TROPISM_Q), (ROOK, TROPISM_R)] {
        let mut bb = b.pieces[pt as usize] & enemy_occ;
        while bb != 0 {
            let s = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            let df = ((s & 7) as i8 - f).abs();
            let dr = ((s >> 3) as i8 - rank).abs();
            score += w * (7 - df.max(dr)) as i32;
        }
    }
    score
}
