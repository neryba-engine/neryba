//! Board representation + move generation (probe 0008, milestone 1).
//!
//! Deliberately boring: 64-square mailbox, pseudo-legal generation filtered
//! by make/attack-check/unmake. Correctness is proven by perft parity with
//! python-chess (PREREG 0008 control (a)); NPS work inside Rust comes in
//! later probes with their own thresholds.
//!
//! Squares are a1=0 … h8=63 (file + 8*rank), matching python-chess.

pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;

pub const EMPTY: u8 = 0;
pub const PAWN: u8 = 1;
pub const KNIGHT: u8 = 2;
pub const BISHOP: u8 = 3;
pub const ROOK: u8 = 4;
pub const QUEEN: u8 = 5;
pub const KING: u8 = 6;

#[inline]
pub fn make_piece(color: u8, ptype: u8) -> u8 {
    ptype | (color << 3)
}
#[inline]
pub fn ptype(p: u8) -> u8 {
    p & 7
}
#[inline]
pub fn pcolor(p: u8) -> u8 {
    p >> 3
}
#[inline]
pub fn file_of(s: u8) -> i8 {
    (s & 7) as i8
}
#[inline]
pub fn rank_of(s: u8) -> i8 {
    (s >> 3) as i8
}

// Castling-right bits.
pub const WK: u8 = 1;
pub const WQ: u8 = 2;
pub const BK: u8 = 4;
pub const BQ: u8 = 8;

/// castling &= MASK[from] & MASK[to] on every move — moving or capturing a
/// king/rook on its home square drops the right, everything else is a no-op.
const fn rights_masks() -> [u8; 64] {
    let mut m = [0xffu8; 64];
    m[0] = 0xff ^ WQ; // a1
    m[7] = 0xff ^ WK; // h1
    m[4] = 0xff ^ (WK | WQ); // e1
    m[56] = 0xff ^ BQ; // a8
    m[63] = 0xff ^ BK; // h8
    m[60] = 0xff ^ (BK | BQ); // e8
    m
}
const RIGHTS_MASK: [u8; 64] = rights_masks();

const fn build_leaper(deltas: [(i8, i8); 8]) -> [u64; 64] {
    let mut t = [0u64; 64];
    let mut s = 0usize;
    while s < 64 {
        let f = (s % 8) as i8;
        let r = (s / 8) as i8;
        let mut i = 0usize;
        while i < 8 {
            let nf = f + deltas[i].0;
            let nr = r + deltas[i].1;
            if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                t[s] |= 1u64 << (nr * 8 + nf);
            }
            i += 1;
        }
        s += 1;
    }
    t
}
const KNIGHT_ATT: [u64; 64] =
    build_leaper([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]);
const KING_ATT: [u64; 64] =
    build_leaper([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]);

const DIAG_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ORTHO_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

// --- ray tables (probe 0010): rays from each square in 8 directions --------
// dir index: 0 (1,1) 1 (1,-1) 2 (-1,1) 3 (-1,-1) 4 (1,0) 5 (-1,0) 6 (0,1) 7 (0,-1)
// dirs 0,2,4,6 increase the square index ("positive"), 1,3,5,7 decrease it.
const DIR8: [(i8, i8); 8] =
    [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)];

const fn build_rays() -> [[u64; 64]; 8] {
    let mut t = [[0u64; 64]; 8];
    let mut d = 0usize;
    while d < 8 {
        let mut s = 0usize;
        while s < 64 {
            let mut nf = (s % 8) as i8 + DIR8[d].0;
            let mut nr = (s / 8) as i8 + DIR8[d].1;
            while nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                t[d][s] |= 1u64 << (nr * 8 + nf);
                nf += DIR8[d].0;
                nr += DIR8[d].1;
            }
            s += 1;
        }
        d += 1;
    }
    t
}
const RAYS: [[u64; 64]; 8] = build_rays();
const POSITIVE_DIR: [bool; 8] = [true, false, true, false, true, false, true, false];

/// Attack set of a slider on `sq` over `occ` for direction range
/// (0..4 = diagonal, 4..8 = orthogonal). Includes the first blocker square.
#[inline]
pub fn slider_attacks(sq: u8, occ: u64, diag: bool) -> u64 {
    let range = if diag { 0..4 } else { 4..8 };
    let mut att = 0u64;
    for d in range {
        let ray = RAYS[d][sq as usize];
        let blockers = ray & occ;
        if blockers == 0 {
            att |= ray;
        } else {
            let fb = if POSITIVE_DIR[d] {
                blockers.trailing_zeros() as usize
            } else {
                63 - blockers.leading_zeros() as usize
            };
            att |= ray ^ RAYS[d][fb];
        }
    }
    att
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promo: u8, // 0 or KNIGHT..QUEEN
}

impl Move {
    pub fn from_uci(s: &str) -> Option<Move> {
        let b = s.as_bytes();
        if b.len() < 4 {
            return None;
        }
        let sq = |f: u8, r: u8| -> Option<u8> {
            if (b'a'..=b'h').contains(&f) && (b'1'..=b'8').contains(&r) {
                Some((f - b'a') + 8 * (r - b'1'))
            } else {
                None
            }
        };
        let promo = match b.get(4) {
            Some(b'n') => KNIGHT,
            Some(b'b') => BISHOP,
            Some(b'r') => ROOK,
            Some(b'q') => QUEEN,
            _ => 0,
        };
        Some(Move { from: sq(b[0], b[1])?, to: sq(b[2], b[3])?, promo })
    }

    pub fn uci(&self) -> String {
        let sq = |s: u8| {
            format!("{}{}", (b'a' + (s & 7)) as char, (b'1' + (s >> 3)) as char)
        };
        let p = match self.promo {
            KNIGHT => "n",
            BISHOP => "b",
            ROOK => "r",
            QUEEN => "q",
            _ => "",
        };
        format!("{}{}{}", sq(self.from), sq(self.to), p)
    }
}

pub struct Undo {
    cap: u8,
    cap_sq: u8,
    castling: u8,
    ep: Option<u8>,
    halfmove: u16,
    key: u64,
    /// probe 0032: accumulator snapshot before make (unmake restores it)
    acc: crate::nnue::Acc,
}

#[derive(Clone)]
pub struct Board {
    pub sq: [u8; 64],
    pub stm: u8,
    pub castling: u8,
    pub ep: Option<u8>,
    pub halfmove: u16,
    pub fullmove: u16,
    /// zobrist key, maintained incrementally by make/unmake
    pub key: u64,
    /// occupancy per colour, maintained incrementally (probe 0010)
    pub occ: [u64; 2],
    /// per-piece-type bitboards, both colours merged (probe 0012);
    /// colour resolved by intersecting with occ[c]. Index by ptype 1..=6.
    pub pieces: [u64; 7],
    king: [u8; 2],
    /// probe 0032: color-indexed NNUE accumulators (White-persp, Black-persp),
    /// maintained in make/unmake (delta update in make, snapshot rollback in unmake)
    pub acc: crate::nnue::Acc,
}

pub const KNIGHT_ATT_PUB: &[u64; 64] = &KNIGHT_ATT;
pub const KING_ATT_PUB: &[u64; 64] = &KING_ATT;

impl Board {
    pub fn startpos() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("bad fen: {fen}"));
        }
        let mut b = Board {
            sq: [EMPTY; 64],
            stm: WHITE,
            castling: 0,
            ep: None,
            halfmove: 0,
            fullmove: 1,
            key: 0,
            occ: [0, 0],
            pieces: [0; 7],
            king: [64, 64],
            acc: [[0i32; crate::nnue::HIDDEN]; 2],
        };
        let mut r: i8 = 7;
        let mut f: i8 = 0;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    r -= 1;
                    f = 0;
                }
                '1'..='8' => f += ch as i8 - '0' as i8,
                _ => {
                    let color = if ch.is_ascii_uppercase() { WHITE } else { BLACK };
                    let t = match ch.to_ascii_lowercase() {
                        'p' => PAWN,
                        'n' => KNIGHT,
                        'b' => BISHOP,
                        'r' => ROOK,
                        'q' => QUEEN,
                        'k' => KING,
                        _ => return Err(format!("bad piece {ch}")),
                    };
                    if r < 0 || f > 7 {
                        return Err(format!("bad fen board: {fen}"));
                    }
                    let s = (r * 8 + f) as usize;
                    b.sq[s] = make_piece(color, t);
                    if t == KING {
                        b.king[color as usize] = s as u8;
                    }
                    f += 1;
                }
            }
        }
        b.stm = if parts[1] == "b" { BLACK } else { WHITE };
        for ch in parts[2].chars() {
            b.castling |= match ch {
                'K' => WK,
                'Q' => WQ,
                'k' => BK,
                'q' => BQ,
                _ => 0,
            };
        }
        if parts[3] != "-" {
            let bytes = parts[3].as_bytes();
            b.ep = Some((bytes[0] - b'a') + 8 * (bytes[1] - b'1'));
        }
        if parts.len() > 4 {
            b.halfmove = parts[4].parse().unwrap_or(0);
        }
        if parts.len() > 5 {
            b.fullmove = parts[5].parse().unwrap_or(1);
        }
        b.key = b.compute_key();
        b.occ = b.compute_occ();
        b.pieces = b.compute_pieces();
        b.acc = crate::nnue::refresh(&b.sq, b.occ[0] | b.occ[1]); // probe 0032
        Ok(b)
    }

    /// probe 0026 (datagen): inverse of from_fen
    pub fn to_fen(&self) -> String {
        let mut board_part = String::new();
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                let p = self.sq[r * 8 + f];
                if p == EMPTY {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    board_part.push((b'0' + empty) as char);
                    empty = 0;
                }
                let ch = match ptype(p) {
                    PAWN => 'p',
                    KNIGHT => 'n',
                    BISHOP => 'b',
                    ROOK => 'r',
                    QUEEN => 'q',
                    _ => 'k',
                };
                board_part.push(if pcolor(p) == WHITE { ch.to_ascii_uppercase() } else { ch });
            }
            if empty > 0 {
                board_part.push((b'0' + empty) as char);
            }
            if r > 0 {
                board_part.push('/');
            }
        }
        let mut castle = String::new();
        for (bit, ch) in [(WK, 'K'), (WQ, 'Q'), (BK, 'k'), (BQ, 'q')] {
            if self.castling & bit != 0 {
                castle.push(ch);
            }
        }
        if castle.is_empty() {
            castle.push('-');
        }
        let ep = self.ep.map_or("-".to_string(), |s| {
            format!("{}{}", (b'a' + (s & 7)) as char, (b'1' + (s >> 3)) as char)
        });
        format!(
            "{} {} {} {} {} {}",
            board_part,
            if self.stm == WHITE { "w" } else { "b" },
            castle,
            ep,
            self.halfmove,
            self.fullmove
        )
    }

    fn compute_occ(&self) -> [u64; 2] {
        let mut occ = [0u64; 2];
        for s in 0..64 {
            let p = self.sq[s];
            if p != EMPTY {
                occ[pcolor(p) as usize] |= 1u64 << s;
            }
        }
        occ
    }

    fn compute_pieces(&self) -> [u64; 7] {
        let mut t = [0u64; 7];
        for s in 0..64 {
            let p = self.sq[s];
            if p != EMPTY {
                t[ptype(p) as usize] |= 1u64 << s;
            }
        }
        t
    }

    /// Ep square participates in the key only when an enemy pawn stands next
    /// to the pushed pawn (cheap approximation of python-chess's
    /// has_legal_en_passant — exact up to pinned-pawn corner cases).
    fn ep_relevant(&self) -> Option<u8> {
        let e = self.ep?;
        let victim_rank: i8 = if self.stm == WHITE { 4 } else { 3 };
        let f = file_of(e);
        for df in [-1i8, 1] {
            let nf = f + df;
            if (0..8).contains(&nf)
                && self.sq[(victim_rank * 8 + nf) as usize] == make_piece(self.stm, PAWN)
            {
                return Some(e);
            }
        }
        None
    }

    fn compute_key(&self) -> u64 {
        use crate::zobrist::*;
        let mut k = 0u64;
        for s in 0..64u8 {
            let p = self.sq[s as usize];
            if p != EMPTY {
                k ^= piece_key(pcolor(p), ptype(p), s);
            }
        }
        k ^= CASTLE_KEYS[self.castling as usize];
        if let Some(e) = self.ep_relevant() {
            k ^= EP_FILE_KEYS[(e & 7) as usize];
        }
        if self.stm == BLACK {
            k ^= STM_KEY;
        }
        k
    }

    #[allow(dead_code)] // used by tests now; UCI (milestone 3) uses it live
    pub fn fen(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                let p = self.sq[r * 8 + f];
                if p == EMPTY {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    s.push((b'0' + empty) as char);
                    empty = 0;
                }
                let ch = match ptype(p) {
                    PAWN => 'p',
                    KNIGHT => 'n',
                    BISHOP => 'b',
                    ROOK => 'r',
                    QUEEN => 'q',
                    _ => 'k',
                };
                s.push(if pcolor(p) == WHITE { ch.to_ascii_uppercase() } else { ch });
            }
            if empty > 0 {
                s.push((b'0' + empty) as char);
            }
            if r > 0 {
                s.push('/');
            }
        }
        s.push(if self.stm == WHITE { ' ' } else { ' ' });
        s.push_str(if self.stm == WHITE { "w" } else { "b" });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (bit, ch) in [(WK, 'K'), (WQ, 'Q'), (BK, 'k'), (BQ, 'q')] {
                if self.castling & bit != 0 {
                    s.push(ch);
                }
            }
        }
        match self.ep {
            Some(e) => {
                s.push(' ');
                s.push((b'a' + (e & 7)) as char);
                s.push((b'1' + (e >> 3)) as char);
            }
            None => s.push_str(" -"),
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }


    /// probe 0061: Static Exchange Evaluation — own derivation of the swap
    /// algorithm (least-valuable-attacker iteratively; x-ray falls out
    /// naturally via the occ mask). Value on the SEE_VAL scale from the
    /// viewpoint of the side making capture m. Convention (mirroring the
    /// python reference): promotion is not counted (attacker = pawn),
    /// en passant — the victim is a pawn.
    pub fn see(&self, m: Move) -> i32 {
        const SEE_VAL: [i32; 7] = [0, 100, 320, 330, 500, 900, 10000];
        let t = m.to as usize;
        let mover = self.sq[m.from as usize];
        let mut occ_all = (self.occ[0] | self.occ[1]) & !(1u64 << m.from);
        // victim: en passant — the pawn behind to
        let victim_val = if self.sq[t] != EMPTY {
            SEE_VAL[ptype(self.sq[t]) as usize]
        } else if ptype(mover) == PAWN && Some(m.to) == self.ep {
            let cap_sq = if pcolor(mover) == WHITE { t - 8 } else { t + 8 };
            occ_all &= !(1u64 << cap_sq);
            SEE_VAL[PAWN as usize]
        } else {
            return 0; // not a capture — SEE outside the convention
        };

        // attackers of square t for side under the occ mask (x-ray via slider recompute)
        let attackers = |side: u8, occ: u64| -> u64 {
            let mut a = 0u64;
            // pawns: attack t from the rank behind (from side's viewpoint)
            let f = file_of(t as u8);
            let r = rank_of(t as u8);
            let dr: i8 = if side == WHITE { -1 } else { 1 };
            for df in [-1i8, 1] {
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    a |= 1u64 << (nr * 8 + nf);
                }
            }
            let mut set = (a & self.pieces[PAWN as usize])
                | (KNIGHT_ATT[t] & self.pieces[KNIGHT as usize])
                | (KING_ATT[t] & self.pieces[KING as usize]);
            let diag = slider_attacks(t as u8, occ, true);
            let orth = slider_attacks(t as u8, occ, false);
            set |= diag & (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize]);
            set |= orth & (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize]);
            set & self.occ[side as usize] & occ
        };

        let mut gain = [0i32; 32];
        let mut d = 0usize;
        gain[0] = victim_val;
        let mut attacker_val = SEE_VAL[ptype(mover) as usize];
        let mut side = pcolor(mover) ^ 1;
        loop {
            let atts = attackers(side, occ_all);
            if atts == 0 {
                break;
            }
            // least valuable attacker
            let mut lva_sq = 64usize;
            let mut lva_val = i32::MAX;
            let mut s = atts;
            while s != 0 {
                let sq = s.trailing_zeros() as usize;
                s &= s - 1;
                let v = SEE_VAL[ptype(self.sq[sq]) as usize];
                if v < lva_val {
                    lva_val = v;
                    lva_sq = sq;
                }
            }
            d += 1;
            if d >= 32 {
                break;
            }
            gain[d] = attacker_val - gain[d - 1];
            attacker_val = lva_val;
            occ_all &= !(1u64 << lva_sq);
            side ^= 1;
        }
        while d > 0 {
            gain[d - 1] = -std::cmp::max(-gain[d - 1], gain[d]);
            d -= 1;
        }
        gain[0]
    }

    /// Is `t` attacked by side `by`?
    pub fn attacked(&self, t: u8, by: u8) -> bool {
        let f = file_of(t);
        let r = rank_of(t);
        // pawns: a `by` pawn attacks t from one rank behind (from `by`'s view)
        let dr: i8 = if by == WHITE { -1 } else { 1 };
        for df in [-1i8, 1] {
            let nf = f + df;
            let nr = r + dr;
            if (0..8).contains(&nf) && (0..8).contains(&nr)
                && self.sq[(nr * 8 + nf) as usize] == make_piece(by, PAWN)
            {
                return true;
            }
        }
        let mut m = KNIGHT_ATT[t as usize];
        let knight = make_piece(by, KNIGHT);
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            if self.sq[s] == knight {
                return true;
            }
        }
        let mut m = KING_ATT[t as usize];
        let king = make_piece(by, KING);
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            if self.sq[s] == king {
                return true;
            }
        }
        // sliders via ray tables over occupancy (probe 0010)
        let occ_all = self.occ[0] | self.occ[1];
        for (diag, slider) in [(true, BISHOP), (false, ROOK)] {
            let mut m = slider_attacks(t, occ_all, diag) & self.occ[by as usize];
            while m != 0 {
                let s = m.trailing_zeros() as usize;
                m &= m - 1;
                let pt = ptype(self.sq[s]);
                if pt == slider || pt == QUEEN {
                    return true;
                }
            }
        }
        false
    }

    #[allow(dead_code)] // search (milestone 2) gates null-move/evasions on this
    pub fn in_check(&self) -> bool {
        self.attacked(self.king[self.stm as usize], 1 - self.stm)
    }

    fn push_pawn_moves(&self, from: u8, to: u8, out: &mut Vec<Move>) {
        if rank_of(to) == 0 || rank_of(to) == 7 {
            for promo in [QUEEN, ROOK, BISHOP, KNIGHT] {
                out.push(Move { from, to, promo });
            }
        } else {
            out.push(Move { from, to, promo: 0 });
        }
    }

    /// Bitboard-driven pseudo generation (probe 0014). Emission ORDER is
    /// deliberately identical to the reference per-square scan: pieces by
    /// ascending square (trailing_zeros over occ), same per-piece move order
    /// (pawn: push→double→cap(-1)→cap(+1)/ep, promo QRBN; sliders: direction
    /// by direction near-to-far; castling last). Verified by debug_assert
    /// against gen_pseudo_ref on every node of the perft suite.
    pub fn gen_pseudo(&self, out: &mut Vec<Move>) {
        let us = self.stm;
        let them = 1 - us;
        let occ_all = self.occ[0] | self.occ[1];
        let fwd: i8 = if us == WHITE { 8 } else { -8 };
        let start_rank: i8 = if us == WHITE { 1 } else { 6 };
        let mut own = self.occ[us as usize];
        while own != 0 {
            let s = own.trailing_zeros() as u8;
            own &= own - 1;
            let p = self.sq[s as usize];
            let f = file_of(s);
            let r = rank_of(s);
            match ptype(p) {
                PAWN => {
                    let one = (s as i8 + fwd) as u8;
                    if occ_all & (1u64 << one) == 0 {
                        self.push_pawn_moves(s, one, out);
                        if r == start_rank {
                            let two = (one as i8 + fwd) as u8;
                            if occ_all & (1u64 << two) == 0 {
                                out.push(Move { from: s, to: two, promo: 0 });
                            }
                        }
                    }
                    for df in [-1i8, 1] {
                        let nf = f + df;
                        if !(0..8).contains(&nf) {
                            continue;
                        }
                        let t = (s as i8 + fwd + df) as u8;
                        if self.occ[them as usize] & (1u64 << t) != 0 {
                            self.push_pawn_moves(s, t, out);
                        } else if Some(t) == self.ep {
                            out.push(Move { from: s, to: t, promo: 0 });
                        }
                    }
                }
                KNIGHT | KING => {
                    let table = if ptype(p) == KNIGHT { &KNIGHT_ATT } else { &KING_ATT };
                    let mut m = table[s as usize] & !self.occ[us as usize];
                    while m != 0 {
                        let t = m.trailing_zeros() as u8;
                        m &= m - 1;
                        out.push(Move { from: s, to: t, promo: 0 });
                    }
                }
                pt => {
                    let dir_range = match pt {
                        BISHOP => 0..4usize,
                        ROOK => 4..8,
                        _ => 0..8,
                    };
                    for d in dir_range {
                        let ray = RAYS[d][s as usize];
                        let blockers = ray & occ_all;
                        let mut sub = if blockers == 0 {
                            ray
                        } else {
                            let fb = if POSITIVE_DIR[d] {
                                blockers.trailing_zeros() as usize
                            } else {
                                63 - blockers.leading_zeros() as usize
                            };
                            let mut sub = ray ^ RAYS[d][fb];
                            if self.occ[us as usize] & (1u64 << fb) != 0 {
                                sub &= !(1u64 << fb); // own blocker excluded
                            }
                            sub
                        };
                        // near-to-far: positive dirs ascend, negative descend
                        if POSITIVE_DIR[d] {
                            while sub != 0 {
                                let t = sub.trailing_zeros() as u8;
                                sub &= sub - 1;
                                out.push(Move { from: s, to: t, promo: 0 });
                            }
                        } else {
                            while sub != 0 {
                                let t = (63 - sub.leading_zeros()) as u8;
                                sub &= !(1u64 << t);
                                out.push(Move { from: s, to: t, promo: 0 });
                            }
                        }
                    }
                }
            }
        }
        self.gen_castling(out);
        #[cfg(debug_assertions)]
        {
            let mut reference = Vec::new();
            self.gen_pseudo_ref(&mut reference);
            debug_assert_eq!(*out, reference, "0014 order diverged: {}", self.fen());
        }
    }

    /// Reference per-square generator — the 0014 oracle (debug builds only
    /// call it via the assert above; release keeps it for the harness).
    #[allow(dead_code)]
    fn gen_pseudo_ref(&self, out: &mut Vec<Move>) {
        let us = self.stm;
        let them = 1 - us;
        let fwd: i8 = if us == WHITE { 8 } else { -8 };
        let start_rank: i8 = if us == WHITE { 1 } else { 6 };
        for s in 0..64u8 {
            let p = self.sq[s as usize];
            if p == EMPTY || pcolor(p) != us {
                continue;
            }
            let f = file_of(s);
            let r = rank_of(s);
            match ptype(p) {
                PAWN => {
                    let one = (s as i8 + fwd) as u8;
                    if self.sq[one as usize] == EMPTY {
                        self.push_pawn_moves(s, one, out);
                        if r == start_rank {
                            let two = (one as i8 + fwd) as u8;
                            if self.sq[two as usize] == EMPTY {
                                out.push(Move { from: s, to: two, promo: 0 });
                            }
                        }
                    }
                    for df in [-1i8, 1] {
                        let nf = f + df;
                        if !(0..8).contains(&nf) {
                            continue;
                        }
                        let t = (s as i8 + fwd + df) as u8;
                        let tp = self.sq[t as usize];
                        if tp != EMPTY && pcolor(tp) == them {
                            self.push_pawn_moves(s, t, out);
                        } else if Some(t) == self.ep {
                            out.push(Move { from: s, to: t, promo: 0 });
                        }
                    }
                }
                KNIGHT | KING => {
                    let table = if ptype(p) == KNIGHT { &KNIGHT_ATT } else { &KING_ATT };
                    let mut m = table[s as usize];
                    while m != 0 {
                        let t = m.trailing_zeros() as u8;
                        m &= m - 1;
                        let tp = self.sq[t as usize];
                        if tp == EMPTY || pcolor(tp) == them {
                            out.push(Move { from: s, to: t, promo: 0 });
                        }
                    }
                }
                _ => {
                    let dirs: &[(i8, i8)] = match ptype(p) {
                        BISHOP => &DIAG_DIRS,
                        ROOK => &ORTHO_DIRS,
                        _ => &[(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)],
                    };
                    for &(df, dr) in dirs {
                        let mut nf = f + df;
                        let mut nr = r + dr;
                        while (0..8).contains(&nf) && (0..8).contains(&nr) {
                            let t = (nr * 8 + nf) as u8;
                            let tp = self.sq[t as usize];
                            if tp == EMPTY {
                                out.push(Move { from: s, to: t, promo: 0 });
                            } else {
                                if pcolor(tp) == them {
                                    out.push(Move { from: s, to: t, promo: 0 });
                                }
                                break;
                            }
                            nf += df;
                            nr += dr;
                        }
                    }
                }
            }
        }
        self.gen_castling(out);
    }

    /// Castling: rights + rook present + empty between + king path not attacked.
    fn gen_castling(&self, out: &mut Vec<Move>) {
        let us = self.stm;
        let them = 1 - us;
        let (ksq, k_bit, q_bit, rank0) = if us == WHITE {
            (4u8, WK, WQ, 0u8)
        } else {
            (60u8, BK, BQ, 56u8)
        };
        if self.sq[ksq as usize] == make_piece(us, KING) && !self.attacked(ksq, them) {
            if self.castling & k_bit != 0
                && self.sq[(rank0 + 5) as usize] == EMPTY
                && self.sq[(rank0 + 6) as usize] == EMPTY
                && self.sq[(rank0 + 7) as usize] == make_piece(us, ROOK)
                && !self.attacked(rank0 + 5, them)
                && !self.attacked(rank0 + 6, them)
            {
                out.push(Move { from: ksq, to: rank0 + 6, promo: 0 });
            }
            if self.castling & q_bit != 0
                && self.sq[(rank0 + 1) as usize] == EMPTY
                && self.sq[(rank0 + 2) as usize] == EMPTY
                && self.sq[(rank0 + 3) as usize] == EMPTY
                && self.sq[rank0 as usize] == make_piece(us, ROOK)
                && !self.attacked(rank0 + 2, them)
                && !self.attacked(rank0 + 3, them)
            {
                out.push(Move { from: ksq, to: rank0 + 2, promo: 0 });
            }
        }
    }

    pub fn make(&mut self, m: Move) -> Undo {
        use crate::zobrist::*;
        let us = self.stm;
        let p = self.sq[m.from as usize];
        // xor OUT state-dependent key parts while the old state is still live
        let mut k = self.key;
        if let Some(e) = self.ep_relevant() {
            k ^= EP_FILE_KEYS[(e & 7) as usize];
        }
        k ^= CASTLE_KEYS[self.castling as usize];
        let mut undo = Undo {
            cap: EMPTY,
            cap_sq: m.to,
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            key: self.key,
            acc: self.acc, // probe 0032: snapshot BEFORE changes; unmake restores it
        };
        // capture (en passant takes the pawn behind the target square)
        if ptype(p) == PAWN && Some(m.to) == self.ep && self.sq[m.to as usize] == EMPTY
            && file_of(m.from) != file_of(m.to)
        {
            undo.cap_sq = if us == WHITE { m.to - 8 } else { m.to + 8 };
        }
        undo.cap = self.sq[undo.cap_sq as usize];
        if undo.cap != EMPTY {
            k ^= piece_key(pcolor(undo.cap), ptype(undo.cap), undo.cap_sq);
            self.occ[pcolor(undo.cap) as usize] &= !(1u64 << undo.cap_sq);
            self.pieces[ptype(undo.cap) as usize] &= !(1u64 << undo.cap_sq);
            crate::nnue::sub_piece(&mut self.acc, pcolor(undo.cap), ptype(undo.cap) - 1, undo.cap_sq);
        }
        self.sq[undo.cap_sq as usize] = EMPTY;

        let placed = if m.promo != 0 { make_piece(us, m.promo) } else { p };
        k ^= piece_key(us, ptype(p), m.from);
        k ^= piece_key(us, ptype(placed), m.to);
        crate::nnue::sub_piece(&mut self.acc, us, ptype(p) - 1, m.from);
        crate::nnue::add_piece(&mut self.acc, us, ptype(placed) - 1, m.to);
        self.sq[m.from as usize] = EMPTY;
        self.sq[m.to as usize] = placed;
        self.occ[us as usize] =
            (self.occ[us as usize] & !(1u64 << m.from)) | (1u64 << m.to);
        self.pieces[ptype(p) as usize] &= !(1u64 << m.from);
        self.pieces[ptype(placed) as usize] |= 1u64 << m.to;

        if ptype(p) == KING {
            self.king[us as usize] = m.to;
            let d = m.to as i8 - m.from as i8;
            if d == 2 {
                // O-O: rook h->f
                self.sq[(m.to + 1) as usize] = EMPTY;
                self.sq[(m.to - 1) as usize] = make_piece(us, ROOK);
                k ^= piece_key(us, ROOK, m.to + 1) ^ piece_key(us, ROOK, m.to - 1);
                self.occ[us as usize] =
                    (self.occ[us as usize] & !(1u64 << (m.to + 1))) | (1u64 << (m.to - 1));
                self.pieces[ROOK as usize] =
                    (self.pieces[ROOK as usize] & !(1u64 << (m.to + 1))) | (1u64 << (m.to - 1));
                crate::nnue::sub_piece(&mut self.acc, us, ROOK - 1, m.to + 1);
                crate::nnue::add_piece(&mut self.acc, us, ROOK - 1, m.to - 1);
            } else if d == -2 {
                // O-O-O: rook a->d
                self.sq[(m.to - 2) as usize] = EMPTY;
                self.sq[(m.to + 1) as usize] = make_piece(us, ROOK);
                k ^= piece_key(us, ROOK, m.to - 2) ^ piece_key(us, ROOK, m.to + 1);
                self.occ[us as usize] =
                    (self.occ[us as usize] & !(1u64 << (m.to - 2))) | (1u64 << (m.to + 1));
                self.pieces[ROOK as usize] =
                    (self.pieces[ROOK as usize] & !(1u64 << (m.to - 2))) | (1u64 << (m.to + 1));
                crate::nnue::sub_piece(&mut self.acc, us, ROOK - 1, m.to - 2);
                crate::nnue::add_piece(&mut self.acc, us, ROOK - 1, m.to + 1);
            }
        }

        self.castling &= RIGHTS_MASK[m.from as usize] & RIGHTS_MASK[m.to as usize];
        self.ep = None;
        if ptype(p) == PAWN {
            let d = m.to as i8 - m.from as i8;
            if d == 16 || d == -16 {
                self.ep = Some(((m.from as i8 + m.to as i8) / 2) as u8);
            }
        }
        if ptype(p) == PAWN || undo.cap != EMPTY {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        if us == BLACK {
            self.fullmove += 1;
        }
        self.stm = 1 - us;
        // xor IN the new state-dependent parts (new stm is live for ep_relevant)
        k ^= CASTLE_KEYS[self.castling as usize];
        k ^= STM_KEY;
        self.key = k;
        if let Some(e) = self.ep_relevant() {
            self.key ^= EP_FILE_KEYS[(e & 7) as usize];
        }
        debug_assert_eq!(self.key, self.compute_key());
        debug_assert_eq!(self.occ, self.compute_occ());
        debug_assert_eq!(self.pieces, self.compute_pieces());
        // probe 0032 oracle: incremental accumulator == fresh recompute
        debug_assert_eq!(self.acc, crate::nnue::refresh(&self.sq, self.occ[0] | self.occ[1]));
        undo
    }

    pub fn unmake(&mut self, m: Move, undo: Undo) {
        let us = 1 - self.stm; // side that made the move
        let moved = self.sq[m.to as usize];
        self.sq[m.to as usize] = EMPTY;
        self.sq[m.from as usize] = if m.promo != 0 { make_piece(us, PAWN) } else { moved };
        self.occ[us as usize] =
            (self.occ[us as usize] & !(1u64 << m.to)) | (1u64 << m.from);
        self.pieces[ptype(moved) as usize] &= !(1u64 << m.to);
        self.pieces[if m.promo != 0 { PAWN } else { ptype(moved) } as usize] |= 1u64 << m.from;
        self.sq[undo.cap_sq as usize] = undo.cap;
        if undo.cap != EMPTY {
            self.occ[pcolor(undo.cap) as usize] |= 1u64 << undo.cap_sq;
            self.pieces[ptype(undo.cap) as usize] |= 1u64 << undo.cap_sq;
        }
        if ptype(moved) == KING {
            // kings never promote, so `moved` on the to-square is the king itself
            self.king[us as usize] = m.from;
            let d = m.to as i8 - m.from as i8;
            if d == 2 {
                self.sq[(m.to + 1) as usize] = make_piece(us, ROOK);
                self.sq[(m.to - 1) as usize] = EMPTY;
                self.occ[us as usize] =
                    (self.occ[us as usize] & !(1u64 << (m.to - 1))) | (1u64 << (m.to + 1));
                self.pieces[ROOK as usize] =
                    (self.pieces[ROOK as usize] & !(1u64 << (m.to - 1))) | (1u64 << (m.to + 1));
            } else if d == -2 {
                self.sq[(m.to - 2) as usize] = make_piece(us, ROOK);
                self.sq[(m.to + 1) as usize] = EMPTY;
                self.occ[us as usize] =
                    (self.occ[us as usize] & !(1u64 << (m.to + 1))) | (1u64 << (m.to - 2));
                self.pieces[ROOK as usize] =
                    (self.pieces[ROOK as usize] & !(1u64 << (m.to + 1))) | (1u64 << (m.to - 2));
            }
        }
        self.castling = undo.castling;
        self.ep = undo.ep;
        self.halfmove = undo.halfmove;
        self.key = undo.key;
        self.acc = undo.acc; // probe 0032: snapshot rollback (zero drift)
        if us == BLACK {
            self.fullmove -= 1;
        }
        self.stm = us;
    }

    /// Any legal move at all? Early-exit version for stalemate checks.
    pub fn has_legal_move(&mut self) -> bool {
        let mut ps = Vec::with_capacity(64);
        self.gen_pseudo(&mut ps);
        for m in ps {
            let undo = self.make(m);
            let mover = 1 - self.stm;
            let ok = !self.attacked(self.king[mover as usize], self.stm);
            self.unmake(m, undo);
            if ok {
                return true;
            }
        }
        false
    }

    /// Null move (pass): flip side to move, clear ep. For null-move pruning.
    pub fn make_null(&mut self) -> (Option<u8>, u64) {
        let saved = (self.ep, self.key);
        self.ep = None;
        self.stm = 1 - self.stm;
        self.key = self.compute_key();
        (saved.0, saved.1)
    }

    pub fn unmake_null(&mut self, saved: (Option<u8>, u64)) {
        self.stm = 1 - self.stm;
        self.ep = saved.0;
        self.key = saved.1;
    }

    /// Side to move has a non-pawn, non-king piece (null-move zugzwang gate).
    pub fn has_non_pawn_material(&self) -> bool {
        (0..64).any(|s| {
            let p = self.sq[s];
            p != EMPTY && pcolor(p) == self.stm && matches!(ptype(p), KNIGHT | BISHOP | ROOK | QUEEN)
        })
    }

    /// Insufficient material: bare kings, or king + single minor vs king.
    /// (Same subset v3 relies on via python-chess for the common cases.)
    pub fn is_insufficient_material(&self) -> bool {
        let mut minors = 0;
        for s in 0..64 {
            let p = self.sq[s];
            if p == EMPTY {
                continue;
            }
            match ptype(p) {
                KING => {}
                KNIGHT | BISHOP => minors += 1,
                _ => return false,
            }
        }
        minors <= 1
    }

    /// Reference legality filter: make -> king attacked? -> unmake.
    /// Kept as the oracle for debug_assert in gen_legal (probe 0009 arm A).
    fn gen_legal_ref(&mut self) -> Vec<Move> {
        let mut ps = Vec::with_capacity(64);
        self.gen_pseudo(&mut ps);
        let mut out = Vec::with_capacity(ps.len());
        for m in ps {
            let undo = self.make(m);
            let mover = 1 - self.stm;
            if !self.attacked(self.king[mover as usize], self.stm) {
                out.push(m);
            }
            self.unmake(m, undo);
        }
        out
    }

    /// Checkers of the stm king + pins: (n_checkers, allowed_target_mask,
    /// pins as (from_sq, allowed_ray_mask) pairs).
    /// allowed_target_mask: squares a NON-king move may target under single
    /// check (block ray + checker square); all-ones when not in check.
    fn checkers_pins(&self) -> (u32, u64, [(u8, u64); 8], usize) {
        let us = self.stm;
        let them = 1 - us;
        let ksq = self.king[us as usize];
        let kf = file_of(ksq);
        let kr = rank_of(ksq);
        let mut n_checkers = 0u32;
        let mut allowed = !0u64;
        let mut pins = [(0u8, 0u64); 8];
        let mut n_pins = 0usize;

        // knight checks
        let mut m = KNIGHT_ATT[ksq as usize];
        let enemy_knight = make_piece(them, KNIGHT);
        while m != 0 {
            let s = m.trailing_zeros() as u8;
            m &= m - 1;
            if self.sq[s as usize] == enemy_knight {
                n_checkers += 1;
                allowed = 1u64 << s;
            }
        }
        // pawn checks
        let dr: i8 = if them == WHITE { -1 } else { 1 };
        for df in [-1i8, 1] {
            let nf = kf + df;
            let nr = kr + dr;
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                let s = (nr * 8 + nf) as u8;
                if self.sq[s as usize] == make_piece(them, PAWN) {
                    n_checkers += 1;
                    allowed = 1u64 << s;
                }
            }
        }
        // slider checks + pins along each ray from the king
        for (dirs, slider) in [(DIAG_DIRS, BISHOP), (ORTHO_DIRS, ROOK)] {
            for (df, dr) in dirs {
                let mut nf = kf + df;
                let mut nr = kr + dr;
                let mut ray = 0u64; // squares between king and current point
                let mut own_block: Option<u8> = None;
                while (0..8).contains(&nf) && (0..8).contains(&nr) {
                    let s = (nr * 8 + nf) as u8;
                    let p = self.sq[s as usize];
                    if p == EMPTY {
                        ray |= 1u64 << s;
                    } else if pcolor(p) == us {
                        if own_block.is_some() {
                            break; // two own pieces: no pin on this ray
                        }
                        own_block = Some(s);
                        ray |= 1u64 << s;
                    } else {
                        if ptype(p) == slider || ptype(p) == QUEEN {
                            match own_block {
                                None => {
                                    n_checkers += 1;
                                    allowed = ray | (1u64 << s);
                                }
                                Some(b) => {
                                    pins[n_pins] = (b, ray | (1u64 << s));
                                    n_pins += 1;
                                }
                            }
                        }
                        break;
                    }
                    nf += df;
                    nr += dr;
                }
            }
        }
        if n_checkers == 0 {
            allowed = !0u64;
        } else if n_checkers > 1 {
            allowed = 0; // double check: no non-king move helps
        }
        (n_checkers, allowed, pins, n_pins)
    }

    pub fn gen_legal(&mut self) -> Vec<Move> {
        let mut out = Vec::with_capacity(64);
        self.gen_legal_into(&mut out);
        out
    }

    /// Alloc-free legal generation into a caller-owned buffer (arm B):
    /// pseudo moves land in `out`, then are filtered IN PLACE preserving the
    /// generation order (order changes would reshuffle stable-sort ties in
    /// the search and shift node counts — the PREREG invariant forbids that).
    /// Cheap path: checkers/pins masks (arm A); king/ep: make/unmake.
    pub fn gen_legal_into(&mut self, out: &mut Vec<Move>) {
        out.clear();
        self.gen_pseudo(out);
        let (_n_chk, allowed, pins, n_pins) = self.checkers_pins();
        let ksq = self.king[self.stm as usize];
        let mut w = 0usize;
        for i in 0..out.len() {
            let m = out[i];
            let is_king = m.from == ksq;
            let is_ep = !is_king
                && ptype(self.sq[m.from as usize]) == PAWN
                && Some(m.to) == self.ep
                && file_of(m.from) != file_of(m.to);
            let legal = if is_king || is_ep {
                // king moves need x-ray-through-king logic, ep needs the
                // double-pawn-discovery check: full make/unmake (rare moves)
                let undo = self.make(m);
                let mover = 1 - self.stm;
                let ok = !self.attacked(self.king[mover as usize], self.stm);
                self.unmake(m, undo);
                ok
            } else {
                let mut ok = allowed & (1u64 << m.to) != 0;
                if ok {
                    for &(pf, pmask) in &pins[..n_pins] {
                        if pf == m.from {
                            ok = pmask & (1u64 << m.to) != 0;
                            break;
                        }
                    }
                }
                ok
            };
            if legal {
                out[w] = m;
                w += 1;
            }
        }
        out.truncate(w);
        debug_assert_eq!(*out, self.gen_legal_ref(), "fast legality diverged: {}", self.fen());
    }

    // NOTE(0009-B): in gen_legal_into, king/ep moves are appended AFTER the
    // masked ones, so the raw generation order differs from gen_legal_ref.
    // This changes ORDER only (the legal SET is verified by the debug_assert
    // above); node counts may shift — gated by the PREREG bench invariant.

    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = self.gen_legal();
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut n = 0;
        for m in moves {
            let undo = self.make(m);
            n += self.perft(depth - 1);
            self.unmake(m, undo);
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Public perft vectors (CPW "Perft Results") — test data, not engine code.
    #[test]
    fn perft_startpos() {
        let mut b = Board::startpos();
        assert_eq!(b.perft(1), 20);
        assert_eq!(b.perft(2), 400);
        assert_eq!(b.perft(3), 8_902);
        assert_eq!(b.perft(4), 197_281);
        assert_eq!(b.perft(5), 4_865_609);
    }

    #[test]
    fn perft_kiwipete() {
        let mut b = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(b.perft(1), 48);
        assert_eq!(b.perft(2), 2_039);
        assert_eq!(b.perft(3), 97_862);
        assert_eq!(b.perft(4), 4_085_603);
    }

    #[test]
    fn perft_pos3_ep_pins() {
        let mut b = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert_eq!(b.perft(4), 43_238);
        assert_eq!(b.perft(5), 674_624);
    }

    #[test]
    fn perft_pos4_promotions() {
        let mut b = Board::from_fen(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        )
        .unwrap();
        assert_eq!(b.perft(3), 9_467);
        assert_eq!(b.perft(4), 422_333);
    }

    #[test]
    fn perft_pos5() {
        let mut b = Board::from_fen(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        )
        .unwrap();
        assert_eq!(b.perft(3), 62_379);
        assert_eq!(b.perft(4), 2_103_487);
    }

    #[test]
    fn perft_pos6() {
        let mut b = Board::from_fen(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        )
        .unwrap();
        assert_eq!(b.perft(3), 89_890);
        assert_eq!(b.perft(4), 3_894_594);
    }

    #[test]
    fn fen_roundtrip() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 3 42",
        ] {
            assert_eq!(Board::from_fen(fen).unwrap().fen(), fen);
        }
    }
}
