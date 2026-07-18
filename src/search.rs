//! Iterative-deepening negamax alpha-beta — semantic port of sparring v3
//! (backend/engines/v3/search.py, post-audit fixes included).
//! Traceability: ADR-0004 baseline + PREREG 0008 (1:1 semantics, no upgrades).
//!
//! Same structure as v3: draw checks BEFORE the TT probe, node-relative mate
//! scores in the TT, root-relative contempt via ply parity, quiescence with
//! check evasions and a stalemate guard, null-move (R=3 flat), PVS + LMR,
//! 2 killers + history, fresh TT per search call.

use crate::board::*;
use crate::eval::{evaluate, MATE, MATE_THRESHOLD, PIECE_VALUES};
use std::collections::HashMap;
use std::time::Instant;

pub const INF: i32 = 1_000_000_000;
const CONTEMPT: i32 = 25;

const EXACT: u8 = 0;
const LOWER: u8 = 1;
const UPPER: u8 = 2;

struct TtEntry {
    depth: i32,
    flag: u8,
    score: i32,
    best: Option<Move>,
}

/// probe 0068 (env NERYBA_FLAT_TT): packed 16B slot.
/// score: the mate range is encoded via ∓70_000 (PREREG addendum); flag=3 = empty.
#[derive(Clone, Copy)]
#[repr(C)]
struct FlatSlot {
    key: u64,
    score: i16,
    depth: i8,
    flag: u8,
    mv: [u8; 3],
    _pad: u8,
}
const FLAT_EMPTY: u8 = 3;
const FLAT_BITS: usize = 23;

#[inline]
fn pack_score(s: i32) -> i16 {
    debug_assert!(s.abs() < 29_000 || s.abs() >= MATE_THRESHOLD, "score outside the packing convention: {}", s);
    if s >= MATE_THRESHOLD { (s - 70_000) as i16 }
    else if s <= -MATE_THRESHOLD { (s + 70_000) as i16 }
    else { s as i16 }
}
#[inline]
fn unpack_score(e: i16) -> i32 {
    let v = e as i32;
    if v >= 29_000 { v + 70_000 } else if v <= -29_000 { v - 70_000 } else { v }
}

pub struct Searcher {
    tt: HashMap<u64, TtEntry>,
    killers: [[Option<Move>; 2]; 64],
    history: Vec<i32>, // [color][from][to]
    /// zobrist keys of every position since game start (fed by UCI `position`),
    /// extended along the search path — repetition detection sees real history
    pub rep_keys: Vec<u64>,
    deadline: Option<Instant>,
    stop: bool,
    tick: u32,
    /// TM3 (probe 0025): whether the instability-based extension fired
    pub tm3_extended: bool,
    /// probe 0026 (datagen): deterministic node-count stop; UCI/matches
    /// never set it — the game path is unchanged (bench invariance)
    pub node_limit: Option<u64>,
    pub nodes: u64,
    /// per-ply reusable move buffers (arm B of 0009): no Vec alloc per node.
    /// A frame returns its buffer to the pool before recursing at the SAME
    /// ply (quiescence handoff); different plies never collide.
    bufs: Vec<Vec<Move>>,
    /// ablation flags — port of v3's _LMR_ENABLED/_NULLMOVE_ENABLED; both off
    /// = exact alpha-beta whose value is ordering-invariant (parity control)
    pub lmr_enabled: bool,
    pub nullmove_enabled: bool,
    /// probe 0013 GREEN (SPRT accept, +16.3 Elo — research/0013-qsearch-tt/
    /// VERDICT.md): TT probe/store in quiescence, default-on; the field stays
    /// as an ablation knob in the lmr/nullmove pattern
    pub qtt_enabled: bool,
    /// probe 0042 (env NERYBA_LMR_LOG): log reduction formula instead of flat −1/−2.
    /// Read once in new() (not per-node). BASE/MUL constants env-overridable.
    pub lmr_log: bool,
    pub lmr_base: f64,
    pub lmr_mul: f64,
    /// probe 0043 (env NERYBA_CONTHIST): 1-ply continuation history —
    /// (piece,to) of the previous move → (piece,to) of this one; added to
    /// flat history in quiet-move scoring. off = tree bit-for-bit unchanged.
    pub conthist_enabled: bool,
    conthist: Vec<i32>, // [prev_ps][cur_ps], ps = (color*6+ptype-1)*64+to
    /// probe 0044 (env NERYBA_RFP): reverse futility pruning — first
    /// consumer of static eval at interior nodes. off = zero evaluate()
    /// calls in alpha_beta, tree bit-for-bit unchanged.
    pub rfp_enabled: bool,
    pub rfp_margin: i32,
    pub rfp_depth: i32,
    /// probe 0046 (env NERYBA_IIR): internal iterative reductions —
    /// depth−1 at nodes without a TT move. off = tree bit-for-bit unchanged.
    pub iir_enabled: bool,
    pub iir_min: i32,
    /// probe 0055 (env NERYBA_PERSIST): TT+killers+history live across
    /// game moves (process = game: match.py/lichess-bot spawn the engine
    /// per game). off = clear every move, as in the v3 port. Cap — RAM backstop.
    pub persist_enabled: bool,
    pub persist_cap: usize,
    /// probe 0045 (env NERYBA_LMP): late move pruning — skip late quiets
    /// at non-PV nodes once movecount ≥ BASE + MUL·depth². off = bit-for-bit.
    pub lmp_enabled: bool,
    pub lmp_base: f64,
    pub lmp_mul: f64,
    /// probe 0057 (env NERYBA_HIST_AGE): in persist mode, history ÷2 +
    /// killers clear on search entry (fresh outweighs stale). off = full
    /// persist as in 0055.
    pub hist_age: bool,
    /// probe 0049 (env NERYBA_HIST2): history-v2 package — gravity bonus
    /// (saturation MAX=8192), malus for tried quiets, conthist on top.
    /// off = default, bit-for-bit.
    pub hist2: bool,
    /// probe 0059 (env NERYBA_QCONTEMPT): quiescence draw returns
    /// (insufficient/stalemate leaf) get the same root-relative
    /// ±CONTEMPT as alpha_beta. off = 0-return, bit-for-bit.
    pub qcontempt: bool,
    /// probe 0060 (env NERYBA_SINGULAR): singular extensions — +1 ply for
    /// the TT move that alone holds the node (verification with exclusion).
    /// off = the excluded path is dead, tree bit-for-bit.
    pub singular: bool,
    pub sing_min: i32,
    pub sing_margin: i32,
    pub sing_count: u64,
    /// probe 0061 (env NERYBA_SEE_QS): skip SEE<0 captures in quiescence
    /// (non-check nodes). off = bit-for-bit.
    pub see_qs: bool,
    pub see_skips: u64,
    /// probe 0071 (env NERYBA_QCHECKS): quiet checks at qply 0 of quiescence.
    pub qchecks: bool,
    pub qchecks_added: u64,
    /// probe 0068 (env NERYBA_FLAT_TT): flat array instead of the HashMap.
    /// off = HashMap path bit-for-bit.
    flat_on: bool,
    flat_tt: Vec<FlatSlot>,
    /// 0068 frame-F1: counter of cap-clear firings (bundle/structure attribution)
    pub cap_clears: u64,
}

/// probe 0049: gravity history update (saturation, self-decay).
#[inline]
fn apply_bonus(e: &mut i32, bonus: i32) {
    const MAX: i32 = 8192;
    let b = bonus.clamp(-MAX, MAX);
    *e += b - b.abs() * *e / MAX;
}

/// piece-square index for conthist: 0..767.
#[inline]
fn ps_index(pc: u8, to: u8) -> usize {
    ((pcolor(pc) as usize * 6) + ptype(pc) as usize - 1) * 64 + to as usize
}

impl Searcher {
    pub fn new() -> Searcher {
        Searcher {
            tt: HashMap::new(),
            killers: [[None; 2]; 64],
            history: vec![0; 2 * 64 * 64],
            rep_keys: Vec::new(),
            deadline: None,
            tm3_extended: false,
            node_limit: None,
            stop: false,
            tick: 0,
            nodes: 0,
            bufs: Vec::new(),
            lmr_enabled: std::env::var("NERYBA_EXACT").is_err(),
            nullmove_enabled: std::env::var("NERYBA_EXACT").is_err(),
            qtt_enabled: true,
            lmr_log: std::env::var("NERYBA_LMR_LOG").is_ok(),
            lmr_base: std::env::var("NERYBA_LMR_BASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.75),
            lmr_mul: std::env::var("NERYBA_LMR_MUL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.40),
            conthist_enabled: std::env::var("NERYBA_CONTHIST").is_ok(),
            conthist: if std::env::var("NERYBA_CONTHIST").is_ok()
                || std::env::var("NERYBA_HIST2").is_ok()
            {
                vec![0; 768 * 768]
            } else {
                Vec::new()
            },
            // probe 0044 GREEN (+90.9 Elo SPRT accept @524 — research/0044-rfp/
            // VERDICT.md): RFP default-on; NERYBA_RFP_OFF — ablation knob,
            // NERYBA_EXACT disables it too (parity control = exact alpha-beta)
            rfp_enabled: std::env::var("NERYBA_RFP_OFF").is_err()
                && std::env::var("NERYBA_EXACT").is_err(),
            rfp_margin: std::env::var("NERYBA_RFP_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(100),
            rfp_depth: std::env::var("NERYBA_RFP_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(6),
            iir_enabled: std::env::var("NERYBA_IIR").is_ok(),
            iir_min: std::env::var("NERYBA_IIR_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4),
            // probe 0055 GREEN (+27.6 Elo SPRT accept @1844 — research/0055-
            // persistent-state/VERDICT.md): persist default-on;
            // NERYBA_PERSIST_OFF — ablation knob (A/B with the same binary)
            persist_enabled: std::env::var("NERYBA_PERSIST_OFF").is_err(),
            persist_cap: std::env::var("NERYBA_PERSIST_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(4_000_000),
            lmp_enabled: std::env::var("NERYBA_LMP").is_ok(),
            lmp_base: std::env::var("NERYBA_LMP_BASE").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0),
            lmp_mul: std::env::var("NERYBA_LMP_MUL").ok().and_then(|v| v.parse().ok()).unwrap_or(1.3),
            // probe 0057 GREEN (+12.8 SPRT accept @4618 — research/0057-
            // history-aging/VERDICT.md): aging default-on; OFF — ablation
            hist_age: std::env::var("NERYBA_HIST_AGE_OFF").is_err(),
            hist2: std::env::var("NERYBA_HIST2").is_ok(),
            qcontempt: std::env::var("NERYBA_QCONTEMPT").is_ok(),
            singular: std::env::var("NERYBA_SINGULAR").is_ok(),
            sing_min: std::env::var("NERYBA_SING_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(7),
            sing_margin: std::env::var("NERYBA_SING_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
            sing_count: 0,
            // probe 0061 GREEN (+44.9 SPRT accept @1104 — research/0061-see-
            // pruning/VERDICT.md): SEE-qs default-on; OFF — ablation
            see_qs: std::env::var("NERYBA_SEE_QS_OFF").is_err()
                && std::env::var("NERYBA_EXACT").is_err(),
            see_skips: 0,
            qchecks: std::env::var("NERYBA_QCHECKS").is_ok(),
            qchecks_added: 0,
            // probe 0068 GREEN (+17.4 STC ACCEPT + LTC sign +27.9 —
            // research/0068-flat-tt/VERDICT.md): flat-TT default-on
            flat_on: std::env::var("NERYBA_FLAT_TT_OFF").is_err(),
            flat_tt: if std::env::var("NERYBA_FLAT_TT_OFF").is_err() {
                vec![FlatSlot { key: 0, score: 0, depth: 0, flag: FLAT_EMPTY, mv: [0; 3], _pad: 0 }; 1 << FLAT_BITS]
            } else {
                Vec::new()
            },
            cap_clears: 0,
        }
    }

    #[inline]
    fn take_buf(&mut self, ply: usize) -> Vec<Move> {
        if self.bufs.len() <= ply {
            self.bufs.resize_with(ply + 1, || Vec::with_capacity(64));
        }
        std::mem::take(&mut self.bufs[ply])
    }

    #[inline]
    fn put_buf(&mut self, ply: usize, v: Vec<Move>) {
        self.bufs[ply] = v;
    }

    /// probe 0068: unified probe over both TT backends.
    #[inline]
    fn tt_probe(&self, key: u64) -> Option<(i32, u8, i32, Option<Move>)> {
        if self.flat_on {
            let s = &self.flat_tt[(key as usize) & ((1 << FLAT_BITS) - 1)];
            if s.flag != FLAT_EMPTY && s.key == key {
                let best = if s.mv == [0; 3] { None } else {
                    Some(Move { from: s.mv[0], to: s.mv[1], promo: s.mv[2] })
                };
                return Some((s.depth as i32, s.flag, unpack_score(s.score), best));
            }
            None
        } else {
            self.tt.get(&key).map(|e| (e.depth, e.flag, e.score, e.best))
        }
    }

    /// probe 0068: unified store (flat = always-replace).
    #[inline]
    fn tt_store(&mut self, key: u64, depth: i32, flag: u8, score: i32, best: Option<Move>) {
        if self.flat_on {
            let idx = (key as usize) & ((1 << FLAT_BITS) - 1);
            let mv = best.map(|m| [m.from, m.to, m.promo]).unwrap_or([0; 3]);
            self.flat_tt[idx] = FlatSlot {
                key, score: pack_score(score), depth: depth as i8, flag, mv, _pad: 0,
            };
        } else {
            self.tt.insert(key, TtEntry { depth, flag, score, best });
        }
    }

    #[inline]
    fn tt_clear_all(&mut self) {
        if self.flat_on {
            self.flat_tt.iter_mut().for_each(|s| s.flag = FLAT_EMPTY);
        } else {
            self.tt.clear();
        }
    }

    #[inline]
    fn tt_over_cap(&self) -> bool {
        // flat: natural eviction, the cap does not apply (PREREG 0068)
        !self.flat_on && self.tt.len() > self.persist_cap
    }

    fn reset_ordering(&mut self) {
        self.killers = [[None; 2]; 64];
        self.history.iter_mut().for_each(|h| *h = 0);
        self.conthist.iter_mut().for_each(|h| *h = 0);
    }

    #[inline]
    fn check_time(&mut self) {
        self.tick += 1;
        if self.tick >= 1024 {
            self.tick = 0;
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    self.stop = true;
                }
            }
            if let Some(nl) = self.node_limit {
                if self.nodes >= nl {
                    self.stop = true;
                }
            }
        }
    }

    #[inline]
    fn hist_index(color: u8, m: Move) -> usize {
        ((color as usize) * 64 + m.from as usize) * 64 + m.to as usize
    }

    pub fn is_capture(b: &Board, m: Move) -> bool {
        b.sq[m.to as usize] != EMPTY
            || (ptype(b.sq[m.from as usize]) == PAWN
                && Some(m.to) == b.ep
                && file_of(m.from) != file_of(m.to))
    }

    fn mvv_lva(b: &Board, m: Move) -> i32 {
        let mut score = 0;
        if Self::is_capture(b, m) {
            let victim = ptype(b.sq[m.to as usize]);
            let victim_val = if victim == 0 { PIECE_VALUES[PAWN as usize] } else { PIECE_VALUES[victim as usize] };
            let attacker_val = PIECE_VALUES[ptype(b.sq[m.from as usize]) as usize];
            score += 10 * victim_val - attacker_val;
        }
        if m.promo != 0 {
            score += PIECE_VALUES[m.promo as usize];
        }
        score
    }

    /// v3 _order_moves: TT move, captures by MVV-LVA, killers, history quiets.
    /// probe 0043: quiets are additionally weighted by conthist[prev][cur] (flag on).
    fn order_moves(&self, b: &Board, moves: &mut [Move], tt_move: Option<Move>, ply: usize, prev: Option<usize>) {
        let k = if ply < 64 { self.killers[ply] } else { [None; 2] };
        let color = b.stm;
        moves.sort_by_key(|&m| {
            let s = if Some(m) == tt_move {
                1_000_000
            } else if Self::is_capture(b, m) || m.promo != 0 {
                100_000 + Self::mvv_lva(b, m)
            } else if Some(m) == k[0] {
                90_000
            } else if Some(m) == k[1] {
                80_000
            } else {
                let mut h = self.history[Self::hist_index(color, m)];
                if let Some(p) = prev {
                    h += self.conthist[p * 768 + ps_index(b.sq[m.from as usize], m.to)];
                }
                h
            };
            -s // sort_by_key ascending -> negate for descending
        });
    }

    fn is_repetition2(&self, key: u64) -> bool {
        self.rep_keys.iter().filter(|&&k| k == key).count() >= 2
    }

    fn quiescence(&mut self, b: &mut Board, mut alpha: i32, beta: i32, ply: i32, qply: i32) -> i32 {
        self.check_time();
        if self.stop {
            return alpha;
        }
        self.nodes += 1;

        // probe 0013: TT in quiescence (depth=0 entries; alpha_beta entries
        // with depth>=0 always qualify). Mate scores rebased as usual.
        let alpha_orig = alpha;
        if self.qtt_enabled {
            if let Some((_qd, q_flag, q_score, _qb)) = self.tt_probe(b.key) {
                let mut score = q_score;
                if score >= MATE_THRESHOLD {
                    score -= ply;
                } else if score <= -MATE_THRESHOLD {
                    score += ply;
                }
                match q_flag {
                    EXACT => return score,
                    LOWER => {
                        if score >= beta {
                            return score;
                        }
                    }
                    _ => {
                        if score <= alpha {
                            return score;
                        }
                    }
                }
            }
        }

        let mut moves = self.take_buf(ply as usize);
        b.gen_legal_into(&mut moves); // one generation serves mate/stalemate/captures
        if b.in_check() {
            if moves.is_empty() {
                self.put_buf(ply as usize, moves);
                return -(MATE - ply);
            }
        } else {
            // probe 0059 (env NERYBA_QCONTEMPT): qsearch draw leaves signed
            // by root parity (same convention as alpha_beta)
            let draw_leaf = if self.qcontempt {
                if ply % 2 == 0 { -CONTEMPT } else { CONTEMPT }
            } else {
                0
            };
            if b.is_insufficient_material() {
                self.put_buf(ply as usize, moves);
                return draw_leaf;
            }
            if moves.is_empty() {
                self.put_buf(ply as usize, moves);
                return draw_leaf; // stalemate leaf
            }
            let stand_pat = if b.stm == WHITE { evaluate(b) } else { -evaluate(b) };
            if stand_pat >= beta {
                self.put_buf(ply as usize, moves);
                self.qtt_store(b.key, beta, LOWER, ply);
                return beta;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
            if self.qchecks && qply == 0 {
                // probe 0071: captures + quiet checks (make-filter, qply 0 only)
                let mut keep = Vec::with_capacity(moves.len());
                for &m in moves.iter() {
                    if Self::is_capture(b, m) {
                        keep.push(m);
                    } else if m.promo == 0 {
                        let undo = b.make(m);
                        let gives = b.in_check();
                        b.unmake(m, undo);
                        if gives {
                            keep.push(m);
                            self.qchecks_added += 1;
                        }
                    }
                }
                moves.clear();
                moves.extend_from_slice(&keep);
            } else {
                moves.retain(|&m| Self::is_capture(b, m));
            }
            moves.sort_by_key(|&m| -Self::mvv_lva(b, m));
        }
        let mut result = None;
        let evasion = b.in_check();
        for &m in moves.iter() {
            // probe 0061: a losing capture at a quiet node — skip (PREREG)
            if self.see_qs && !evasion && Self::is_capture(b, m) && b.see(m) < 0 {
                self.see_skips += 1;
                continue;
            }
            let undo = b.make(m);
            self.rep_keys.push(b.key);
            let score = -self.quiescence(b, -beta, -alpha, ply + 1, qply + 1);
            self.rep_keys.pop();
            b.unmake(m, undo);
            if score >= beta {
                result = Some(beta);
                break;
            }
            if score > alpha {
                alpha = score;
            }
        }
        self.put_buf(ply as usize, moves);
        let ret = result.unwrap_or(alpha);
        let flag = if result.is_some() {
            LOWER
        } else if alpha > alpha_orig {
            EXACT
        } else {
            UPPER
        };
        self.qtt_store(b.key, ret, flag, ply);
        ret
    }

    /// probe 0013: depth-0 store that never clobbers a real alpha_beta entry.
    #[inline]
    fn qtt_store(&mut self, key: u64, score: i32, flag: u8, ply: i32) {
        if !self.qtt_enabled || self.stop {
            return;
        }
        if let Some((e_depth, _f, _s, _b)) = self.tt_probe(key) {
            if e_depth > 0 {
                return; // deeper info wins
            }
        }
        let mut s = score;
        if s >= MATE_THRESHOLD {
            s += ply;
        } else if s <= -MATE_THRESHOLD {
            s -= ply;
        }
        self.tt_store(key, 0, flag, s, None);
    }

    fn alpha_beta(&mut self, b: &mut Board, mut depth: i32, mut alpha: i32, mut beta: i32, ply: i32, prev: Option<usize>, excluded: Option<Move>) -> i32 {
        self.check_time();
        if self.stop {
            return alpha;
        }
        self.nodes += 1;

        // Draw conditions FIRST (before TT): the TT key carries no clock or
        // history, so a clean-path EXACT entry must not mask a drawish path.
        // Contempt is root-relative: stm == root exactly when ply is even.
        let contempt = if ply % 2 == 0 { -CONTEMPT } else { CONTEMPT };
        if b.is_insufficient_material() {
            return contempt;
        }
        if b.halfmove >= 100 {
            let mate_now = b.in_check() && !b.has_legal_move();
            if !mate_now {
                return contempt;
            }
        }
        if b.halfmove >= 4 && self.is_repetition2(b.key) {
            return contempt;
        }

        let key = b.key;
        let mut tt_move = None;
        let mut tt_depth = -1;
        let mut tt_flag = UPPER;
        let mut tt_score_raw = 0;
        // probe 0060: during excluded verification the TT is neither read
        // (cutoff/narrow) nor written — the sub-search verdict does not belong to the node key
        if excluded.is_none() { if let Some((e_depth, e_flag, e_score, e_best)) = self.tt_probe(key) {
            tt_move = e_best;
            tt_depth = e_depth;
            tt_flag = e_flag;
            tt_score_raw = e_score;
            if e_depth >= depth {
                // mate scores are stored node-relative; rebase to this ply
                let mut score = e_score;
                if score >= MATE_THRESHOLD {
                    score -= ply;
                } else if score <= -MATE_THRESHOLD {
                    score += ply;
                }
                match e_flag {
                    EXACT => return score,
                    LOWER => {
                        if score > alpha {
                            alpha = score;
                        }
                    }
                    _ => {
                        if score < beta {
                            beta = score;
                        }
                    }
                }
                if alpha >= beta {
                    return score;
                }
            }
        } }
        // probe 0046 (env NERYBA_IIR): a node without a TT move has the
        // worst ordering — reduce depth (classic IIR form, conditions in PREREG)
        if self.iir_enabled && depth >= self.iir_min && tt_move.is_none() {
            depth -= 1;
        }

        // AFTER the TT narrowed the window: fail-low vs the raised alpha is
        // only an upper bound, not EXACT.
        let alpha_orig = alpha;

        let mut moves = self.take_buf(ply as usize);
        b.gen_legal_into(&mut moves);
        if moves.is_empty() {
            self.put_buf(ply as usize, moves);
            return if b.in_check() { -(MATE - ply) } else { 0 };
        }
        if depth == 0 {
            // return the buffer BEFORE quiescence re-borrows the same ply slot
            self.put_buf(ply as usize, moves);
            return self.quiescence(b, alpha, beta, ply, 0);
        }

        let in_check = b.in_check();
        // probe 0044 (env NERYBA_RFP): reverse futility — non-PV, not in check,
        // shallow: static eval exceeds beta by margin·depth → leaf without search.
        // Conditions and constants fixed in PREREG 0044 before the run.
        if self.rfp_enabled
            && depth <= self.rfp_depth
            && !in_check
            && beta - alpha == 1
            && beta.abs() < MATE_THRESHOLD
        {
            let eval = if b.stm == WHITE { evaluate(b) } else { -evaluate(b) };
            if eval - self.rfp_margin * depth >= beta {
                self.put_buf(ply as usize, moves);
                return eval;
            }
        }

        if self.nullmove_enabled && depth >= 3 && !in_check && b.has_non_pawn_material() {
            let saved = b.make_null();
            self.rep_keys.push(b.key);
            // 0043: after a null move there is no prev (conthist chain breaks)
            let null_score = -self.alpha_beta(b, depth - 3, -beta, -beta + 1, ply + 1, None, None);
            self.rep_keys.pop();
            b.unmake_null(saved);
            if null_score >= beta {
                return beta;
            }
        }

        self.order_moves(b, &mut moves, tt_move, ply as usize, prev);

        // probe 0060 (env NERYBA_SINGULAR): verification with exclusion —
        // conditions and constants fixed in the PREREG
        let mut sing_ext = 0;
        if self.singular
            && excluded.is_none()
            && ply > 0
            && depth >= self.sing_min
            && tt_move.is_some()
            && tt_depth >= depth - 3
            && (tt_flag == LOWER || tt_flag == EXACT)
        {
            let mut ts = tt_score_raw;
            if ts >= MATE_THRESHOLD { ts -= ply; } else if ts <= -MATE_THRESHOLD { ts += ply; }
            if ts.abs() < MATE_THRESHOLD {
                let s_beta = ts - self.sing_margin * depth;
                let v = self.alpha_beta(b, (depth - 1) / 2, s_beta - 1, s_beta, ply, prev, tt_move);
                if !self.stop && v < s_beta {
                    sing_ext = 1;
                    self.sing_count += 1;
                }
            }
        }

        let mut best = -INF;
        let mut best_move = None;
        let mut move_count = 0;
        for mi in 0..moves.len() {
            let m = moves[mi];
            if Some(m) == excluded {
                continue; // probe 0060: the excluded move stays outside the counters
            }
            move_count += 1;
            let m_ext = if sing_ext > 0 && Some(m) == tt_move { sing_ext } else { 0 };
            let is_cap = Self::is_capture(b, m);
            // probe 0045 (env NERYBA_LMP): late quiet at a non-PV node — skip
            // (continue, not break: captures later in the list are still tried)
            if self.lmp_enabled
                && beta - alpha == 1
                && !in_check
                && !is_cap
                && m.promo == 0
                && move_count as f64 >= self.lmp_base + self.lmp_mul * (depth * depth) as f64
            {
                continue;
            }
            let undo = b.make(m);
            self.rep_keys.push(b.key);
            let gives_check = b.in_check();
            // 0043: piece-square of this move (after make — promotion accounted for)
            let child_prev = if self.conthist_enabled || self.hist2 {
                Some(ps_index(b.sq[m.to as usize], m.to))
            } else {
                None
            };

            let score = if move_count == 1 {
                -self.alpha_beta(b, depth - 1 + m_ext, -beta, -alpha, ply + 1, child_prev, None)
            } else {
                let mut reduction = 0;
                if self.lmr_enabled && depth >= 3 && move_count >= 4 && !is_cap && !gives_check && m.promo == 0 && !in_check {
                    reduction = if self.lmr_log {
                        // probe 0042: r = BASE + ln(d)·ln(mc)·MUL, clamp(1, depth−2)
                        let r = (self.lmr_base
                            + (depth as f64).ln() * (move_count as f64).ln() * self.lmr_mul)
                            as i32;
                        r.clamp(1, depth - 2)
                    } else {
                        if move_count >= 8 { 2 } else { 1 }
                    };
                }
                let mut s =
                    -self.alpha_beta(b, depth - 1 + m_ext - reduction, -alpha - 1, -alpha, ply + 1, child_prev, None);
                if reduction > 0 && s > alpha {
                    s = -self.alpha_beta(b, depth - 1 + m_ext, -alpha - 1, -alpha, ply + 1, child_prev, None);
                }
                if alpha < s && s < beta {
                    s = -self.alpha_beta(b, depth - 1 + m_ext, -beta, -alpha, ply + 1, child_prev, None);
                }
                s
            };
            self.rep_keys.pop();
            b.unmake(m, undo);

            if score > best {
                best = score;
                best_move = Some(m);
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                if !is_cap && m.promo == 0 {
                    if (ply as usize) < 64 {
                        let kl = &mut self.killers[ply as usize];
                        if kl[0] != Some(m) {
                            kl[1] = kl[0];
                            kl[0] = Some(m);
                        }
                    }
                    if self.hist2 {
                        // probe 0049: gravity bonus for the winner + malus
                        // for this node's tried quiets (b already after unmake)
                        let d2 = depth * depth;
                        apply_bonus(&mut self.history[Self::hist_index(b.stm, m)], d2);
                        if let Some(p) = prev {
                            apply_bonus(
                                &mut self.conthist[p * 768 + ps_index(b.sq[m.from as usize], m.to)],
                                d2,
                            );
                        }
                        for qi in 0..mi {
                            let q = moves[qi];
                            if !Self::is_capture(b, q) && q.promo == 0 {
                                apply_bonus(&mut self.history[Self::hist_index(b.stm, q)], -d2);
                                if let Some(p) = prev {
                                    apply_bonus(
                                        &mut self.conthist
                                            [p * 768 + ps_index(b.sq[q.from as usize], q.to)],
                                        -d2,
                                    );
                                }
                            }
                        }
                    } else {
                        self.history[Self::hist_index(b.stm, m)] += depth * depth;
                        // 0043: same formula for conthist (b already after unmake)
                        if let Some(p) = prev {
                            self.conthist[p * 768 + ps_index(b.sq[m.from as usize], m.to)] +=
                                depth * depth;
                        }
                    }
                }
                break;
            }
        }
        self.put_buf(ply as usize, moves);

        if !self.stop && excluded.is_none() {
            let flag = if best <= alpha_orig {
                UPPER
            } else if best >= beta {
                LOWER
            } else {
                EXACT
            };
            // store mate scores node-relative
            let mut tt_score = best;
            if tt_score >= MATE_THRESHOLD {
                tt_score += ply;
            } else if tt_score <= -MATE_THRESHOLD {
                tt_score -= ply;
            }
            self.tt_store(key, depth, flag, tt_score, best_move);
        }
        best
    }

    fn root(
        &mut self,
        b: &mut Board,
        depth: i32,
        prev_best: Option<Move>,
        window: (i32, i32),
    ) -> (Option<Move>, i32) {
        let (alpha0, beta) = window;
        let mut alpha = alpha0;
        let mut best = -INF;
        let mut best_move = None;
        let mut moves = b.gen_legal();
        moves.sort_by_key(|&m| -Self::mvv_lva(b, m));
        if let Some(pb) = prev_best {
            if let Some(pos) = moves.iter().position(|&m| m == pb) {
                moves.remove(pos);
                moves.insert(0, pb);
            }
        }
        for m in moves {
            let undo = b.make(m);
            self.rep_keys.push(b.key);
            // 0043: the root move = prev for ply 1
            let child_prev = if self.conthist_enabled || self.hist2 {
                Some(ps_index(b.sq[m.to as usize], m.to))
            } else {
                None
            };
            let score = -self.alpha_beta(b, depth - 1, -beta, -alpha, 1, child_prev, None);
            self.rep_keys.pop();
            b.unmake(m, undo);
            if self.stop {
                break;
            }
            if score > best {
                best = score;
                best_move = Some(m);
                if score > alpha {
                    alpha = score;
                }
            }
        }
        (best_move, best)
    }

    /// Iterative deepening. Returns (move, stm-relative score, completed depth).
    /// `movetime` aborts the running iteration and keeps the last completed one.
    pub fn find_best_move(
        &mut self,
        b: &mut Board,
        max_depth: i32,
        movetime: Option<f64>,
    ) -> (Option<Move>, i32, i32) {
        self.find_best_move_tm(b, max_depth, movetime, None)
    }

    /// TM3 (probe 0025): `soft` — base budget; `hard` — ceiling for a single
    /// extension when at the soft limit the best move is unstable between
    /// the last completed iterations (easy/hard move).
    pub fn find_best_move_tm(
        &mut self,
        b: &mut Board,
        max_depth: i32,
        soft: Option<f64>,
        hard: Option<f64>,
    ) -> (Option<Move>, i32, i32) {
        // probe 0055 (env NERYBA_PERSIST): state lives across game moves;
        // cap-clear — backstop against TT bloat (PREREG 0055)
        if !self.persist_enabled || self.tt_over_cap() {
            if self.tt_over_cap() {
                self.cap_clears += 1; // 0068-F1: measurable attribution
            }
            self.tt_clear_all();
            self.reset_ordering();
        } else if self.hist_age {
            // probe 0057: decay history + reset ply-relative killers
            self.history.iter_mut().for_each(|h| *h /= 2);
            self.killers = [[None; 2]; 64];
        }
        self.nodes = 0;
        self.stop = false;
        self.tick = 0;
        self.tm3_extended = false;
        let t0 = Instant::now();
        self.deadline = soft.map(|t| t0 + std::time::Duration::from_secs_f64(t));
        let hard_deadline = hard.map(|t| t0 + std::time::Duration::from_secs_f64(t));

        // aspiration windows (probe 0011, env-gated for the A/B match):
        // narrow window around the previous iteration's score from d>=4,
        // immediate full-window re-search on fail-low/high.
        let aspiration = std::env::var("NERYBA_ASPIRATION").map_or(false, |v| v != "0");
        const ASP: i32 = 50;

        // probe 0037 (env NERYBA_INST_TM): time reallocation by root
        // instability (winprob spread across iterations = 0036 signal), budget-neutral
        let inst_tm = std::env::var("NERYBA_INST_TM").is_ok();
        let mut winprobs: Vec<f64> = Vec::new();

        // TM4 draw-shortcut (probe 0038, env NERYBA_TM4): stop ID early
        // when the best move is stable in the draw zone — bank the unspent soft.
        // Clock mode only (hard is set); consts fixed in PREREG 0038.
        let tm4 = std::env::var("NERYBA_TM4").is_ok() && hard.is_some();
        // probe 0039: env-parameterized (defaults = 0038: STAB 4 / DRAW 40 / MIN_D 8)
        let envi = |k: &str, d: i32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let tm4_stab = envi("NERYBA_TM4_STAB", 4);
        let tm4_draw_cp = envi("NERYBA_TM4_DRAW", 40);
        let tm4_min_d = envi("NERYBA_TM4_MIND", 8);
        let mut tm4_stable = 0i32;

        let mut best: (Option<Move>, i32, i32) = (None, 0, 0);
        let mut prev_best: Option<Move> = None; // best of the completed d-1 iteration
        // probe 0087 (PREREG addendum B): per-iteration info for the label generator.
        // Output-only, default OFF — tree/nodes untouched (bit-exact bench gate).
        let iter_info = std::env::var("NERYBA_ITER_INFO").is_ok();
        for d in 1..=max_depth {
            let window = if aspiration && d >= 4 && best.0.is_some() {
                (best.1 - ASP, best.1 + ASP)
            } else {
                (-INF, INF)
            };
            let (mut mv, mut score) = self.root(b, d, best.0, window);
            if !self.stop && window.0 > -INF && (score <= window.0 || score >= window.1) {
                // fail-low/high: the truth is outside the guess — full re-search
                let r = self.root(b, d, best.0, (-INF, INF));
                mv = r.0;
                score = r.1;
            }
            if self.stop {
                // TM3: soft limit exhausted mid-iteration; if the best move
                // is unstable between the two last completed iterations —
                // ONE extension up to hard and a retry of this depth (warm TT)
                if let Some(hd) = hard_deadline {
                    let unstable = match (best.0, prev_best) {
                        (Some(a), Some(p)) => a != p,
                        _ => true,
                    };
                    if !self.tm3_extended && unstable && Instant::now() < hd {
                        self.tm3_extended = true;
                        self.stop = false;
                        self.deadline = Some(hd);
                        let r = self.root(b, d, best.0, (-INF, INF));
                        if !self.stop && r.0.is_some() {
                            prev_best = best.0;
                            best = (r.0, r.1, d);
                        }
                    }
                }
                break; // the aborted iteration is not used
            }
            if mv.is_some() {
                prev_best = best.0;
                best = (mv, score, d);
                if iter_info {
                    let ss = if score.abs() >= MATE_THRESHOLD {
                        let plies = MATE - score.abs();
                        let mm = std::cmp::max(1, (plies + 1) / 2);
                        format!("mate {}", if score > 0 { mm } else { -mm })
                    } else {
                        format!("cp {score}")
                    };
                    println!(
                        "info depth {} score {} nodes {} pv {}",
                        d,
                        ss,
                        self.nodes,
                        mv.map(|m| m.uci()).unwrap_or_default()
                    );
                }
            }
            // TM4 (0038): best stable for ≥STAB iterations in the draw zone at
            // depth ≥MIN_D → stop (unspent soft = increment bank)
            if tm4 && best.0.is_some() {
                if best.0 == prev_best {
                    tm4_stable += 1;
                } else {
                    tm4_stable = 0;
                }
                if tm4_stable >= tm4_stab && best.1.abs() <= tm4_draw_cp && best.2 >= tm4_min_d {
                    break;
                }
            }
            // probe 0037: after a completed iteration — modulate the soft deadline
            // by winprob-curve instability (more time on shaky, less on stable)
            if inst_tm {
                if let (Some(ss), Some(hd_s)) = (soft, hard) {
                    let wp = 1.0 / (1.0 + 10f64.powf(-(score as f64) / 400.0));
                    winprobs.push(wp);
                    if winprobs.len() >= 3 {
                        let k = winprobs.len().min(4);
                        let rec = &winprobs[winprobs.len() - k..];
                        let mean = rec.iter().sum::<f64>() / k as f64;
                        let var = rec.iter().map(|w| (w - mean) * (w - mean)).sum::<f64>() / k as f64;
                        let inst = var.sqrt();
                        const HI: f64 = 0.06;
                        const LO: f64 = 0.02;
                        let new_soft = if inst > HI {
                            hd_s.min(ss * 1.8) // shaky → dig deeper (up to hard)
                        } else if inst < LO {
                            ss * 0.7 // stable → bank the time
                        } else {
                            ss
                        };
                        self.deadline = Some(t0 + std::time::Duration::from_secs_f64(new_soft));
                    }
                }
            }
            if score.abs() >= MATE_THRESHOLD {
                break;
            }
        }
        self.deadline = None;
        if best.0.is_none() {
            best.0 = b.gen_legal().first().copied();
        }
        best
    }

    /// Principal variation from the TT chain (display only).
    pub fn pv(&self, b: &Board, first: Option<Move>, depth: i32) -> Vec<String> {
        let mut out = Vec::new();
        let mut work = b.clone();
        let mut mv = first;
        for _ in 0..depth {
            let m = match mv {
                Some(m) if work.gen_legal().contains(&m) => m,
                _ => break,
            };
            out.push(m.uci());
            work.make(m);
            mv = self.tt_probe(work.key).and_then(|(_d, _f, _s, b)| b);
        }
        out
    }
}
