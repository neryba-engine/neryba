//! probe 0026: self-play data generation for M2 NNUE.
//!
//! Each thread plays independent games: 8–9 random prolog plies
//! (not recorded), then self-play at fixed nodes/move. A position is
//! written as the line `FEN | score_cp_white_pov | result_white`
//! (bullet-convertible text); filters per PREREG 0026.

use crate::board::{ptype, Board, EMPTY, PAWN, WHITE};
use crate::eval::MATE_THRESHOLD;
use crate::search::Searcher;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

const PROLOG_MIN: u64 = 8;
const ADJ_SCORE: i32 = 2500; // |score| ≥ this for 4 consecutive plies → win
const ADJ_PLIES: i32 = 4;
const MAX_PLIES: u32 = 400;
const SEED_PLIES: u64 = 2; // probe 0034: 0..=2 random plies from the seed FEN

/// probe 0034: pool of starting FENs (env NERYBA_SEED_FENS=<file>, one per
/// line). Empty → old behavior (startpos prolog, bench-invariant).
fn load_seeds() -> Vec<String> {
    match std::env::var("NERYBA_SEED_FENS") {
        Ok(path) => std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// SplitMix64 — reproducible per-thread RNG (date/thread-independent).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub fn run(out_prefix: &str, threads: u32, games_per_thread: u32, seed: u64, nodes: u64) {
    let seeds = Arc::new(load_seeds());
    if !seeds.is_empty() {
        println!("# probe 0034: {} seed FENs, generating from sharp positions", seeds.len());
    }
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let prefix = out_prefix.to_string();
            let seeds = Arc::clone(&seeds);
            std::thread::spawn(move || gen_thread(&prefix, t, games_per_thread, seed, nodes, seeds))
        })
        .collect();
    let mut total = (0u64, 0u64);
    for h in handles {
        let (games, positions) = h.join().expect("datagen thread panicked");
        total.0 += games;
        total.1 += positions;
    }
    println!("done {} games  {} positions", total.0, total.1);
}

fn gen_thread(
    prefix: &str,
    t: u32,
    games: u32,
    seed: u64,
    nodes: u64,
    seeds: Arc<Vec<String>>,
) -> (u64, u64) {
    let mut pos_out = BufWriter::new(
        File::create(format!("{prefix}.t{t}.txt")).expect("create positions file"),
    );
    let mut game_out = BufWriter::new(
        File::create(format!("{prefix}.games.t{t}.csv")).expect("create games file"),
    );
    let mut rng = Rng(seed ^ (t as u64 + 1).wrapping_mul(0xA076_1D64_78BD_642F));
    let mut s = Searcher::new();
    s.node_limit = Some(nodes);
    let mut n_positions = 0u64;
    for _ in 0..games {
        n_positions += play_game(&mut s, &mut rng, &mut pos_out, &mut game_out, &seeds);
    }
    pos_out.flush().expect("flush positions");
    game_out.flush().expect("flush games");
    (games as u64, n_positions)
}

fn play_game(
    s: &mut Searcher,
    rng: &mut Rng,
    pos_out: &mut impl Write,
    game_out: &mut impl Write,
    seeds: &[String],
) -> u64 {
    // prolog: start from a seed FEN (0034) OR startpos + 8–9 plies; mate/stalemate
    // during the prolog → retry
    let (mut b, mut keys) = loop {
        let (mut b, prolog) = if seeds.is_empty() {
            (Board::startpos(), PROLOG_MIN + (rng.next() & 1))
        } else {
            // 0034: random seed FEN + 0..=SEED_PLIES random plies
            let fen = &seeds[rng.below(seeds.len())];
            match Board::from_fen(fen) {
                Ok(board) => (board, rng.next() % (SEED_PLIES + 1)),
                Err(_) => (Board::startpos(), PROLOG_MIN), // broken FEN → fallback
            }
        };
        let mut keys = vec![b.key];
        let mut ok = true;
        for _ in 0..prolog {
            let ms = b.gen_legal();
            if ms.is_empty() {
                ok = false;
                break;
            }
            b.make(ms[rng.below(ms.len())]);
            keys.push(b.key);
        }
        if ok {
            break (b, keys);
        }
    };

    let mut lines: Vec<String> = Vec::new();
    let mut adj_sign = 0i32;
    let mut adj_run = 0i32;
    let mut plies = 0u32;
    let (result_white, reason) = loop {
        if !b.has_legal_move() {
            if b.in_check() {
                break (if b.stm == WHITE { 0.0 } else { 1.0 }, "mate");
            }
            break (0.5, "stalemate");
        }
        if b.halfmove >= 100 {
            break (0.5, "50move");
        }
        if b.is_insufficient_material() {
            break (0.5, "insufficient");
        }
        if keys.iter().filter(|&&k| k == b.key).count() >= 3 {
            break (0.5, "3fold");
        }
        if plies >= MAX_PLIES {
            break (0.5, "maxplies");
        }

        s.rep_keys = keys.clone();
        let (mv, score_stm, _depth) = s.find_best_move(&mut b, 32, None);
        let Some(mv) = mv else {
            break (0.5, "nomove");
        };
        let score_white = if b.stm == WHITE { score_stm } else { -score_stm };

        let is_capture = b.sq[mv.to as usize] != EMPTY
            || (ptype(b.sq[mv.from as usize]) == PAWN && Some(mv.to) == b.ep);
        if !b.in_check() && !is_capture && mv.promo == 0 && score_stm.abs() < MATE_THRESHOLD {
            lines.push(format!("{} | {} | ", b.to_fen(), score_white));
        }

        let sign = score_white.signum();
        if score_white.abs() >= ADJ_SCORE && sign == adj_sign {
            adj_run += 1;
        } else if score_white.abs() >= ADJ_SCORE {
            adj_sign = sign;
            adj_run = 1;
        } else {
            adj_sign = 0;
            adj_run = 0;
        }
        if adj_run >= ADJ_PLIES {
            break (if adj_sign > 0 { 1.0 } else { 0.0 }, "adjudicated");
        }

        b.make(mv);
        keys.push(b.key);
        plies += 1;
    };

    for l in &lines {
        writeln!(pos_out, "{l}{result_white}").expect("write position");
    }
    writeln!(game_out, "{result_white},{plies},{reason}").expect("write game");
    lines.len() as u64
}
