//! UCI protocol (ADR-0004 baseline). Same command surface as sparring/uci_v3.py
//! so harness/match.py drives both identically.

use crate::board::*;
use crate::eval::{MATE, MATE_THRESHOLD};
use crate::search::Searcher;
use std::io::{BufRead, Write};

const MAX_DEPTH: i32 = 64;
const DEFAULT_DEPTH: i32 = 6;

struct State {
    board: Board,
    keys: Vec<u64>, // position keys since game start (repetition detection)
    /// probe 0055: Searcher lives in State so NERYBA_PERSIST can carry
    /// TT/killers/history across moves; without the flag find_best_move_tm
    /// clears them on entry — bit-for-bit the same as a per-go Searcher::new()
    searcher: Searcher,
}

impl State {
    fn new() -> State {
        let board = Board::startpos();
        let keys = vec![board.key];
        State { board, keys, searcher: Searcher::new() }
    }

    fn set_position(&mut self, tokens: &[&str]) {
        let mut i = 0;
        if tokens.get(0) == Some(&"startpos") {
            self.board = Board::startpos();
            i = 1;
        } else if tokens.get(0) == Some(&"fen") {
            let fen = tokens[1..tokens.len().min(7)].join(" ");
            if let Ok(b) = Board::from_fen(&fen) {
                self.board = b;
            }
            i = 7.min(tokens.len());
        }
        self.keys = vec![self.board.key];
        if tokens.get(i) == Some(&"moves") {
            for u in &tokens[i + 1..] {
                if let Some(m) = Move::from_uci(u) {
                    if self.board.gen_legal().contains(&m) {
                        self.board.make(m);
                        self.keys.push(self.board.key);
                    }
                }
            }
        }
    }
}

fn budget_seconds(args: &std::collections::HashMap<String, i64>, stm: u8) -> Option<f64> {
    let (t, inc) = if stm == WHITE {
        (args.get("wtime"), args.get("winc"))
    } else {
        (args.get("btime"), args.get("binc"))
    };
    // TM1 (v3 legacy): 1/30 slice + 0.8*inc. TM2 (probe 0024, env NERYBA_TM2):
    // increment cushion + LAG_RESERVE against flagging — formula in PREREG 0024.
    t.map(|&remain| {
        let inc_s = inc.copied().unwrap_or(0) as f64 / 1000.0;
        let remain_s = remain as f64 / 1000.0;
        if std::env::var("NERYBA_TM2").is_ok() {
            let eff = (remain_s - 1.5).max(0.0);
            (eff / 40.0 + 0.75 * inc_s).min(eff / 4.0).max(0.010)
        } else {
            (remain_s / 30.0 + inc_s * 0.8).max(0.010)
        }
    })
}

pub fn run() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let mut st = State::new();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let cmd = match tokens.first() {
            Some(c) => *c,
            None => continue,
        };
        match cmd {
            "uci" => {
                let _ = writeln!(out, "id name Neryba 0.0.1");
                let _ = writeln!(out, "id author Dmytro Dehtiarov");
                let _ = writeln!(out, "uciok");
            }
            "isready" => {
                let _ = writeln!(out, "readyok");
            }
            "ucinewgame" => {
                st = State::new();
            }
            "position" => st.set_position(&tokens[1..]),
            "go" => {
                let mut args = std::collections::HashMap::new();
                let mut it = tokens[1..].iter();
                while let Some(tok) = it.next() {
                    if ["depth", "movetime", "wtime", "btime", "winc", "binc", "movestogo", "nodes"]
                        .contains(tok)
                    {
                        if let Some(v) = it.next().and_then(|v| v.parse::<i64>().ok()) {
                            args.insert(tok.to_string(), v);
                        }
                    }
                }
                // TM3 (probe 0025 GREEN, +38..+67 Elo): default in clock mode;
                // ablation knob NERYBA_TM1=1 restores the old formula (A/B)
                let tm3 = std::env::var("NERYBA_TM1").is_err()
                    && args.get("depth").is_none()
                    && args.get("movetime").is_none();
                let (depth, movetime, hard) = if let Some(&d) = args.get("depth") {
                    ((d as i32).clamp(1, MAX_DEPTH), None, None)
                } else if let Some(&mt) = args.get("movetime") {
                    (MAX_DEPTH, Some(mt as f64 / 1000.0), None)
                } else if args.contains_key("nodes") {
                    // probe 0070: the node limit drives the stop — depth wide open
                    (MAX_DEPTH, None, None)
                } else if let Some(t) = budget_seconds(&args, st.board.stm) {
                    if tm3 {
                        // TM1 base; opening discount ×0.5 (fullmove ≤ 10);
                        // LAG_RESERVE 1.5s: neither soft nor hard dips into the reserve
                        let remain = *if st.board.stm == WHITE {
                            args.get("wtime").unwrap_or(&0)
                        } else {
                            args.get("btime").unwrap_or(&0)
                        } as f64 / 1000.0;
                        let cap = (remain - 1.5).max(0.010);
                        let base = if st.board.fullmove <= 10 { t * 0.5 } else { t };
                        let soft = base.min(cap);
                        (MAX_DEPTH, Some(soft), Some((soft * 1.8).min(cap)))
                    } else {
                        (MAX_DEPTH, Some(t), None)
                    }
                } else {
                    (DEFAULT_DEPTH, None, None)
                };

                let searcher = &mut st.searcher;
                // probe 0070: `go nodes N` -> node_limit (the core had it from the
                // datagen path, UCI never wired it). Resetting it every move is
                // mandatory: the persist Searcher (0055) lives across moves.
                searcher.node_limit = args.get("nodes").map(|&n| n as u64);
                searcher.rep_keys = st.keys.clone();
                let mut b = st.board.clone();
                let (mv, score, reached) = searcher.find_best_move_tm(&mut b, depth, movetime, hard);
                if st.searcher.cap_clears > 0 {
                    let _ = writeln!(out, "info string capclears {}", st.searcher.cap_clears);
                }
                let searcher = &mut st.searcher;
                if searcher.tm3_extended {
                    let _ = writeln!(out, "info string tm3 extend"); // positive control for 0025
                }
                match mv {
                    Some(m) => {
                        let score_str = if score.abs() >= MATE_THRESHOLD {
                            let plies = MATE - score.abs();
                            let mm = std::cmp::max(1, (plies + 1) / 2);
                            format!("mate {}", if score > 0 { mm } else { -mm })
                        } else {
                            format!("cp {score}")
                        };
                        let pv = searcher.pv(&st.board, Some(m), reached.max(1));
                        let _ = writeln!(
                            out,
                            "info depth {} score {} nodes {} pv {}",
                            reached.max(1),
                            score_str,
                            searcher.nodes,
                            if pv.is_empty() { m.uci() } else { pv.join(" ") }
                        );
                        let _ = writeln!(out, "bestmove {}", m.uci());
                    }
                    None => {
                        let _ = writeln!(out, "bestmove 0000");
                    }
                }
            }
            "stop" => {}
            "quit" => break,
            _ => {}
        }
        let _ = out.flush();
    }
}
