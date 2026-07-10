//! Neryba CLI (probe 0008 / ADR-0004).
//!
//!   neryba                          UCI mode (default)
//!   neryba perft <depth> [fen]      node count
//!   neryba divide <depth> [fen]     per-root-move counts
//!   neryba bench [depth]            search NPS on a fixed suite (M1 metric)

mod board;
mod datagen;
mod eval;
mod nnue;
mod nnue2;
mod search;
mod uci;
mod zobrist;

use board::Board;
use std::time::Instant;

// Fixed public middlegame FENs for the NPS bench (same spirit as probe suite).
const BENCH_FENS: [&str; 5] = [
    "r1bqkb1r/ppp2pp1/5n1p/3p4/3PP3/8/PPPN1PPP/RNBQK2R w KQkq - 2 8",
    "r2q1rk1/pp2ppbp/2npb1p1/2p5/2P1PPP1/2NPBN1P/PP2Q3/R3K1R1 b Q - 0 13",
    "r1bq1rk1/1p2ppbp/p1np1np1/2p5/2B1P3/1P3NN1/PBPP1PPP/R2Q1RK1 b - - 1 9",
    "rn3rk1/1pq2bbp/2pp1np1/3Ppp2/1pP5/PQN1P1P1/4NPBP/R1BR2K1 w - - 0 14",
    "r2q1rk1/pb3pbp/4pnp1/3p2B1/8/1BP2N2/PP3PPP/R2QR1K1 b - - 1 14",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "uci" {
        uci::run();
        return;
    }
    match args[1].as_str() {
        "perft" | "divide" => {
            let depth: u32 = args.get(2).and_then(|d| d.parse().ok()).expect("depth");
            let fen = if args.len() > 3 { args[3..].join(" ") } else { String::new() };
            let mut b = if fen.is_empty() {
                Board::startpos()
            } else {
                Board::from_fen(&fen).expect("bad fen")
            };
            if args[1] == "perft" {
                let t0 = Instant::now();
                let n = b.perft(depth);
                let dt = t0.elapsed().as_secs_f64();
                println!("{n}");
                eprintln!("# {:.3}s  {:.0} leaves/s", dt, n as f64 / dt.max(1e-9));
            } else {
                let mut total = 0u64;
                for m in b.gen_legal() {
                    let undo = b.make(m);
                    let n = b.perft(depth - 1);
                    b.unmake(m, undo);
                    println!("{} {}", m.uci(), n);
                    total += n;
                }
                println!("total {total}");
            }
        }
        "bench" => {
            let depth: i32 = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(5);
            let mut total_nodes = 0u64;
            let mut total_sing = 0u64;
            let mut total_see_skips = 0u64;
            let mut total_qchecks = 0u64;
            let t0 = Instant::now();
            for fen in BENCH_FENS {
                let mut b = Board::from_fen(fen).expect("bench fen");
                let mut s = search::Searcher::new();
                s.rep_keys = vec![b.key];
                let (mv, score, d) = s.find_best_move(&mut b, depth, None);
                total_nodes += s.nodes;
                total_sing += s.sing_count;
                total_see_skips += s.see_skips;
                total_qchecks += s.qchecks_added;
                println!(
                    "d{d} best={} score={score} nodes={}",
                    mv.map(|m| m.uci()).unwrap_or_default(),
                    s.nodes
                );
            }
            let dt = t0.elapsed().as_secs_f64();
            println!("total {total_nodes} nodes  {:.3}s  {:.0} nps", dt, total_nodes as f64 / dt.max(1e-9));
            if total_sing > 0 {
                println!("singular extensions: {total_sing}"); // probe 0060, positive control
            }
            if total_see_skips > 0 {
                println!("see skips: {total_see_skips}"); // probe 0061, positive control
            }
            if total_qchecks > 0 {
                println!("qchecks added: {total_qchecks}"); // probe 0071
            }
        }
        "seebatch" => {
            // probe 0061: oracle channel — stdin "FEN;uci" → SEE
            use std::io::BufRead;
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                let line = line.unwrap_or_default();
                let mut it = line.splitn(2, ';');
                let (fen, uci) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                if fen.is_empty() || uci.is_empty() { continue; }
                let mut b = match Board::from_fen(fen) { Ok(b) => b, Err(_) => { println!("ERR"); continue; } };
                let mv = b.gen_legal().into_iter().find(|m| m.uci() == uci);
                match mv {
                    Some(m) => println!("{}", b.see(m)),
                    None => println!("ILLEGAL"),
                }
            }
        }
        "eval" => {
            // probe 0010 invariant: direct eval parity vs sparring v3
            let fen = args[2..].join(" ");
            let b = Board::from_fen(&fen).expect("bad fen");
            println!("{}", eval::evaluate(&b));
        }
        "nnueeval" => {
            // probe 0029 parity gate: stm-cp == net_eval.py <net> eval <fen>
            let fen = args[2..].join(" ");
            let b = Board::from_fen(&fen).expect("bad fen");
            println!("{}", nnue::evaluate(&b));
        }
        "nnue2feat" => {
            // probe 0035 Stage 1: threat feature-set indices for the parity gate.
            // Argument: file with one FEN per line (the "| cp | res" tail is ignored).
            // Output per line: "<white-persp indices> ; <black-persp indices>".
            let path = args.get(2).expect("fen file");
            let text = std::fs::read_to_string(path).expect("read fen file");
            let join = |v: Vec<usize>| {
                v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
            };
            for line in text.lines() {
                let fen = line.split('|').next().unwrap_or("").trim();
                if fen.is_empty() {
                    continue;
                }
                let b = Board::from_fen(fen).expect("bad fen");
                let w = join(nnue2::feature_indices(&b, board::WHITE));
                let bl = join(nnue2::feature_indices(&b, board::BLACK));
                println!("{w} ; {bl}");
            }
        }
        "micro" => {
            // probe 0009 step 0: per-component cost on the bench suite
            let iters: u32 = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(20_000);
            let mut boards: Vec<Board> =
                BENCH_FENS.iter().map(|f| Board::from_fen(f).unwrap()).collect();
            let mut sink = 0u64;

            let t0 = Instant::now();
            for _ in 0..iters {
                for b in boards.iter_mut() {
                    sink += b.gen_legal().len() as u64;
                }
            }
            let gen_ns = t0.elapsed().as_nanos() as f64 / (iters as f64 * boards.len() as f64);

            let t0 = Instant::now();
            for _ in 0..iters {
                for b in boards.iter() {
                    sink = sink.wrapping_add(eval::evaluate(b) as u64);
                }
            }
            let eval_ns = t0.elapsed().as_nanos() as f64 / (iters as f64 * boards.len() as f64);

            // make+unmake over each position's legal moves
            let move_lists: Vec<Vec<board::Move>> =
                boards.iter_mut().map(|b| b.gen_legal()).collect();
            let n_moves: usize = move_lists.iter().map(|m| m.len()).sum();
            let t0 = Instant::now();
            for _ in 0..iters {
                for (b, ms) in boards.iter_mut().zip(&move_lists) {
                    for &m in ms {
                        let u = b.make(m);
                        b.unmake(m, u);
                        sink += 1;
                    }
                }
            }
            let mk_ns = t0.elapsed().as_nanos() as f64 / (iters as f64 * n_moves as f64);

            // arm C (0009): copy-make = clone board, make, drop copy —
            // vs make+unmake measured above
            let t0 = Instant::now();
            for _ in 0..iters {
                for (b, ms) in boards.iter_mut().zip(&move_lists) {
                    for &m in ms {
                        let mut child = b.clone();
                        child.make(m);
                        sink += child.key & 1;
                    }
                }
            }
            let cm_ns = t0.elapsed().as_nanos() as f64 / (iters as f64 * n_moves as f64);

            println!("gen_legal : {gen_ns:>8.0} ns/node");
            println!("evaluate  : {eval_ns:>8.0} ns/node");
            println!("make+unmk : {mk_ns:>8.0} ns/move (~{:.0} ns/node at ~35 moves)", mk_ns * 35.0);
            println!("copy-make : {cm_ns:>8.0} ns/move (clone+make, no unmake)");
            eprintln!("# sink {sink}");
        }
        "datagen" => {
            // probe 0026: self-play data for the M2 NNUE
            let out = args.get(2).expect("out prefix");
            let threads: u32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
            let games: u32 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(100);
            let seed: u64 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(26);
            let nodes: u64 = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(5000);
            datagen::run(out, threads, games, seed, nodes);
        }
        _ => {
            eprintln!("usage: neryba [uci] | perft|divide <depth> [fen] | bench [depth] | micro [iters] | datagen <out_prefix> [threads] [games/thread] [seed] [nodes/move]");
            std::process::exit(2);
        }
    }
}
