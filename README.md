<div align="center">
  <h1>Neryba</h1>
  <p><em>Нериба — "not-a-fish"</em></p>

<br>

[![lichess-blitz](https://lichess-shield.vercel.app/api?username=neryba&format=blitz)](https://lichess.org/@/neryba/perf/blitz)
[![lichess-rapid](https://lichess-shield.vercel.app/api?username=neryba&format=rapid)](https://lichess.org/@/neryba/perf/rapid)

<br>

</div>

Neryba is an original chess engine written in Rust, grown exclusively
through pre-registered experiments: no line of code enters the engine
without a measured, committed-in-advance verdict behind it. Zero
borrowed code, zero dependencies — the *not-a-clone* principle is baked
into the name.

Every change is gated by a pre-registered SPRT; the experiment log —
preregistrations, verdicts, the harness, training pipelines — lives in
a private research repository and will be published in full. The
`// probe NNNN` comments in the source are keys into that record; this
repository is the engine as it plays. Preregistration commits are
notarized against an off-machine mirror since July 9, 2026; records
before that date rest on the author's word alone.

## Play it

Neryba plays live on Lichess: **[BOT neryba](https://lichess.org/@/neryba)** —
challenges welcome (blitz, 3+2 to 5+3).

Baseline at the time of this commit: **2377 Lichess blitz (July 9,
2026)**. The live badges above track it from here. The pool rating is
an invitation to play, not a measurement — it moves with provisional
convergence, pool composition, and infrastructure noise.

The measurement is this. The first SPRT-gated package (RFP + persistent
search state + history aging) was claimed at +131 Elo by the internal
self-play probes that gated it. An external gauntlet at 10+0.1 measured
+75 (probe 0065). The 43% shrinkage is documented rather than explained
away — and it is not yet resolved whether it comes from the self-play
pool, from the time control, or from the false-green rate implied by
running SPRTs at α=0.05.

## Facts

- Language: Rust, single thread (for now), zero external crates
- Evaluation: own NNUE `(768→128)x2→1`, trained from scratch on
  self-play data with Syzygy-filtered labels
- Search: iterative deepening alpha-beta, TT, quiescence + QTT,
  null-move pruning, LMR, RFP, persistent search state, killers/history
- Time management: non-uniform budget with soft/hard bounds
- Born: May 2026 (Python prototypes), Rust core: July 2026

## Building

```
cargo build --release
./target/release/neryba bench 5
```

The production NNUE weights (`src/nets/neryba1.bin`, ~190K, trained on
the engine's own self-play data) are included — the repository builds
out of the box.

## License

Neryba is free software, licensed under the
[GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0) — derivatives
must stay open, including network use. Ideas from other engines are
welcome with citation; lines of code from other engines never enter
this repository.

## Philosophy

Ukrainian has the idiom «ні риба ні м'ясо» — "neither fish nor meat."
Stockfish is the reigning optimum of computer chess; you don't beat the
fish by becoming a better fish in its ocean. Neryba is an attempt to be
a different animal in the same water — and to document, with numbers,
every place where a shortcut turned out not to exist.
