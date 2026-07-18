<div align="center">
  <h1>Neryba</h1>
  <p><em>Нериба — "not-a-fish"</em></p>

<br>

[![lichess-blitz](https://lichess-shield.vercel.app/api?username=neryba&format=blitz)](https://lichess.org/@/neryba/perf/blitz)
[![lichess-rapid](https://lichess-shield.vercel.app/api?username=neryba&format=rapid)](https://lichess.org/@/neryba/perf/rapid)
[![lichess-bullet](https://lichess-shield.vercel.app/api?username=neryba&format=bullet)](https://lichess.org/@/neryba/perf/bullet)

<br>

</div>

Neryba is an original chess engine written in Rust, grown exclusively
through pre-registered experiments: no line of code enters the engine
without a measured, committed-in-advance verdict behind it. Zero
borrowed code: no line of source from another engine has entered this
repository. The Rust engine has no external crates. The offline
kitchen — data generation, Syzygy label filtering, NNUE training — uses
python-chess and PyTorch openly. The *not-a-clone* principle is baked
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
+75 (probe 0065). A dedicated decomposition run (probe 0069) then split
the 43% shrinkage: time control contributes ~0 (the package holds
+101 at 10+0.1 vs +100 at 8+0.08), ~31 Elo is chain non-additivity —
summing sequential SPRT verdicts overstates the package they build,
through patch interactions, early-stop bias, and the false-green rate
of α=0.05 — and ~25 Elo is the self-play pool versus external anchors.
The lesson is now house policy: packages are measured A/B as a whole;
sums of individual verdicts are never quoted as strength claims.

## Facts

- Language: Rust, single thread (for now), zero external crates
- Evaluation: own NNUE `(768→128)x2→1` with 8 phase-conditioned output
  buckets (probe 0063) and an i16 SIMD-dense layout (probe 0093,
  +12% NPS bit-exact), trained from scratch on self-play data with
  Syzygy-filtered labels (filtered offline via python-chess; the engine
  does no tablebase probing at runtime)
- Search: iterative deepening alpha-beta, flat TT, quiescence + QTT
  + SEE pruning, null-move pruning, LMR, RFP, persistent search state
  with history aging, killers/history
- Time management: non-uniform budget with soft/hard bounds
- Born: May 2026 (Python prototypes), Rust core: July 2026

## Building

```
cargo build --release
./target/release/neryba bench 5
```

The production NNUE weights (`src/nets/neryba0063.bin`, ~196K, trained
on the engine's own self-play data) are included — the repository builds
out of the box. net-0063 (probe 0063, production since 2026-07-17) adds
8 phase-conditioned output buckets on top of the net-2 flywheel weights
(probe 0085, also included); it beats net-2 by +10.9 Elo in the SPRT
deploy gate — a relative, internal number, not an external rating. An
external gauntlet of the *previous* stack (net-2, probe 0089) measured
≈2761 CCRL-anchored; the current stack has not been externally measured
yet.

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
