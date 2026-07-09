<div align="center">
  <h1>Neryba</h1>
  <p><em>Нериба — "not-a-fish"</em></p>
</div>

Neryba is an original chess engine written in Rust, grown exclusively
through pre-registered experiments: no line of code enters the engine
without a measured, committed-in-advance verdict behind it. Zero
borrowed code, zero dependencies — the *not-a-clone* principle is baked
into the name.

Every feature you see here — NNUE evaluation (trained on the engine's
own self-play data), the search stack, time management — earned its
place through an SPRT gate or was honestly buried trying. The graveyard
of killed ideas is as much a product as the engine itself.

## Play it

Neryba plays live on Lichess: **[BOT neryba](https://lichess.org/@/neryba)** —
challenges welcome (blitz, 3+2 to 5+3).

The Rust engine entered the bot pool at around the 2000 mark (July 2026);
as of **July 9, 2026** its Lichess blitz rating is **2370** — every point
of that climb is an SPRT-gated change, deployed one verdict at a time.

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
