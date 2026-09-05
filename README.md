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
  + SEE pruning, null-move pruning, LMR, RFP, razoring, persistent
  search state with history aging, killers/history
- Time management: non-uniform budget with soft/hard bounds
- Born: May 2026 (Python prototypes), Rust core: July 2026

## Versions

Each published snapshot carries a version (`Cargo.toml`, echoed in
`uci` → `id name`); tags `vX.Y.Z` mark them in this repository. MINOR
bumps on every strength-changing deployment, PATCH on non-playing
changes. Production weights are attached to the matching GitHub release.
Current: **0.9.0** — the buckets+i16 stack plus razoring, SEE-based
capture ordering, the flywheel v3 net and the six-flag search package.

## Building

```
cargo build --release
./target/release/neryba bench 5     # 61578 nodes for this version
```

The production NNUE weights (`src/nets/neryba0183.bin`, ~196K, trained
on the engine's own self-play data) are included — the repository builds
out of the box. net-0063 (probe 0063, production since 2026-07-17) adds
8 phase-conditioned output buckets on top of the net-2 flywheel weights
(probe 0085, also included); it beats net-2 by +10.9 Elo in the SPRT
deploy gate — a relative, internal number, not an external rating.
Version 0.6.0 adds razoring (probe 0161): +24.4 Elo [+6.95, +41.90] over
0.5.0 over 1000 fixed games at the production time control 180+2, after
a +61.1 screening at 8+0.08 — the gap between the two is the usual one,
short time controls overstate an improvement. `NERYBA_RAZOR_OFF=1`
restores the 0.5.0 search tree bit-for-bit.

Version 0.7.0 adds SEE-based capture ordering (probe 0126 for the metric,
probe 0175 for the strength): a capture with `see(m) < 0` is scored
`20_000 + mvv_lva` instead of `100_000 + mvv_lva`, so it drops below the
killers but stays above quiet moves. The share of beta-cutoffs happening
on the very first move goes from 83.5% to 91.3%, and the tree shrinks by
24% at `bench 13`. Measured at +23.0 Elo [+11.9, +34.1] over 0.6.0 across
2500 fixed games at 180+2, after a +20.8 screening at 8+0.08.
`NERYBA_SEE_ORD_OFF=1` restores the 0.6.0 search tree bit-for-bit.

Version 0.8.0 replaces the net (probe 0183). The architecture does not
change at all — still HIDDEN=128 with 8 output buckets — and neither does
the training volume, mix, node budget or filter. The only difference is
who generated the corpus: the 0.7.0 engine itself, rather than the older
stack the previous net had learned from. That single change is worth
+88.3 Elo [+76.6, +100.2] over 0.7.0 across 2500 fixed games at 180+2.
The cargo feature `nnue_0063` builds the previous net instead and
restores the 0.7.0 bench (82636 nodes).

An external gauntlet against four Stash anchors (probe 0193, 4x400 games
at 10+0.1) puts 0.8.0 at **≈2917 CCRL** [2896.9, 2938.7], measured
+54.1 above the 0.7.0 control run in the same session. Two things are
worth stating plainly. First, the internal number and the external one
are not the same quantity: +88.3 internally became +54.1 externally, a
transfer of 61%. Second, that ratio is not a constant — across three
measured points it falls as the effect grows (87% at +23 Elo, 61% at
+88, 47% at +134), so an internal gate cannot be extrapolated to an
external rating.

Version 0.9.0 turns on six search features at once (probe 0191): internal
iterative reductions (0046), singular extensions (0060), counter-moves
(0163), history pruning (0164), losing captures ordered last (0178) and a
transposition-table age guard (0111). Each of them had been measured alone
on a fast screening control and each came back inconclusive — together
they were worth +27.0 Elo by the sum of those point estimates. Measured as
a package at the production time control they are worth **+133.7 Elo**
[+122.0, +145.6] over 2500 fixed games, 4.9x the sum of the parts.

The interesting part is not the number, it is why the parts looked small.
At least two of the six cannot express themselves at a fast control at
all: the age guard governs transposition-table replacement, and at 8+0.08
the table never fills up, so replacement policy decides nothing; singular
extensions fire at depth, and there is little depth to speak of. Those
features were being judged by an instrument that physically could not
measure what they do. Externally the package is worth +75.8 on top of the
0.8.0 net, putting the engine at **≈2993 CCRL** [2971.2, 3016.5]
(probe 0193). Each flag has its own `*_OFF` ablation knob, and all six
together restore the 0.8.0 search tree bit-for-bit.

### One difference between this snapshot and production

**`NERYBA_TT_AGE_GUARD` is not part of this snapshot.** The guard refuses
a transposition-table write when the slot already holds a deeper, equally
recent entry for a different position — and that requires an age field in
the slot. Production uses an atomic, aged flat table; the table published
here is the simpler `FlatSlot { key, score, depth, flag, mv }` with no age
at all, so the feature has nothing to key off. Porting it means porting a
different table, which is a separate piece of work rather than a flag.

The difference is measured, not estimated. This snapshot reproduces
production bit-for-bit on every bench once the guard is disabled there:

```
                        bench 5   bench 8   bench 13
this snapshot            61578    320526    9234230
production, guard off    61578    320526    9234230
production, guard on     61578    320482   10633799
```

Note the shape of it: the guard costs 44 nodes at `bench 8` and 1.4M at
`bench 13`. That gap is the same phenomenon described above — whether the
table fills up — and it is why the feature looked worthless on a fast
control. The five features that are present here account for the rest of
the package.

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
