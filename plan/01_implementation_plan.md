# Micro-Jev — implementation plan

Spec: [`../14_micro_jev_design.md`](../14_micro_jev_design.md) (§ numbers below refer to it).
Aims: [`00_aims.md`](00_aims.md). Running log: [`notes.md`](notes.md).

**Scope of this round (2026-10-01): implement only.** GPU is busy: no training, no evaluation runs,
no dataset downloads. Everything is tested on CPU with a tiny randomly initialised ModernBERT
config, so the tests need no network.

## Layout (in `code_repo/microjev/`, own git repo)

```
microjev/
  14_micro_jev_design.md
  plan/                    00_aims.md  01_implementation_plan.md  notes.md
  jevcore/                 shared code (Micro-Jev now, Tiny-Jev later)
    schema.py templates.yaml packing.py collate.py heads.py
    backbones/modernbert.py
    data/builders.py data/augment.py
    loss.py calibration.py report.py registry.py scoring.py decider.py
  microjev/__init__.py     thin public API: microjev.load(), microjev.Q
    loss.py trainer.py invariance.py
  scripts/                 prepare_packed.py train.py evaluate.py bench_latency.py invariance.py baselines.py compare.py
  tests/                   schema packing collate mask_invariance data decider trainer scripts (+ network, opt-in)
  configs/                 micro_base.yaml micro_large.yaml
```

## Steps (commit after each)

| # | Step | Design § | Status |
|---|---|---|---|
| 0 | git init, exclude from outer repo, aims + plan + notes | §9 | done |
| 1 | Scaffold: pyproject (pin transformers 5.17.0), .gitignore, README stub, configs | §2, App. A | done |
| 2 | `schema.py`: dataclasses, builtin decisions, `validate()`, Nano-row adapter; `templates.yaml` | §3.1–3.2, §5.2 | done |
| 3 | `packing.py`: markers, `Row`, `Group`, `truncate_state`, `render`, `allowed`, `local_from_global`, multi-example rows | §3.3–3.8, §5.5 | done |
| 4 | `heads.py` (PairScorer + linear ablation) and `backbones/modernbert.py` (MicroJev, marker init) | §3.3, §4 | done |
| 5 | `collate.py`: bucketing, row packing, group flattening, batch masks | §5.5 | done |
| 6 | Tests: schema, packing, collate, **mask invariance on random weights** (CPU, tiny config) | §6.4, M0 | done |
| 7 | `data/augment.py` + `data/builders.py` (hotpot / squad2 / mnli / musique / vitaminc → packs; Nano-row export for re-scoring) | §5.1–5.2 | done |
| 8 | Loss + `calibration.py` / `report.py` (copied from nano, + AUROC, QWK, AURC, risk–coverage) | §5.3, §5.6, §6.2 | done |
| 9 | `decider.py` + `microjev` API (run, Nano-compatible wrappers, split passes, temperatures, OOD flag) | §8 | done |
| 10 | Scripts: prepare_packed, train (incl. `--overfit N`), evaluate, invariance, bench_latency, baselines (B-pair, B-k1) | §5, §6 | done |
| 11 | README with how-to-run for M0–M3; update notes | — | done |

**Round 1 result (2026-10-01):** steps 0–11 done; 54 offline tests pass on CPU, 3 opt-in network tests (M0 on real weights) skipped.
No training, no downloads, GPU untouched.

## Next (needs GPU / network; not this round), in order

1. M0 proper: `MICROJEV_NETWORK_TESTS=1 pytest tests/test_network.py` + untrained latency bench
   (gives the G1/H1 latency numbers before any training).
2. M1: prepare data, length stats (decide 2048 vs 4096), re-score Nano v1.0 on the rebuilt sets.
3. M2: overfit 200 packs, then phase-A seed 0. Watch the first ~200 steps for the "options start
   identical" risk (notes.md).
4. M3: full eval + invariance on trained weights + B-pair + B-k1; then seeds 1–2.
5. Check results against `00_aims.md` (Must / Target) and the kill criteria.

## Original "later" list

- M0 on the real `answerdotai/ModernBERT-base` weights: confirm config values (§2), dict-mask
  path, fp32 invariance ≤ 1e-4.
- M1: `prepare_packed.py` (downloads), length percentiles, Nano re-scored on rebuilt held-out.
- M2: overfit 200 packs, phase-A seed 0. M3–M6 as in §10.
