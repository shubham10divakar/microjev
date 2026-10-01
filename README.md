# Micro-Jev

One-pass packed decision encoder for RAG control. A whole decision set (every chunk's
relevance, sufficiency, groundedness, custom questions) is read from **one** ModernBERT-base
forward pass. A block-isolation mask plus position restart make each decision's output
identical to asking it alone.

- Design: [`14_micro_jev_design.md`](14_micro_jev_design.md)
- Aims and expectations: [`plan/00_aims.md`](plan/00_aims.md)
- Plan / status: [`plan/01_implementation_plan.md`](plan/01_implementation_plan.md)
- Working notes and deviations from the design: [`plan/notes.md`](plan/notes.md)
- Release and paper plan: [`plan/02_release_and_paper_plan.md`](plan/02_release_and_paper_plan.md)

**Status (2026-10-01): pre-release, not trained yet.** Offline tests run on CPU with a tiny
random ModernBERT. On the real ModernBERT-base weights, packed decisions match separate ones to
≤ 1e-4 (fp32), and one pass at k = 10 chunks is 5.7× faster than per-pair scoring with the same
backbone (RTX 3060, untrained; `results/latency_untrained.json`).

## Layout

```
jevcore/              shared package (Micro-Jev now, Tiny-Jev later)
  schema.py           packed-example schema, builtin decisions, validate()
  templates.yaml      question / option wordings (train vs held-out for the paraphrase test)
  packing.py          render(), truncation, block masks, local window, row packing
  collate.py          batching: row packing, token budgets, group flattening
  heads.py            PairScorer (+ linear ablation)
  backbones/modernbert.py   MicroJev: encoder + head, marker tokens, save / load
  loss.py trainer.py  per-decision CE; training loop
  calibration.py report.py scoring.py invariance.py decider.py registry.py
  data/builders.py data/augment.py
microjev/             public API: microjev.load(), microjev.Q
scripts/              prepare_packed, train, evaluate, invariance, bench_latency, baselines, compare
configs/              micro_base.yaml (design App. A), micro_large.yaml (A8)
tests/                offline tests (+ test_network.py for M0 on the real weights)
```

## Setup

Uses the shared `code_repo/.venv` (Python 3.12, torch 2.11 cu128, **transformers 5.17.0 pinned**).

```bash
python -m pytest tests -q                                   # offline, CPU, ~45 s
MICROJEV_NETWORK_TESTS=1 python -m pytest tests/test_network.py -v   # M0 on ModernBERT-base
```

## Runbook (when the GPU is free)

**M0 — real backbone (1 evening)**
```bash
MICROJEV_NETWORK_TESTS=1 python -m pytest tests/test_network.py -v
python scripts/bench_latency.py --base answerdotai/ModernBERT-base --device cuda --k 1 2 5 10 20 \
    --nano ../nano_jev/runs/nano-jev-v1.0 --out results/latency_untrained.json
```
Exit: fp32 |Δp| ≤ 1e-4 on pretrained weights, no NaN; latency table (weights don't affect it).

**M1 — data**
```bash
python scripts/prepare_packed.py --preset default --out data
python scripts/prepare_packed.py --heldout --out data_heldout
python scripts/prepare_packed.py --ood --out data_ood
python scripts/prepare_packed.py --ksweep --out data_ksweep
python scripts/prepare_packed.py --length-stats data/train.jsonl     # raise max_len_train if > 1% truncated at 2048
python scripts/baselines.py nano --data data --nano ../nano_jev/runs/nano-jev-v1.0 --name nano-test
python scripts/baselines.py nano --data data_heldout --calib-data data --nano ../nano_jev/runs/nano-jev-v1.0 --name nano-heldout
```
Exit: Nano v1.0 numbers on the rebuilt sets within ± 0.5 pt of `nano_jev/results/v1.0_comparison.md`.

**M2 — training**
```bash
python scripts/train.py --data data --out runs/overfit --overfit 200          # loss < 0.05
python scripts/train.py --data data --out runs/micro-a-s0 --seed 0
```

**M3 — evaluation**
```bash
python scripts/evaluate.py --model runs/micro-a-s0 --save-preds
python scripts/evaluate.py --model runs/micro-a-s0 --test-data data_heldout --name heldout --save-preds
python scripts/evaluate.py --model runs/micro-a-s0 --ood-calib data_ood --test-data data_heldout --name heldout_oodT
for k in 1 2 5 10 20; do python scripts/evaluate.py --model runs/micro-a-s0 --test-data data_ksweep --k $k --name k$k --save-preds; done
python scripts/evaluate.py --model runs/micro-a-s0 --paraphrase 0 0 --name para_q0_o0
python scripts/invariance.py --model runs/micro-a-s0 --n 500 && python scripts/invariance.py --model runs/micro-a-s0 --n 500 --bf16
python scripts/compare.py runs/micro-a-s0/eval/results.preds.jsonl results/baselines/nano-test.preds.jsonl
```
B-pair (A10): train the Nano architecture on the same items with nano_jev's trainer, see
`scripts/baselines.py` docstring.

**Ablations** use `--set`, e.g. `--set model.mask=full model.position_restart=false` (A1),
`model.option_mode=siblings` (A2), `model.decision_global=false` (A3), `model.readout=mean` (A4),
`model.head=linear` (A5), `model.ref_view=all` (A6), `model.position_restart=false` (A7),
`--config configs/micro_large.yaml` (A8), `data.aug.verb_p=0 data.aug.qpara_p=0` (A9).

## API

```python
import microjev
d = microjev.load("runs/micro-a-s0")
res = d.run(query=q, passages=chunks,
            decisions=["relevance", "sufficient", microjev.Q("Is the query time-sensitive?", ["yes", "no"])])
res["relevance"]    # one {option: p} per chunk
res["sufficient"]   # {option: p}
res["custom_0"]     # {option: p}
d.relevance(q, chunks); d.sufficient(q, chunks); d.grounded(claim, ctx); d.decide(question, options, state)
```

## Licence

Code: Apache-2.0 (`LICENSE`). Weights: licence set at the first release (see the release plan).
