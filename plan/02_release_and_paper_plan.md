# Micro-Jev — going public and the paper

_Written 2026-10-01, right after M0. Revisit after M3._

Goal: Micro-Jev ships the way Nano-Jev did (GitHub + PyPI + Hugging Face, Apache-2.0 code)
**and** gets an arXiv paper, followed by a venue submission. This file says what goes public
when, what the paper needs, and who does what.

## 1. Where we are (2026-10-01)

- Implementation done (round 1). 54 offline tests pass.
- **M0 done on the real ModernBERT-base weights:** config matches §2; pretrained invariance
  |Δp| ≤ 1e-4 in fp32 for SDPA and eager (`tests/test_network.py`, 3/3 pass).
- **Untrained latency (RTX 3060, bf16, median of 200)** in `results/latency_untrained.json`:
  see `notes.md` (2026-10-01, M0). At k = 10, one pass is **5.7× faster than B-pair**
  (same backbone, 34 pair passes). Against Nano v1.0 (4.5× smaller) it is about even, as the
  aims predicted; at k = 20 it is slower than Nano (dense-mask cost, see notes).
- Found and fixed: row packing (several packs per 8k row) made batched inference ~3× slower,
  because SDPA with a dense mask computes the full T × T matrix. Inference now uses one pack
  per row. Training (2048-token rows) is measured in M2.
- Names are free: PyPI `micro-jev`, GitHub `shubham10divakar/micro-jev`, HF `sdmlai/micro-jev`.

## 2. Release ladder (what goes public when)

| Stage | When | What goes public | Gate |
|---|---|---|---|
| **R0 — open code** | now (your call, §6) | GitHub repo: code, design doc, plan, tests, M0 results. README says "pre-release, untrained" | Tests green in CI; no personal paths; LICENSE |
| **R1 — v0.1 + preprint** | after M3 (3 seeds) + M4 core ablations | HF `sdmlai/micro-jev` tag `v0.1` + model card; PyPI `micro-jev 0.1.0`; GitHub release; **arXiv v1** | All "Must" aims met or honestly reported; card has held-out + OOD calibration |
| **R2 — v0.2 + submission** | after M5 (phase B, maybe C) | HF `v0.2`, PyPI 0.2, arXiv v2, venue submission | Phase-B licences checked (FEVER, 2Wiki) |

Why open the code before results: it timestamps the idea, matches how Nano-Jev was done, and
costs nothing because the design is already written. The risk (someone scoops the packing
idea) is small and is covered by getting arXiv v1 out soon after M3. **If you'd rather stay
private until the preprint, skip R0 and do R0 + R1 together.** Either is fine.

### R0 checklist

- [x] `LICENSE` (Apache-2.0, same as nano-jev) and licence line in README / `pyproject.toml`
- [ ] `pyproject.toml`: name `micro-jev`, version `0.0.1.dev0`, authors, URLs, classifiers
- [x] GitHub Actions (`.github/workflows/tests.yml`): `pytest tests -q` on CPU (Linux, Python 3.12, transformers 5.17.0 pinned)
- [ ] README: status badge-free "pre-release" banner, link to Nano-Jev, citation placeholder
- [x] Grep for absolute Windows paths (none found) / `../nano_jev` assumptions; document them as optional
- [x] `CITATION.cff` (software citation; add co-authors per D3)
- [ ] Create repo `shubham10divakar/micro-jev` (public), push `main`

### R1 checklist

- [ ] Weights: `MicroJev.save` → HF repo, tag `v0.1`; `microjev.load("v0.1")` downloads them
- [ ] Model card (copy Nano's `model_cards/` layout): intended use, decisions, data, numbers
      with CIs, held-out gap, OOD calibration, latency, limitations, licences of training data
- [ ] PyPI: `python -m build` → `twine upload`; test `pip install micro-jev` in a clean venv
- [ ] Results JSON + preds committed under `results/` (small) so every table is reproducible
- [ ] arXiv: cs.CL primary, cs.IR cross-list

## 3. Licences (check before R1; not legal advice)

| Thing | Licence | Note |
|---|---|---|
| Our code | Apache-2.0 | Same as nano-jev |
| ModernBERT-base | Apache-2.0 | Fine |
| HotpotQA | CC BY-SA 4.0 | Train (phase A) |
| SQuAD 2.0 | CC BY-SA 4.0 | Train (phase A) |
| MultiNLI | mixed (OANC + CC BY-SA 3.0 parts) | Train; same as Nano v1.0 |
| MuSiQue | CC BY 4.0 | Held-out only |
| VitaminC | CC BY-SA 3.0 | Held-out only |
| 2WikiMultihopQA, FEVER | to verify | Phase B, before R2 |

Weights: Nano-Jev v1.0 shipped MIT weights trained on the same phase-A data. Use the same
licence for consistency unless you decide share-alike sources need CC BY-SA weights. Decide
once, before R1, and write the reason in the model card.

## 4. Paper plan

**Working title** (design §12): _Micro-Jev: Exact, Order-Invariant Packed Decisions for
Calibrated RAG Control in One Encoder Pass_.

**One-sentence claim:** a block-isolation mask plus position restart lets one encoder pass
answer a whole RAG decision set with outputs that are *exactly* the same as asking each
decision alone, at a fraction of the cost and with calibrated probabilities.

### 4.1 Claims → evidence → paper element

| Claim | Evidence (milestone) | Paper element |
|---|---|---|
| Exactness (G2, H1 part 1) | invariance probes, fp32 + bf16, random + pretrained + trained (M0 ✔, M3) | Table 2 |
| Speed (G1, H1 part 2) | S10 latency GPU b1 / b32 / CPU; k-curve (M0 untrained ✔, M3 trained) | Table 3 + Fig. 2 |
| Packing costs no accuracy (H1, T-3) | Micro vs B-pair, same data, 3 seeds, paired bootstrap (M3) | Table 1 |
| Context helps relevance (H2) | Micro pack vs B-k1 on MuSiQue k-sweep (M3) | Fig. 3 |
| Isolation needed for consistency, not accuracy (H3) | A1, A7 (M4) | Table 2 + Table 4 |
| Calibration in-domain and OOD (G5, T-6) | ECE / reliability before/after T, OOD-T (M3) | Fig. 4 |
| Wording robustness (H4, G7) | paraphrase test, A9 (M4) | Table 4 / 5 |
| Useful as a RAG controller | selective prediction: risk–coverage, AURC (M3) | Fig. 5 |

### 4.2 Minimum for arXiv v1 (R1)

Phase A only, 3 seeds: Tables 1–3, Figs 2–4, ablations A10, A1, A2, A3, A7.
Baselines: Nano v1.0, B-pair, B-k1. B-rerank (bge-reranker-v2-m3, relevance only) and B-llm
(Qwen3-4B) are strongly wanted (reviewers will ask), but can be partial.

### 4.3 For the venue version (R2)

Phase B (held-out gap), A4–A6, A8 (ModernBERT-large) if compute allows, A9 / paraphrase,
phase C (one model for RAG + Cyber-Jev), CPU ONNX latency.

### 4.4 Fairness rules (write them into the paper's setup section)

- Latency vs B-pair must use B-pair's **fastest** path (it has no custom mask, so it can use
  unpadded / flash kernels); today `bench_latency.py` pads it. Report both if they differ.
- Same items for every system (rebuilt Nano-format rows from the same raw pass, §5.1).
- Claim a difference only when the paired-bootstrap 95% CI excludes 0 (§6.6).
- Report what is *not* better: wall-clock vs Nano, held-out gap, OOD ECE.
- Report the row-packing slowdown as an implementation note (dense-mask SDPA cost).

### 4.5 Outline (8 pages + appendix)

1. Introduction — RAG controllers ask many small questions; per-pair encoders pay k× cost;
   naive packing breaks consistency. Contributions (design §12).
2. Related work — PCW (Ratner et al. 2023), Set-Encoder (Schlatt et al. 2025), FiD, Self-RAG /
   CRAG / Adaptive-RAG, ModernBERT, Nano-Jev, JEV-as-a-Judge. **Verify every citation.**
3. Method — packed format, mask + position restart, `<ref>` / `<opt>` readout, pair head,
   global decision tokens in local layers, the exactness argument.
4. Setup — data (phase A), baselines, metrics, hardware (one RTX 3060), seeds, stats.
5. Results — exactness, speed, accuracy, calibration, k-sweep, selective prediction.
6. Ablations.
7. Limitations — not zero-shot (Tiny-Jev), held-out gap, English only, dense-mask cost.
8. Release — code, weights, package.

Appendix: config (design App. A), templates, full per-decision metrics, reliability diagrams,
row-packing note.

### 4.6 Venue

1. **arXiv first** (R1), so the work is citable while it is under review.
2. Then pick by timing and fit; **check current deadlines, don't trust memory**:
   - IR venues (SIGIR short / resource track, ECIR, CIKM): best fit for "RAG control + reranking".
   - ACL Rolling Review (ACL / EMNLP / NAACL), or an efficient-NLP workshop for a shorter version.
   - The journal used for Nano-Jev (ICCK) if a journal suits you better.

## 5. Order of work from here

1. **M1 data** (now): build phase-A packs, held-out, OOD, k-sweep; length stats; re-score Nano.
2. **M2**: overfit 200 packs → phase-A seed 0. Measure row packing vs not at 2048 for training.
3. **M3**: full eval, trained latency + invariance, B-pair (A10), B-k1, seeds 1–2.
4. **R0** any time you say yes; **M4** core ablations; write paper sections 3–4 while M3/M4 run.
5. **R1**: weights, card, PyPI, arXiv v1.
6. M5 phase B → R2.

## 6. Decisions for you

| # | Decision | Recommendation |
|---|---|---|
| D1 | Open the code now (R0) or with the preprint? | Now |
| D2 | Weights licence | MIT, same as Nano (re-check share-alike) |
| D3 | Paper authorship / affiliation, same as Nano paper? | — |
| D4 | Target venue after arXiv | IR venue; decide after M3 numbers |
| D5 | arXiv endorsement for cs.CL | Needed only if you have no prior cs.CL submission |

## 7. What only you can do

- Create the GitHub repo and push (or tell Claude to do it with `gh`).
- `huggingface-cli login` + create `sdmlai/micro-jev`; PyPI token for `twine`.
- arXiv account / endorsement; final read of the paper and author list.
