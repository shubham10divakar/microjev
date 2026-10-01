# Micro-Jev — working notes

Newest first. Decisions, deviations from the design doc, and things to check later.

## 2026-10-02 — seed 0 (bilinear) trained and evaluated; B-pair started

**Training** (`runs/micro-a-s0b`, 1,988 steps, 47 min; log in `results/micro-a-s0b/`):

| dev NLL (uncalibrated) | init | ep 1 | ep 2 | ep 3 |
|---|---|---|---|---|
| grounded | 0.706 | 0.688 | 0.561 | 0.529 |
| sufficient | 0.693 | 0.696 | 0.467 | 0.378 |
| relevance | 1.121 | 1.065 | 0.510 | 0.465 |

Epoch 1 was spent on the plateau; it escaped early in epoch 2. So only ~2 of 3 epochs did
useful learning, on a decaying LR → undertrained.

**Evaluation** (temperature per decision fitted on in-domain calib; Nano = v1.0 re-scored on
the same rebuilt items):

| | Micro s0 rel / suff / grd | Nano v1.0 rel / suff / grd |
|---|---|---|
| test acc | 0.796 / 0.805 / 0.768 | 0.815 / 0.846 / 0.848 |
| held-out acc | 0.618 / **0.720** / 0.640 | 0.640 / 0.679 / 0.675 |
| test ECE (cal) | 0.027 / 0.041 / 0.046 | 0.020 / 0.018 / 0.037 |
| held-out ECE (cal) | **0.158 / 0.090 / 0.068** | 0.173 / 0.176 / 0.171 |

- M-3 (in-domain ≥ Nano) **not met** yet: −2 / −4 / −8 pts. M-4 (ECE ≤ 0.03) met for relevance only.
- Held-out **sufficient +4.1 pts** over Nano, and held-out calibration much better on all three
  (ECE 0.07–0.16 vs 0.17). Held-out relevance / grounded still below Nano.
- Grounded (MNLI → VitaminC) is the weakest: the isolation mask means the premise never sees
  the claim (see 2026-10-01 evening).

**B-pair started** (`runs/b-pair`, nano_jev trainer, ModernBERT-base, 3 epochs, lr 5e-5,
batch 8, max_length 512 = Nano's setting; 1024 at batch 8 or 16 ran out of memory without
gradient checkpointing). It learns immediately (loss 0.85 → 0.67 in 600 steps, no plateau):
the plateau is specific to the packed / option-marker formulation. ~5.5 h for 3 epochs.

**Next:** (1) evaluate B-pair → backbone vs packing; (2) fix the plateau so all epochs count:
two-stage (full mask first, then block) or more epochs, and the 3-way controlled comparison
on `data_dbg` (full + Nano format / full + markers / block + markers); (3) seeds 1–2.

## 2026-10-01 (night) — bilinear head; seed 0 restarted

- The pair head + linear path still sat at chance on `data_dbg2` (MNLI + Hotpot, 2 epochs:
  macro dev NLL 0.829 → 0.811).
- **Why option scoring plateaus.** With one shared scoring vector, logit(yes) − logit(no) can
  only come from the *option tokens* carrying the decision differently, which a pretrained
  encoder doesn't do. A normal classifier has a free weight vector per class instead. Escape
  from this plateau is slow and random (same config, different settings: escape after ~100
  steps or not within 2 epochs).
- **Bilinear head** (`heads.BilinearScorer`, now the default; `pair` kept as an ablation):
  `logit = w·LN(o) + (U·LN(a))·(V·LN(o))/√r`, r = 256. logit(yes) − logit(no) is then linear in
  the anchor, like a classifier, but still scores any option wording, and `<ref>` anchors keep
  relevance per chunk. MNLI debug, trainer, 2 epochs: dev 0.704 → 0.687 → **0.603** (pair: flat
  0.693); without augmentation 0.695 → 0.597 → 0.570. Still slower than plain ModernBERT
  (0.32 after 1 epoch), so watch the first epoch of the full run.
- Micro-batches are now capped at `packs_per_step` rows, so the estimated 2,430 steps match the
  real count and the LR schedule is right.
- Seed 0 restarted as `runs/micro-a-s0b` (the old folder is locked on Windows; the failed
  pair-head checkpoint is still in `runs/micro-a-s0`).

## 2026-10-01 (evening) — seed 0 did not learn; debugging the head

**Symptom.** Phase-A seed 0, epoch 1 (446 steps): dev NLL 0.692 / 0.696 / 1.059 (grd / suff /
rel) = chance. Stopped the run. The checkpoint outputs almost the same probabilities for every
pack (MNLI spread 0.036): it learned the label priors only.

**Red herring.** The logged per-decision train losses (grounded 0.15) were wrong: each name's
sum was divided by *all* micro-batches, including ones without that decision. Fixed
(`parts_n` in `trainer.py`). Also: steps/epoch was 446 not 810 because a 16k-token
micro-batch holds ~58 short packs (> `packs_per_step`); the LR schedule length is off. Open.

**Ruled out** (MNLI-only debug set, 4k train / 300 dev, 2 epochs, `data_dbg/`):
- labels after option shuffling: 0 / 200 mismatches; outputs exactly order-invariant (3e-6);
- grad checkpointing, bf16 vs fp32, lr_backbone 2e-5 / 1e-4, lr_head 2e-3, mean readout (A4):
  all stay at ln 2;
- environment: plain `ModernBertForSequenceClassification` on the same pairs learns fast
  (dev 0.32 after epoch 1 with mean pooling, 0.44 with CLS pooling);
- readout indices: anchor = `<q>`, options = `<opt>`, bookkeeping matches §3.6.

**Found** (minimal loop, same data, our model):

| mask | head | dev NLL ep 1 → ep 2 |
|---|---|---|
| full (A1) | pair | 0.685 → 0.565 |
| full (A1) | linear (A5) | 0.628 → **0.460** |
| block | pair | 0.694 → 0.693 (nothing) |
| block | pair + input LayerNorm | 0.693 → 0.693 |
| block | linear (A5) | 0.699 → **0.577** |

1. **The pair head is the blocker.** With the isolation mask it learns nothing in 2 epochs; a
   linear head on the option token learns. Two hidden dims (67, 251) of ModernBERT's last layer
   are ~50× the median magnitude and dominate `wa`/`wo`; input LayerNorm alone did not fix it.
2. **Isolation slows learning** (block 0.577 vs full 0.460 with the same linear head): state
   tokens never see the claim, so NLI happens only in the decision tokens. Expected from the
   design, and an honest paper point (H3 is about consistency, but there is a learning-speed
   cost too).
3. The last head layer is now small-random instead of zero (a zero last layer passes zero
   gradient below it), and the pair head got input LayerNorms **plus a direct linear path on
   the option token** (`z = lin(LN(o)) + MLP(...)`), so global decisions learn like the linear
   head while `<ref>` decisions keep the anchor term. Offline tests pass. **Not yet verified on
   GPU**: the mixed MNLI + Hotpot check (`data_dbg2/`) was cut short when other GPU jobs
   (not Micro-Jev) started at 15:45.

**Next (when the GPU is free):** `train.py --data data_dbg2 --epochs 3` → per-decision dev
NLL must drop clearly below chance for all three decisions (relevance via `<ref>` too). If
relevance still sits at its prior, try `ref_view=all` (A6) and a linear term on the `<ref>`
anchor. Then restart seed 0.

## 2026-10-01 — M1 data, Nano re-scored, M2 overfit, seed 0 started

- **M1 data** (`prepare_packed.py`, log in `logs/m1_prepare.log`): train 25,920 packs
  (139k labelled groups), calib 1,994, test 1,992; held-out 3,000 (MuSiQue + VitaminC);
  OOD calib 1,000 MuSiQue-train packs; k-sweep 1,520 packs (21k groups). Train lengths:
  p50 251, p95 1,147, p99 1,289, max 1,999 tokens → **0% truncated at 2048**, keep
  `max_len_train: 2048`.
- **Nano v1.0 re-scored on the rebuilt sets** (`results/baselines/`):

  | | rel | suff | grd |
  |---|---|---|---|
  | test, rebuilt | 0.815 | 0.846 | 0.848 |
  | test, v1.0 report | 0.815 | 0.845 | 0.844 |
  | held-out, rebuilt | 0.640 | 0.679 | 0.675 |
  | held-out, v1.0 report | 0.635 | 0.670 | 0.675 |

  5 of 6 within ± 0.5 pt; held-out sufficient is +0.9. Checked: same questions, labels and
  structure (3 paragraphs, 520/520), but **different distractor paragraphs** (only 7 / 1,040
  states identical; relevance 3,158 / 4,655), because Nano's builder and ours draw distractors
  through different RNG paths. 0.9 pt is within sampling noise (SE ≈ 1.4 pt at n = 1,040). All
  paper comparisons use the rebuilt items for every system, so this does not bias them. The
  paper should cite the rebuilt Nano numbers, not the v1.0 report.
- **M2 overfit (200 packs, 30 epochs, constant LR):** train loss 0.022 at step 200 (< 0.05 ✔),
  best dev NLL 0.009. Grounded and sufficient fit by epoch ~13; **relevance sat at ~0.55–0.6
  for ~15 epochs before dropping** — consistent with the "option markers start identical"
  warning. It did break through, so no A4 switch yet; watch relevance in the full run.
- **Row packing off for training too.** 40-step test on the full data: 3.4 s/step with row
  packing vs 2.3 s/step without (~1.5× faster). `configs/*.yaml` now set `row_packing: false`.
  Seed 0 at ~810 steps/epoch × 3 epochs ≈ 1.5–2 h on the 3060.

## 2026-10-01 — M0 on the real weights (GPU)

- `MICROJEV_NETWORK_TESTS=1 pytest tests/test_network.py`: **3/3 pass.** Config matches §2
  (22 layers, 768, 12 heads, 8192 positions, global every 3rd layer, window 128, vocab 50368;
  markers fit in the spare vocab rows). Pretrained invariance probes ≤ 1e-4 fp32, SDPA and eager.
- Untrained latency, RTX 3060, bf16, S10-style packs (150-token chunks), median of 200 after 20
  warm-ups (`results/latency_untrained.json`):

  | k | Micro 1 pass (ms) | Micro batch-32 (ms/pack) | B-pair (ms, pairs) | Nano v1.0 (ms) | B-pair / Micro |
  |---|---|---|---|---|---|
  | 1 | 22.2 | 5.0 | 44.3 (7) | 40.3 | 2.0× |
  | 2 | 22.1 | 8.4 | 66.0 (10) | 41.7 | 3.0× |
  | 5 | 27.9 | 19.6 | 141.0 (19) | 44.2 | 5.1× |
  | 10 | 52.1 | 43.1 | 295.9 (34) | 58.2 | **5.7×** |
  | 20 | 114.5 | 112.6 | 675.6 (64) | 85.3 | 5.9× |

  M-2 (≥ 5× vs B-pair at k = 10) met on latency. Against Nano: faster up to k = 10, **slower at
  k = 20** (114 vs 85 ms). Cause: dense-mask SDPA computes the full T × T matrix (k = 20 is
  ~3.4k tokens), most of which the block mask throws away. Block-sparse attention
  (FlexAttention, WSL2/Linux) is the fix to try; it is also a paper limitation to state.
- **Row packing slowed inference ~3×.** The first run put 32 packs into shared 8k rows:
  130.7 ms/pack at k = 10 vs 43.6 with one pack per row (same reason: dense T × T). Inference
  (`score_packs`, `evaluate.py`, `bench_latency.py`) now defaults to one pack per row;
  `evaluate.py --row-packing` turns it back on. Training keeps row packing at 2048 tokens for
  now; measure both ways in M2 before the long run.
- Fairness (paper): B-pair here runs padded batches through HF's masked path. Before the paper,
  time B-pair on its fastest path too (no custom mask → unpadded / flash) and report both.
- HF Hub warns about symlinks on Windows (cache works, uses more disk). Set
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1` or enable Developer Mode.

## 2026-10-01 — scripts, trainer, end-to-end smoke (steps 10–11)

- **Marker learning rate (§4.5):** one embedding tensor can't sit in two AdamW groups, and
  scaling gradients does nothing under Adam. So the markers get a separate zero-initialised
  `marker_delta` parameter (lr_head, no decay) added through a forward hook on the token
  embedding. `MicroJev.save` folds it into the embedding rows, so checkpoints are plain
  ModernBERT + `head.pt`. Tested: save → load reproduces logits exactly.
- **Free-text evidence goes in a segment, not the header** (deviation from §3.2 / §5.1
  "header = premise"). The header is capped at 256 tokens (§3.7), which would have cut MNLI /
  VitaminC evidence, `grounded(claim, ctx)`, `decide(..., state)` and Cyber-Jev HTTP requests.
  `text_state()` = empty header + one untitled segment. Side benefit: grounded training now
  looks like grounded-over-passages at inference (the off-distribution concern below is
  smaller, though MNLI premises are still single short segments).
- Trainer: micro-batches by token budget; gradients accumulated until ≥ `packs_per_step`
  packs, each micro-batch's loss weighted by its pack count. Step count for the LR schedule
  is estimated as ceil(n_train / packs_per_step) × epochs.
- Overfit check on the tiny CPU model: dev NLL starts at exactly log K (zero-init head) and
  drops below 0.1 on 8 packs. The real M2 check (200 packs, loss < 0.05) needs the GPU.
- B-pair (A10) reuses nano_jev's own trainer on the Nano-format export (`data/nano/`), so the
  control really is "Nano architecture, same items". The export is deduplicated. Note B-pair
  sees all ~8 distractors per question every epoch; Micro samples k ~ U{0..8} (mean 4).
- `custom` temperature: the mean of the builtin temperatures until a paraphrase-calib split
  exists (§8 wants it fitted on the paraphrase-test calib half; not built yet).
- Windows console (cp1252) can't print "→" / "Δ": every script reconfigures stdout to UTF-8.
  The smoke test runs the scripts without PYTHONIOENCODING, so it would catch this again.
- Smoke-tested `baselines.py nano` on CPU against the local `nano_jev/runs/nano-jev-v1.0`
  weights with synthetic rows (plumbing only; numbers meaningless).
- `tests/test_network.py` (opt-in) checks the §2 config table and runs the invariance probes
  on real ModernBERT-base weights. This is M0 proper; not run yet (no downloads this round).

Open items (not blocking training):
- Phase B builders (2Wiki, FEVER): HF ids / licences unverified.
- Paraphrase-test calib half for `T_custom`.
- ONNX / CPU int8 export (latency stretch goal).

## 2026-10-01 — core implemented (steps 1–9), CPU tests only

Verified against installed `transformers 5.17.0` source (`modeling_modernbert.py`):
- `ModernBertModel.forward` uses a dict `attention_mask` as-is, keyed by layer type. ✔ (§2)
- SDPA takes our bool masks (True = attend). The `sliding_window` kwarg is ignored by SDPA, so
  our `sliding_attention` mask alone sets the window. Eager *adds* the mask, so
  `MicroJev._mask` converts bool → additive float for `attn="eager"`.
- Embeddings have no absolute positions; RoPE uses our `position_ids` (separate θ per layer type).

M0 on random weights (tiny 3-layer config, CPU, fp32): **all §6.4 perturbations within 1e-4**
for both SDPA and eager — alone, reversed order, options permuted, extra decision, row-packed
after another example, batched next to a longer padded row. The no-mask ablation (A1) does
change outputs (> 1e-3), so the test can detect leaks. Still to do on the real
ModernBERT-base weights (M0 proper).

Design deviations / decisions:
- `render()` renders one example with positions from 0; `pack_row()` concatenates examples into
  a row (offsets `blk` and indices). Multi-example rows (§5.5) come from this for free.
- **A2 (siblings) mask fix:** the §3.8 sketch (`k_opt = k_is_opt & ~q_ref`) lets *question-span*
  tokens see options in siblings mode, contrary to the §3.6 table. Implemented per the table:
  only option tokens see sibling options.
- Ablation switches live in `PackConfig` (`mask`, `position_restart`, `isolated`, `ref_view`,
  `decision_global`) and the model cfg (`readout`, `head`). A4 mean readout uses per-option
  spans from collate.
- Truncation works on token ids (no re-tokenising), header head+tail 128+128, segment cap by
  binary search. If untargeted segments are dropped, the kept ones are renumbered "[1]..[n]".
- `Group` moved from schema to packing (it holds token indices).
- Eval packs (phase A): `n_distract=1` → full pack = 2 gold + 1 distractor (sufficient yes),
  minus pack = 1 gold + 2 distractors (no). Relevance: all of the full pack + only the *new*
  distractor of the minus pack = Nano's 2 gold + 2 distractors per question, nothing counted
  twice. `to_nano_rows()` exports the identical items for re-scoring Nano v1.0.
- Paraphrase templates: removed a negated held-out question idea (would need a label flip).
- Phase B builders (2Wiki, FEVER) raise `NotImplementedError` until their HF ids and licences
  are checked (M5). Phase C uses `cyber_packs()` on Cyber-Jev JSONL.
- Decider splitting: halves the passage list recursively until each part fits. Segment-scope
  results are mapped back by offset; global decisions are averaged over parts with a warning.

Things to watch when training starts:
- **Option markers start nearly identical.** Every `<opt>` is the same token at the same
  (restarted) position; at ModernBERT's 0.02 init in the tiny test model, all options got
  ~identical hidden states. The pretrained backbone attends far more sharply, but if early
  training is slow to move off log K, try A4 (mean readout over the option span) first.
- **`grounded` inside a RAG pack is off-distribution in phase A**: it is trained with
  header = premise and no segments (MNLI), but the S10 scenario asks it over query + passages.
  Consider grounded packs with segments in phase B (e.g. FEVER evidence as segments).
- Invariance only holds when no truncation is triggered: adding decisions shrinks the state
  budget. Probe packs must fit without truncation.

## 2026-10-01 — start

- Repo: `code_repo/microjev/` with its own `git init`; added `microjev/` to
  `code_repo/.git/info/exclude` so the outer nano-jev repo ignores it.
- **Deviation:** design §9 names the folder `code_repo/jev_core/`. Code lives in `microjev/`
  instead (where the design doc is). The shared package is still called `jevcore`, so Tiny-Jev
  can import it later or the package can be moved without renames.
- GPU busy: implement only. Tests use a tiny random ModernBERT config on CPU.
