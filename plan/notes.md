# Micro-Jev — working notes

Newest first. Decisions, deviations from the design doc, and things to check later.

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
