# Micro-Jev — working notes

Newest first. Decisions, deviations from the design doc, and things to check later.

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
