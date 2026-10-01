# 14 — Micro-Jev: one-pass packed decision encoder (design)

_Status: design, 2026-10-01. Predecessor: Nano-Jev v1.0 (baseline and code donor).
Successor: [15_tiny_jev_design.md](15_tiny_jev_design.md), which reuses §3 (format), §4.3–4.4
(mask, head), §5.3 (loss) and §6 (evaluation) of this document._

---

## 0. Summary

Nano-Jev scores one (question, option, state) triple per forward pass. A RAG step with
10 chunks needs 10×3 relevance passes + 2 sufficient + 2 grounded = **34 passes**, each one
re-encoding the state.

Micro-Jev reads **a whole set of decisions from one forward pass** of ModernBERT-base:

```
[CLS] <state> query … <seg> [1] chunk … <seg> [2] chunk … [SEP]
      <dec> <q> question A <opt> yes <opt> no
      <dec> <q> question B <opt> irrelevant <opt> partially relevant <opt> directly answers <ref> <ref> … <ref>
                                                                              (one <ref> per chunk)
```

Three design choices do most of the work:

1. **Block attention mask + position restart.** The state never attends to decisions.
   A decision never attends to another decision. Each decision block's positions restart
   right after the state. Result: **a decision's output does not change with which other
   decisions are packed next to it, or in what order.** This is exact (up to floating-point
   error), not approximate.
2. **Readout from marker tokens with a pair-scoring head.** One logit per option is computed
   from (anchor hidden state, option hidden state). Options are still free text, so any label
   set works without retraining, as in Nano-Jev.
3. **Per-segment `<ref>` tokens.** One `<ref>` per chunk reads "this chunk, under this
   question". So 10 chunk relevances cost 10 extra tokens, not 10 passes.

## 1. Goals, non-goals, hypotheses

### Goals

| # | Goal | Measured by |
|---|---|---|
| G1 | One forward pass per (state, decision set) | Latency for the full set at k = 10 chunks (§6.5) |
| G2 | Invariance to which decisions are packed together, their order, and option order | max \|Δp\| ≤ 1e-3 in fp32 (§6.4) |
| G3 | In-domain accuracy ≥ Nano v1.0 (0.815 / 0.845 / 0.844 for rel / suff / grounded) | Test split, 3 seeds |
| G4 | Better held-out accuracy than Nano v1.0 (0.635 / 0.670 / 0.675) | MuSiQue, VitaminC |
| G5 | Calibrated: in-domain ECE ≤ 0.03 after temperature; report OOD ECE honestly | §6.2 |
| G6 | 8k context: up to 20 chunks in one pass | Accuracy-vs-k curve on MuSiQue (20 paragraphs) |
| G7 | Custom option sets better than Nano v1.0 (its known weak spot) | Paraphrase test (§6.3) |

### Non-goals

- General zero-shot decisions on new kinds of questions. That's Tiny-Jev's job (doc 15).
- `next_action`. The schema supports it, but labels need outcome rollouts (plan 03), so it's added later.
- Text generation or explanations.

### Hypotheses (the paper's claims, each with a falsifying test)

- **H1. Packing is free.** With the isolation mask, packed accuracy equals single-decision
  accuracy for the same weights (|Δacc| < 0.5 pt), while latency for the set drops ≥ 5× at
  k = 10. _Test:_ §6.4 + §6.5.
- **H2. Context helps relevance.** A chunk's relevance read inside the full pack beats the same
  chunk read alone (k = 1), especially on multi-hop held-out data (MuSiQue), because chunk tokens
  attend to the other chunks. _Test:_ Micro(pack) vs Micro(k=1) vs Nano, paired bootstrap.
- **H3. Isolation is needed for consistency, not for accuracy.** The no-mask ablation reaches
  similar accuracy but violates G2 (|Δp| ≫ 1e-3). _Test:_ ablation A1.
- **H4. Varying the option wording in training fixes custom option sets.** _Test:_ ablation A9
  on the paraphrase test.

## 2. Backbone

`answerdotai/ModernBERT-base`. Check every value against the downloaded `config.json` in M0.

| Property | Value | Consequence |
|---|---|---|
| Parameters | ~149M | ~4.5× Nano v1.0; still fast on GPU, feasible on CPU |
| Layers / hidden / heads | 22 / 768 / 12 | Head input dim 768 |
| Context | 8192 tokens | Query + 20 chunks of ~300 tokens fit |
| Attention | Global every 3rd layer; the other layers use a **local window of 128** (±64 tokens) | Decision tokens sit far from most of the state, so they need global attention in every layer (§4.3) |
| Positions | RoPE (`position_ids` accepted) | Position restart is possible (§3.6) |
| Tokenizer | BPE (OLMo-derived), cased, code-aware; vocab 50,368 (embedding padded) | Good for cyber payloads later; spare embedding rows may hold the new markers |
| Licence | Apache-2.0 | Clean for release |
| HF API (installed transformers 5.17.0) | `ModernBertModel.forward(input_ids, attention_mask, position_ids, inputs_embeds)`; **`attention_mask` may be a dict `{"full_attention": 4D, "sliding_attention": 4D}`, used as-is** | Custom masks need no monkey-patching |

Ablation backbone: `answerdotai/ModernBERT-large` (~395M, 28 layers, hidden 1024).

**Pin `transformers==5.17.0`.** The mask API changed between 4.x and 5.x. The invariance test (§6.4)
runs in CI and catches the next change.

**Attention implementation:** `sdpa`. Flash-attention's unpadded path ignores custom masks and is
hard to install on Windows. FlexAttention needs Triton, which isn't available on native Windows
(use WSL2 if speed becomes the bottleneck).

## 3. Packed decision format (shared with Tiny-Jev)

### 3.1 Concepts

- **State:** a `header` (free text, e.g. `query: …`) plus 0..K `segments` (chunks, each with
  title and text).
- **Decision:** `name`, `kind` (`choice` | `noul` | `score`), `scope` (`global` | `segment`),
  `question`, `options`.
- **Group:** the unit of softmax.
  - A global decision has 1 group.
  - A segment-scope decision has one group per target segment.
- **Label:** an index into `options`, or `-100` when unlabeled.

### 3.2 On-disk schema (one JSON object per line)

```json
{"id": "hotpot-5a8b57f2-full", "source": "hotpot", "split": "train",
 "state": {"header": "query: Which magazine was started first, Arthur's Magazine or First for Women?",
           "segments": [{"title": "Arthur's Magazine", "text": "Arthur's Magazine (1844–1846) was …"},
                        {"title": "First for Women",   "text": "First for Women is a woman's magazine …"},
                        {"title": "Radio City",        "text": "…"}]},
 "decisions": [
   {"name": "relevance", "kind": "score", "scope": "segment",
    "question": "How relevant is this passage to the query?",
    "options": ["irrelevant", "partially relevant", "directly answers"],
    "targets": [0, 1, 2], "labels": [2, 1, 0]},
   {"name": "sufficient", "kind": "noul", "scope": "global",
    "question": "Do the passages contain enough information to answer the query?",
    "options": ["yes", "no"], "label": 0}]}
```

Rules:

- `targets` lists segment indices; `labels` is aligned with `targets`.
- A plain-string state (`decide(question, options, state)`) becomes `header = state`, `segments = []`.
- **Adapter from Nano rows:** `{decision, question, options, state, label}` → one global decision,
  `header = state`. This gives the "unpacked" control on identical data.
- Validate with `schema.validate(example)`: labels in range, targets exist, at least one decision.

### 3.3 Special tokens

| Token | Role | Init embedding (mean of these words' embeddings) |
|---|---|---|
| `<state>` | Start of state | "context" |
| `<seg>` | Start of a segment | "passage" |
| `<dec>` | Start of a decision block | "decide" |
| `<q>` | Question marker; **anchor for global decisions** (Micro) | "question" |
| `<opt>` | Option marker; **option readout** (Micro) | "option" |
| `<ref>` | Per-segment anchor for segment-scope decisions | "passage", "relevant" |
| `<qe>`, `<oe>` | End of question / end of option (Tiny-Jev only; causal readout) | "answer" / "option" |

Add the tokens with `tok.add_special_tokens({"additional_special_tokens": [...]})`. If
`len(tok) > embedding rows`, resize with `pad_to_multiple_of=64`. Initialize the new rows as
in the table above, plus N(0, 0.02·std) noise.

### 3.4 Rendered sequence (Micro-Jev, bidirectional)

```
[CLS] <state> {header} <seg> [1] {title}: {text} <seg> [2] {title}: {text} … [SEP]
<dec> <q> {question} <opt> {option 1} <opt> {option 2} … [<ref> × |targets| if scope=segment]
<dec> <q> {question} <opt> … 
[PAD] …
```

Put the state first and the decisions after. Reasons:

- The state's positions (0..S−1) never depend on the decisions.
- Truncation only ever touches the state.

### 3.5 Per-token bookkeeping arrays (the heart of the packer)

| Array | State tokens | Decision-block tokens |
|---|---|---|
| `ex` | Index of the example inside the row (several short examples can share a row, §5.5) | same |
| `blk` | `0` | `d` = unique decision-block id ≥ 1 within the row |
| `prt` | `0` = header/CLS/SEP/`<state>`; `i+1` = tokens of segment i | `0` = question span (`<dec> <q> …`); `j+1` = option j span; `REF_BASE+i` = the `<ref>` for segment i |
| `pos` | 0..S−1 | Restarts per block (§3.6) |
| `valid` | 1 | 1 (0 for padding) |

`REF_BASE = 1_000_000`.

### 3.6 Attention rules and position restart

The query token on the left may attend to the key tokens on the right. Everything else is masked.

| Query token | May attend to |
|---|---|
| State token | State tokens of the same example |
| Question-span token of decision d | Whole state + question span of d |
| Option j of decision d | Whole state + question span of d + option j (**isolated**, default)<br>or + all options of d (**siblings**, ablation A2) |
| `<ref>` for segment i of decision d | State header (`prt = 0`) + segment i tokens + question span of d + itself |
| Padding | Itself only (prevents all-masked rows, which give NaN in SDPA) |

Further constraints:

- Always the same `ex`.
- Tiny-Jev additionally ANDs this with causality.

Why `<ref>` sees only its own segment: segment i's tokens already carry context from the other
chunks, because they attend to the whole state in the encoder. The `<ref>` then knows exactly
which chunk it is judging. "All-state" view is ablation A6.

**Position restart:**

- Each decision block starts at position `S` (the state length).
- In isolated mode, every option span starts at `S + Lq` (Lq = question-span length).
- All `<ref>` tokens sit at `S + Lq`.

With the mask, this makes each decision's computation **identical** to running it alone.
That's the invariance property G2, and the unit test in §6.4 checks it.

### 3.7 Truncation (decisions are never truncated)

```
budget = max_len − (tokens of all decision blocks) − overhead
```

1. Header keeps at most 256 tokens (head + tail if longer).
2. If the segments exceed the remaining budget, shrink each segment to the same token cap
   (binary search on the cap; minimum 32 tokens per segment). Each segment keeps its head.
3. If that still doesn't fit:
   - drop segments that no segment-scope decision targets;
   - then split into several passes. Merging back is exact for segment-scope decisions. For
     global decisions it isn't, so the `Decider` warns.

### 3.8 Reference implementation: masks (shared code, `jevcore/packing.py`)

```python
import torch

REF_BASE = 1_000_000

def allowed(ex, blk, prt, valid, isolated=True, causal=False):
    """Bool [T, T] for one row: True where query q may attend key k."""
    Q = lambda t: t[:, None]; K = lambda t: t[None, :]
    T = blk.shape[0]
    same_ex = Q(ex) == K(ex)
    q_state, k_state = Q(blk) == 0, K(blk) == 0
    q_ref = Q(prt) >= REF_BASE
    ref_seg = Q(prt) - REF_BASE + 1                      # state segment i has prt = i + 1
    see_state = k_state & (q_state | ~q_ref | (K(prt) == 0) | (K(prt) == ref_seg))
    same_dec = ~q_state & (Q(blk) == K(blk))
    k_is_opt = (K(prt) > 0) & (K(prt) < REF_BASE)
    k_opt = k_is_opt & ((K(prt) == Q(prt)) if isolated else ~q_ref)
    k_self_ref = (K(prt) >= REF_BASE) & (K(prt) == Q(prt))
    see_dec = same_dec & ((K(prt) == 0) | k_opt | k_self_ref)
    m = same_ex & (see_state | see_dec) & Q(valid.bool()) & K(valid.bool())
    if causal:
        idx = torch.arange(T, device=blk.device)
        m &= K(idx) <= Q(idx)
    return m | torch.eye(T, dtype=torch.bool, device=blk.device)


def local_from_global(m, blk, half_window=64):
    """ModernBERT sliding layers: state tokens keep their ±64 window; decision tokens stay global."""
    idx = torch.arange(blk.shape[0], device=blk.device)
    near = (idx[:, None] - idx[None, :]).abs() <= half_window
    return m & (near | (blk[:, None] != 0))
```

Batch masks are `[B, 1, T, T]` bool (True = attend), passed as:

```python
attention_mask = {"full_attention": full, "sliding_attention": local}
```

### 3.9 Reference implementation: renderer (sketch)

```python
def render(example, tok, M, max_len, isolated=True, causal=False):
    """M: marker name -> token id. Returns Row (ids, pos, ex, blk, prt, valid) and groups."""
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    st = example["state"]
    header, segs = truncate_state(st, example["decisions"], tok, max_len)     # §3.7
    row = Row()
    row.push(([tok.cls_token_id] if not causal else []) + [M["state"]] + enc(header), blk=0, prt=0)
    for i, s in enumerate(segs):
        row.push([M["seg"]] + enc(f"[{i+1}] {s['title']}: {s['text']}"), blk=0, prt=i + 1)
    if not causal:
        row.push([tok.sep_token_id], blk=0, prt=0)
    S = len(row)                                          # state positions are 0..S-1
    groups = []
    for d, dec in enumerate(example["decisions"], start=1):
        q_ids = [M["dec"], M["q"]] + enc(dec["question"]) + ([M["qe"]] if causal else [])
        q0 = row.push(q_ids, blk=d, prt=0, pos_start=S)
        Lq = len(q_ids)
        anchor = q0 + 1 if not causal else q0 + Lq - 1    # <q> (Micro) or <qe> (Tiny)
        opt_tok, nxt = [], S + Lq
        for j, o in enumerate(dec["options"]):
            o_ids = [M["opt"]] + enc(o) + ([M["oe"]] if causal else [])
            start = S + Lq if isolated else nxt
            o0 = row.push(o_ids, blk=d, prt=j + 1, pos_start=start)
            opt_tok.append(o0 if not causal else o0 + len(o_ids) - 1)   # <opt> or <oe>
            nxt = start + len(o_ids)
        if dec["scope"] == "global":
            groups.append(Group(dec["name"], anchor, opt_tok, dec.get("label", -100), seg=-1))
        else:
            for t, i in enumerate(dec["targets"]):
                r = row.push([M["ref"]], blk=d, prt=REF_BASE + i, pos_start=S + Lq)
                lab = dec["labels"][t] if "labels" in dec else -100
                groups.append(Group(dec["name"], r, opt_tok, lab, seg=i))
    return row, groups
```

`Row.push(ids, blk, prt, pos_start=None)` appends the tokens and returns the index of the first
one. `pos` continues sequentially, from `pos_start` when given.

## 4. Model

### 4.1 Overview

```
packed row ──► ModernBERT-base (sdpa, dict mask, position_ids)  ──► H [B, T, 768]
                    │
     anchors a_g = H[b_g, anchor_g]        options o_gj = H[b_g, opt_gj]
                    │
              PairScorer(a_g, o_gj) ─► z_gj   (−inf for padded options)
                    │
         per-group softmax(z_g / T_name) ─► {option: p}
```

### 4.2 Backbone wrapper

```python
class MicroJev(nn.Module):
    def __init__(self, base="answerdotai/ModernBERT-base", n_vocab=None):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base, attn_implementation="sdpa")
        if n_vocab and n_vocab > self.enc.get_input_embeddings().num_embeddings:
            self.enc.resize_token_embeddings(n_vocab, pad_to_multiple_of=64)
        self.head = PairScorer(self.enc.config.hidden_size)

    def forward(self, input_ids, position_ids, full_mask, local_mask,
                g_batch, g_anchor, g_opts, g_opt_valid):
        h = self.enc(input_ids=input_ids, position_ids=position_ids,
                     attention_mask={"full_attention": full_mask,
                                     "sliding_attention": local_mask}).last_hidden_state
        a = h[g_batch, g_anchor]                          # [G, d]
        o = h[g_batch[:, None], g_opts]                   # [G, Kmax, d]
        z = self.head(a, o)                               # [G, Kmax]
        return z.masked_fill(~g_opt_valid, float("-inf"))
```

### 4.3 Global attention for decision tokens (in sliding layers)

In ModernBERT's local layers, a token sees only ±64 positions. Decision tokens come after a state
of up to ~8k tokens. With the plain window, they would see the state only in the 7 global layers.

`local_from_global` (§3.8) therefore makes **decision tokens global in every layer**. State tokens
keep their pretrained ±64 window.

- Cost: (number of decision tokens) × S. Small, since decision tokens are a few hundred at most.
- Ablation A3 tests the plain window.

### 4.4 Pair-scoring head (shared with Tiny-Jev)

```python
class PairScorer(nn.Module):
    """logit(anchor, option) = MLP(Wa·a + Wo·o + (Wa·a) ⊙ (Wo·o))"""
    def __init__(self, d, h=None, p=0.1):
        super().__init__()
        h = h or d
        self.wa, self.wo = nn.Linear(d, h), nn.Linear(d, h)
        self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, 1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)  # start uniform

    def forward(self, a, o):                              # a [G, d], o [G, K, d]
        ha, ho = self.wa(a)[:, None], self.wo(o)
        return self.mlp(ha + ho + ha * ho).squeeze(-1)
```

- About 1.8M parameters at d = 768.
- The zero-initialized last layer means the model starts at uniform probabilities (NLL = log K).
  That gives a clean training start and a sanity check.
- Ablation A5: `linear(o)` only (Nano-style).

### 4.5 What is trained

| Part | Trained | Learning rate |
|---|---|---|
| Whole backbone | Yes (full fine-tune; 149M fits easily) | 5e-5 |
| Head + new token embeddings | Yes | 5e-4 |

## 5. Training

### 5.1 Data phases

**Phase A ("parity")** uses exactly Nano v1.0's source questions, so architecture is the only
variable.

| Source | Questions | Packs built | Decisions in a pack |
|---|---|---|---|
| HotpotQA distractor (train) | 6,000 | 12,000: "full" (both gold + distractors → sufficient = yes) and "minus-one-gold" (→ no) | `relevance` (segment scope, all chunks) + `sufficient` |
| SQuAD 2.0 (train) | 6,000 | 6,000, one segment | `sufficient` (+ `relevance`: answerable → directly answers, unanswerable → partially relevant; **ablate**, Nano didn't train this) |
| MultiNLI (train) | 8,000 | 8,000; header = premise | `grounded` (question carries the claim; entailment → yes) |

Hotpot labels follow Nano's `hotpot_examples`:

- gold paragraph containing the answer → `directly answers`;
- the other gold paragraph → `partially relevant`;
- distractor → `irrelevant`.

Splits:

- **Calib / test:** the same validation halves as Nano.
- **Held-out:** MuSiQue (answerable dev; CC BY 4.0) and VitaminC (test; CC BY-SA 3.0), rebuilt as
  packs by question id.
- Regenerate the Nano-format held-out rows **from the same raw pass**, and re-score Nano v1.0 on
  them, so every comparison is on identical (query, passage) pairs.

**Phase B ("data+")** scales the same tasks:

| Source | Amount |
|---|---|
| HotpotQA | 30k questions |
| 2WikiMultihopQA | 15k (Apache-2.0) |
| SQuAD 2.0 | 20k |
| MultiNLI | 30k |
| FEVER | 20k claim + evidence for `grounded` (check licence before release) |

**Phase C ("multi-domain")** adds the Cyber-Jev decisions (`http_attack`, `prompt_injection`,
`phishing_url`) to test one model across RAG and security.

### 5.2 Augmentation (train split only)

| Id | What | Rate |
|---|---|---|
| A-shuffle | Shuffle segment order | Always |
| A-k | Number of distractors k ~ U{0..8} (total ≤ 10) | Always (Hotpot) |
| A-subset | Keep each decision with p = 0.8 (at least one) | Always |
| A-optperm | Shuffle option order, remap labels (also for `score` kind; isolated mode is order-free anyway) | Always |
| A-verb | Swap the option wording from a fixed table: yes/no ↔ true/false ↔ supported/not supported; relevance levels ↔ "not relevant / somewhat relevant / fully answers" … | p = 0.3 |
| A-qpara | Swap the question for one of 5 training templates per decision | p = 0.3 |

Keep **2 templates and 1 option wording per decision held out of training** for the paraphrase
test (§6.3). Store all templates in `jevcore/templates.yaml`.

### 5.3 Loss (shared with Tiny-Jev)

```python
ce = F.cross_entropy(z, labels, ignore_index=-100, reduction="none")   # z [G, Kmax]
loss = sum(w[n] * ce[name == n].mean() for n in present) / sum(w[n] for n in present)
```

- Each decision name is averaged separately, so 10 relevance groups per pack don't drown out
  `sufficient`. Default `w = 1`.
- **No label smoothing and no class weights.** Both distort calibration, and temperature fixes
  calibration afterwards. Handle imbalance in reporting (macro-F1).

### 5.4 Hyperparameters

| | Value |
|---|---|
| Optimizer | AdamW, β = (0.9, 0.98), ε = 1e-6, weight decay 0.01 (none on norms / biases / embeddings) |
| LR | Backbone 5e-5 (sweep 3e-5 / 8e-5 on seed 0); head + markers 5e-4 |
| Schedule | 6% linear warmup, linear decay |
| Epochs | 3; keep the best by dev NLL (macro over decisions) |
| Batch | Token budget ~16k tokens per micro-batch, gradient accumulation to ~32 packs per step |
| Max length | Train 2048 (check the 99th length percentile in M1; raise to 4096 if > 1% truncated). Eval up to 8192 |
| Precision | bf16 autocast; gradient checkpointing on |
| Seeds | 0, 1, 2 for the headline runs |
| Hardware | RTX 3060 12 GB |

Memory check: 149M params with fp32 Adam is ~2.4 GB; masks `[8,1,2048,2048]` bool × 2 are 64 MB;
with checkpointing the rest fits. Throughput isn't estimated here; measure it in M2 and size phase B
from that.

### 5.5 Collation and row packing

- **Length bucketing:** sort by length within shuffled mega-batches of 64 × batch.
- **Several examples per row:** short examples (MNLI, SQuAD) are concatenated into one row up to
  `max_len`. `ex` keeps them apart in the mask, and positions restart per example. This is what
  makes the short tasks cheap.
- **Flattened groups:**
  - `g_batch`, `g_anchor`: `[G]`
  - `g_opts`: `[G, Kmax]`, padded with index 0
  - `g_opt_valid`: `[G, Kmax]` bool
  - `labels`: `[G]`
  - `names`: `[G]`

### 5.6 Calibration

- **Default:** fit one temperature per decision name on the calib split
  (`nanojev.calibration.fit_temperature` on group logits). Save it to `calibration.json`.
- **OOD option** (lesson from Cyber-Jev: calibration in domain didn't transfer out of domain):
  also fit on an OOD validation set disjoint from the held-out test (e.g. 500 MuSiQue *train*
  questions). Ship both:
  - `calibration.json` (in-domain);
  - `calibration.ood.json`;
  - plus a flag in `Decider.from_pretrained(..., calibration="ood")`.

## 6. Evaluation (shared protocol with Tiny-Jev)

### 6.1 Sets

| Set | Decisions | Purpose |
|---|---|---|
| Test (in-domain halves) | rel / suff / grounded | G3 |
| MuSiQue, VitaminC held-out | rel / suff / grounded | G4 |
| MuSiQue k-sweep: k ∈ {1, 2, 5, 10, 20} chunks | rel / suff | G6, H2 |
| Paraphrase test: held-out templates and option wordings | all | G7, H4 |
| Invariance probes (500 packs) | all | G2, H1, H3 |

### 6.2 Metrics

- **Per decision:**
  - accuracy, macro-F1;
  - NLL, Brier, ECE (15 bins), before and after temperature;
  - AUROC for `noul` decisions;
  - quadratic weighted kappa for `relevance` (ordinal).
- **Selective prediction:**
  - risk–coverage curve and AURC;
  - accuracy on auto-decided items at 0 / 10 / 20 / 30% escalated.

  This is what the cascade uses.
- Reuse `nanojev/report.py` and add the new columns.

### 6.3 Baselines and controls

| Id | System | Isolates |
|---|---|---|
| B-nano | Nano v1.0 (released weights) | Reference |
| B-pair | **ModernBERT-base per-pair cross-encoder** (Nano architecture, same data) | Backbone effect vs packing effect. **Essential control** |
| B-rerank | bge-reranker-v2-m3 (relevance only) | Strong off-the-shelf reranker |
| B-llm | Qwen3-4B prompted (existing Ollama setup) | LLM judge reference |
| B-k1 | Micro-Jev, each chunk packed alone | H2 |

### 6.4 Invariance tests (`tests/test_mask_invariance.py`, CI on random weights + eval on trained)

For each probe pack, compare decision d's probabilities across:

1. d alone;
2. d packed with all others;
3. decisions in reversed order;
4. options permuted (then un-permuted);
5. an extra unrelated decision added.

| Precision | Pass threshold (max \|Δp\|) |
|---|---|
| fp32 | ≤ 1e-3 |
| bf16 | ≤ 1e-2 |

For the no-mask ablation, report the same numbers; they are the H3 result.

### 6.5 Latency

- **Scenario S10:** query + 10 chunks × ~150 tokens. Decision set = 10 relevance + sufficient +
  grounded.
- **Configurations:**
  - GPU (3060), batch 1;
  - CPU (PyTorch; ONNX int8 later, as for Cyber-Jev);
  - batch 32 throughput.
- Report the median of 200 runs after 20 warm-ups for Nano v1.0 (34 pair passes), B-pair, and
  Micro-Jev (1 pass).
- Also report the **k-curve**: latency vs number of chunks.

### 6.6 Statistics

- 3 seeds; report mean ± std.
- Model-vs-model on identical items: paired bootstrap, 10,000 resamples, 95% CI.
- Claim a difference only when the CI excludes 0.

## 7. Ablations (paper table; priority order)

| Id | Change | Question |
|---|---|---|
| A10 | B-pair control | Is the gain from the backbone or from packing? |
| A1 | No mask (full attention) + no position restart | H3: is isolation needed? |
| A2 | Siblings options | Does seeing other options help accuracy? Cost: order sensitivity |
| A3 | Decision tokens use the ±64 window | Is §4.3 needed? |
| A6 | `<ref>` sees the whole state (segment chosen by "[i]" text) | Own-segment vs all-state readout |
| A4 | Mean-pool option span instead of the `<opt>` marker | Readout choice |
| A5 | `linear(o)` head | Pair head value |
| A7 | Mask on, no position restart | Restart's share of invariance |
| A9 | No A-verb / A-qpara | H4 |
| A8 | ModernBERT-large | Scale |

## 8. Inference API (`jevcore/decider.py`)

```python
import microjev
d = microjev.load()                       # HF "sdmlai/micro-jev", tag v0.1

res = d.run(query=q, passages=chunks,
            decisions=["relevance", "sufficient",
                       microjev.Q("Is the query time-sensitive?", ["yes", "no"])])
res["relevance"]    # [{'irrelevant': .., 'partially relevant': .., 'directly answers': ..}, …] per chunk
res["sufficient"]   # {'yes': .., 'no': ..}
res["custom_0"]     # {'yes': .., 'no': ..}

# Nano-compatible surface (drop-in):
d.relevance(q, passages); d.sufficient(q, passages); d.grounded(claim, ctx)
d.decide(question, options, state)
```

- `run` builds one pack, makes one forward pass, then applies `softmax(z / T_name)` per group.
- Custom decisions use the global temperature `T_custom`, fitted on the paraphrase-test calib half.
- Batched input (a list of states) packs several rows into one batch.

## 9. Repository layout and reuse

Create a new folder with its own git repo, like `cyber_jev`. Add it to `code_repo/.git/info/exclude`.

```
code_repo/jev_core/
  jevcore/
    schema.py            # PackedExample, DecisionSpec, validate(), builtin DECISIONS (RAG + cyber)
    templates.yaml       # question templates and option wordings (train / held-out marked)
    packing.py           # render(), truncate_state(), allowed(), local_from_global()   [§3]
    collate.py           # bucketing, row packing, group flattening                    [§5.5]
    heads.py             # PairScorer                                                    [§4.4]
    backbones/modernbert.py   # MicroJev                                                 [§4.2]
    backbones/qwen3.py        # TinyJev (doc 15)
    data/builders.py     # hotpot / squad2 / mnli / musique / vitaminc / 2wiki / fever → packs
    data/augment.py      # §5.2
    calibration.py report.py registry.py   # copied from nano_jev, extended
    decider.py           # §8
  scripts/  prepare_packed.py  train.py  evaluate.py  bench_latency.py  invariance.py  baselines.py
  tests/    test_packing.py  test_mask_invariance.py  test_collate.py  test_decider.py  test_schema.py
  configs/  micro_base.yaml  micro_large.yaml  tiny_0p6b.yaml
```

Reused from `nano_jev`:

- dataset loaders and labeling (`nanojev/data.py`: `hotpot_examples`, `musique_examples`, …);
- `calibration.py`, `report.py`, `registry.py`;
- the CLI structure.

Release: separate HF repos (`sdmlai/micro-jev`, `sdmlai/tiny-jev`), one shared code package.

## 10. Milestones and exit criteria

| M | Work | Exit criterion |
|---|---|---|
| M0 (1 evening) | Load ModernBERT-base on transformers 5.17; dict mask + `position_ids` path; `allowed()`; invariance test on random weights | fp32 \|Δp\| ≤ 1e-4 on random weights; no NaN with padding |
| M1 (1 weekend) | Schema, templates, phase-A builders, held-out rebuild, Nano re-scored on rebuilt held-out | `prepare_packed.py` writes train/calib/test/heldout; length stats; Nano numbers reproduced ± 0.5 pt |
| M2 | Collate + training loop; overfit 200 packs; phase-A run, seed 0 | Overfit loss < 0.05; dev NLL beats B-pair after 1 epoch, or a documented reason why not |
| M3 | Full evaluation suite + latency + invariance; seeds 1–2; B-pair, B-k1 | Report table with CIs; G1–G3 checked |
| M4 | Ablations A1, A2, A3, A10 (then the rest) | Ablation table |
| M5 | Phase B, then phase C | G4 / G7 progress; multi-domain numbers |
| M6 | Package, model card, HF `v0.1`, paper draft | `pip install micro-jev` works; card reports held-out and OOD calibration |

## 11. Risks

| Risk | Mitigation |
|---|---|
| transformers mask API changes | Pin 5.17.0; invariance test in CI |
| SDPA with arbitrary masks is slower / uses more memory at 8k | Length bucketing; masks are bool; FlexAttention on WSL2 if needed |
| Global decision tokens in local layers differ from pretraining | Full fine-tune adapts; ablation A3 |
| Relevance is ~80% `irrelevant` | Report macro-F1 and QWK; no class weights (calibration) |
| Held-out gap stays large (Nano: −18 pts) | Phase B data; OOD temperature; H2 test tells whether context helps |
| Single-chunk SQuAD relevance labels are noisy | Train it only in an ablation first |
| Native Windows: no Triton / FlexAttention / compile | SDPA path only; WSL2 for long runs |

## 12. Paper framing

**Proposed title:** _Micro-Jev: Exact, Order-Invariant Packed Decisions for Calibrated RAG Control
in One Encoder Pass_.

**Contributions:**

1. A block-isolation mask plus position restart that makes packed decisions exactly equal to
   separate ones, at about 1/k the cost.
2. `<ref>` readout for per-chunk decisions.
3. Calibration in domain and OOD.
4. A controlled comparison against a per-pair cross-encoder on the same backbone and data.

**Related work** (verify each citation before use):

- Parallel Context Windows (Ratner et al., ACL 2023): position reuse with block masks.
- Set-Encoder (Schlatt et al., ECIR 2025): permutation-invariant inter-passage attention for
  reranking.
- Fusion-in-Decoder (Izacard & Grave, 2021).
- Self-RAG, CRAG, Adaptive-RAG.
- ModernBERT (Warner et al., 2024).
- JEV-as-a-Judge (arXiv 2609.26550).

## Appendix A — `configs/micro_base.yaml`

```yaml
model: {backbone: answerdotai/ModernBERT-base, attn: sdpa, decision_global: true,
        option_mode: isolated, ref_view: own, readout: marker, head: pair, head_dropout: 0.1}
data:  {phase: A, max_len_train: 2048, max_len_eval: 8192, row_packing: true,
        aug: {shuffle: true, k_distractors: [0, 8], subset_p: 0.8, optperm: true, verb_p: 0.3, qpara_p: 0.3}}
train: {lr_backbone: 5.0e-5, lr_head: 5.0e-4, wd: 0.01, betas: [0.9, 0.98], eps: 1.0e-6,
        warmup: 0.06, epochs: 3, tokens_per_microbatch: 16384, packs_per_step: 32,
        bf16: true, grad_ckpt: true, seed: 0, select_by: dev_nll}
calib: {per_decision: true, ood_set: musique_train_500}
```
