# Micro-Jev — aims and expectations

_Written 2026-10-01, before any training. Revisit after M3 (first full evaluation)._

## Where it sits

End goal (from `code_repo/end goal.txt`): an open, offline "System One" decision model.
Unstructured state in, typed probabilistic decisions out, never free text.

- **Nano-Jev v1.0** (33M, MiniLM-L12) proved the decision interface works and is calibrated
  in domain. Its weak points: one forward pass per (question, option, state), a short 512-token
  context, a ~15–20 pt drop on held-out data, and weak custom option sets.
- **Micro-Jev** (149M, ModernBERT-base) keeps the interface and changes the *execution*: a
  whole decision set (all chunk relevances + sufficient + grounded + custom questions) is read
  from **one** encoder pass over an 8k context, with exact isolation between decisions.
- **Tiny-Jev** (doc 15) is the next step for zero-shot generality. Micro-Jev is not trying to be that.

So the one-line aim: **Micro-Jev is the fast, exact, long-context RAG controller.**
One pass per RAG step, same calibrated probabilities you'd get by asking each question alone.

## What we expect (three levels)

Numbers are vs Nano-Jev v1.0 on the same items (test: 0.815 / 0.845 / 0.844 rel / suff / grd;
held-out: 0.635 / 0.670 / 0.675).

### Must (otherwise the design is wrong and we stop to rethink)

| | Expectation | Why we're confident |
|---|---|---|
| M-1 | Packed == separate: max \|Δp\| ≤ 1e-3 fp32 (G2) | It's a property of the mask + positions, testable on random weights before training |
| M-2 | One pass per RAG step at k = 10 (G1); ≥ 5× faster than the **same backbone** run per pair (B-pair) | 1 pass vs 34 passes over the same state |
| M-3 | In-domain accuracy ≥ Nano v1.0 on all three decisions (G3) | 4.5× bigger, newer backbone on the same data |
| M-4 | In-domain ECE ≤ 0.03 after temperature (G5) | Nano already does this; no label smoothing / class weights |

### Target (what we'd call a success worth a paper)

| | Expectation |
|---|---|
| T-1 | In-domain +2–4 pts over Nano on each decision |
| T-2 | Held-out MuSiQue relevance ≥ 0.69 (+5 pts) and sufficient ≥ 0.72; the k-sweep shows the pack beating k = 1 on multi-hop (H2) |
| T-3 | Micro ≈ B-pair on accuracy (within 1 pt, CI includes 0): packing costs nothing (H1) |
| T-4 | 20 chunks in one 8k pass with accuracy within 2 pts of k = 10 (G6) |
| T-5 | Paraphrase test: held-out wordings lose ≤ 3 pts vs training wordings (G7, H4) |
| T-6 | OOD temperature brings held-out ECE from Nano's ~0.17 to ≤ 0.08 |

### Stretch

- Grounded on VitaminC ≥ 0.75 after phase B (FEVER). The 4B LLM is at 0.787; Nano at 0.675.
- One model for RAG + Cyber-Jev decisions (phase C) with no more than 1 pt loss on either.
- CPU latency for a full S10 pack under ~150 ms (ONNX int8 later).

## Honest caveats (to write in the paper, not hide)

- **Wall-clock vs Nano is not the 34× story.** Nano is 4.5× smaller, and its 34 pair passes batch
  well on a GPU (~3 ms/decision batched). The clean speed claim is against B-pair (same backbone).
  Against Nano we expect roughly the same or a few × faster at k = 10, and better as k grows.
  Measure it, then state it.
- **Most of the in-domain gain may be the backbone, not packing.** B-pair (A10) is the control
  that tells us. If Micro ≈ B-pair, the claim is "same accuracy, k× cheaper, exactly consistent",
  which is still the point.
- **Held-out gap will not close.** Nano lost 15–20 pts. A bigger backbone and context should win
  back some of that, not all. Phase B data is the main lever.
- **Groundedness from MNLI transfers poorly** (VitaminC). Expect it to stay the weakest held-out
  number until FEVER is added.

## Kill / pivot criteria

- M-1 fails after debugging (the HF dict-mask path doesn't honour our masks): pin a different
  transformers version or write a small custom attention, before any training.
- After phase A seed 0: in-domain below Nano v1.0 on 2+ decisions **and** B-pair also below →
  the recipe (lr, length, data) is off; fix before ablations.
- Micro clearly below B-pair (CI excludes 0): packing hurts. Then try A2 (siblings) / A6
  (`<ref>` sees all state) before going further.
