"""Latency (design §6.5; G1, H1). Weights don't affect latency, so --base works before training.

    python scripts/bench_latency.py --base answerdotai/ModernBERT-base --device cuda
    python scripts/bench_latency.py --model runs/micro-a-s0 --device cpu --runs 50
    python scripts/bench_latency.py --base answerdotai/ModernBERT-base --nano v1.0 --k 1 2 5 10 20

Scenario S10: query + k chunks of ~150 tokens; decision set = k relevance + sufficient + grounded.
Systems:
    micro        one packed pass (Micro-Jev)
    micro_b32    32 such packs in token-budget batches; reported per pack (throughput)
    pair         B-pair cost: one (question + option, state) sequence per option through the same
                 encoder, all 3k + 4 pairs batched (Nano architecture on ModernBERT)
    nano         Nano v1.0 via its own Decider (3k + 4 pair passes), if --nano is given
Median of --runs after --warmup.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import _common
import torch
from _common import device_arg, nano_path

from jevcore.backbones.modernbert import MicroJev
from jevcore.collate import collate, make_rows, to_device, token_batches
from jevcore.decider import Decider
from jevcore.packing import PackConfig, render
from jevcore.schema import DECISIONS
from jevcore.scoring import autocast

SENT = ("The committee reviewed the annual report and noted that revenue grew in every region "
        "except the north, where new competitors entered the market during the second quarter. ")
QUERY = "Which region did not grow last year and why?"
CLAIM = "Revenue fell in the north because of new competitors."


def chunk(tok, n_tokens=150):
    ids = tok(SENT * 8, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, runs, warmup):
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(1000 * (time.perf_counter() - t0))
    return statistics.median(ts)


def micro_fn(d: Decider, packs, bf16):
    cfg = d.cfg
    rows = make_rows([render(p, d.tok, d.M, cfg) for p in packs], cfg.max_len, row_packing=False)
    batches = [to_device(collate(b, d.tok.pad_token_id, cfg), d.device)
               for b in token_batches(rows, 10**9 if len(packs) == 1 else 65536)]

    @torch.no_grad()
    def run():
        for b in batches:
            with autocast(d.device, bf16):
                d.model(**b)
    return run


def pair_fn(d: Decider, k: int, passages, bf16, max_len=8192):
    """B-pair: (question + option, state) sequences, Nano-style, through the same encoder."""
    tok, enc = d.tok, d.model.enc
    firsts, seconds = [], []
    rel, suf, grd = DECISIONS["relevance"], DECISIONS["sufficient"], DECISIONS["grounded"]
    state = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    for p in passages:
        for o in rel.options:
            firsts.append(f"question: {rel.question} query: {QUERY} option: {o}")
            seconds.append(p)
    for spec, q in ((suf, f"{suf.question} query: {QUERY}"), (grd, grd.question.format(claim=CLAIM))):
        for o in spec.options:
            firsts.append(f"question: {q} option: {o}")
            seconds.append(state)
    # bucket by length: relevance pairs are short, sufficient / grounded pairs carry all chunks
    batches = []
    for lo, hi in ((0, 3 * k), (3 * k, len(firsts))):
        b = tok(firsts[lo:hi], seconds[lo:hi], truncation="only_second", max_length=max_len,
                padding=True, return_tensors="pt")
        batches.append({kk: v.to(d.device) for kk, v in b.items() if kk in ("input_ids", "attention_mask")})

    @torch.no_grad()
    def run():
        for b in batches:
            with autocast(d.device, bf16):
                enc(**b)
    return run, len(firsts)


def nano_fn(version, device, passages):
    sys.path.insert(0, str(nano_path()))
    import nanojev
    nd = nanojev.Decider.from_pretrained(version, device=device)

    def run():
        nd.relevance(QUERY, passages)
        nd.sufficient(QUERY, passages)
        nd.grounded(CLAIM, "\n".join(passages))
    return run


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--base")
    ap.add_argument("--k", type=int, nargs="+", default=[10])
    ap.add_argument("--chunk-tokens", type=int, default=150)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--nano", help="Nano-Jev version / path to compare (e.g. v1.0)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--device")
    args = ap.parse_args()

    device = device_arg(args.device)
    if args.model:
        d = Decider.from_pretrained(args.model, device, max_len=8192)
    else:
        model, tok, M = MicroJev.from_base(args.base)
        cfg = PackConfig(max_len=8192, half_window=model.half_window)
        d = Decider(model.to(device).eval(), tok, M, cfg, {}, device)
    bf16 = not args.fp32 and torch.device(device).type == "cuda"
    passage = chunk(d.tok, args.chunk_tokens)

    results = []
    print(f"device {device}, bf16 {bf16}, median of {args.runs} after {args.warmup} warm-ups\n")
    print("| k | micro 1 pass (ms) | micro batch-32 (ms/pack) | B-pair (ms, n pairs) | Nano (ms) | pair / micro |")
    print("|---|---|---|---|---|---|")
    for k in args.k:
        passages = [passage] * k
        pack = d.pack(["relevance", "sufficient", "grounded"], query=QUERY, passages=passages,
                      claim=CLAIM)
        r = {"k": k, "pack_tokens": len(render(pack, d.tok, d.M, d.cfg))}
        r["micro_ms"] = timed(micro_fn(d, [pack], bf16), device, args.runs, args.warmup)
        packs32 = [{**pack, "id": f"p{i}"} for i in range(32)]
        r["micro_b32_ms_per_pack"] = timed(micro_fn(d, packs32, bf16), device,
                                           max(3, args.runs // 20), 2) / 32
        fn, n_pairs = pair_fn(d, k, passages, bf16)
        r["pair_ms"], r["pair_passes"] = timed(fn, device, args.runs, args.warmup), n_pairs
        if args.nano:
            r["nano_ms"] = timed(nano_fn(args.nano, device, passages), device, args.runs, args.warmup)
        results.append(r)
        print(f"| {k} | {r['micro_ms']:.1f} | {r['micro_b32_ms_per_pack']:.1f} "
              f"| {r['pair_ms']:.1f} ({n_pairs}) | {r.get('nano_ms', float('nan')):.1f} "
              f"| {r['pair_ms'] / r['micro_ms']:.1f}× |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"device": device, "bf16": bf16, "results": results},
                                             indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
