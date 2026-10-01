"""Invariance probes on trained weights (design §6.4; H1, H3).

    python scripts/invariance.py --model runs/micro-a-s0 --data data --n 500
    python scripts/invariance.py --model runs/micro-a-s0 --data data --n 500 --bf16
    python scripts/invariance.py --base answerdotai/ModernBERT-base --data data --n 50   # M0 on real pretrained weights, before training

Pass thresholds: max |Δp| <= 1e-3 (fp32), <= 1e-2 (bf16). For the no-mask ablation (A1) the
same numbers are the H3 result.
"""

import argparse
import json
import random
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from _common import device_arg

from jevcore.backbones.modernbert import MicroJev
from jevcore.invariance import fits_untruncated, probe
from jevcore.packing import PackConfig
from jevcore.report import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="trained Micro-Jev folder")
    g.add_argument("--base", help="pretrained backbone (random head) for M0")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--attn", default="sdpa", help="--base only: sdpa | eager")
    ap.add_argument("--out", help="write the summary JSON here")
    ap.add_argument("--device")
    args = ap.parse_args()

    device = device_arg(args.device)
    if args.model:
        model, tok, M, _ = MicroJev.load(args.model, device)
    else:
        model, tok, M = MicroJev.from_base(args.base, {"attn": args.attn})
        import torch
        torch.manual_seed(0)
        for p in model.head.parameters():          # zero-init last layer -> make it non-trivial
            if p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.5)
        model.to(device).eval()
    cfg = PackConfig.from_model_cfg(model.cfg, args.max_len)
    cfg.half_window = model.half_window

    packs = read_jsonl(Path(args.data) / f"{args.split}.jsonl")
    random.Random(0).shuffle(packs)
    probes = [p for p in packs if fits_untruncated(p, tok, M, cfg)]
    multi = [p for p in probes if len(p["decisions"]) > 1] or probes
    probes = multi[: args.n]
    others = probes[1:] + probes[:1]
    print(f"{len(probes)} probe packs ({len(packs)} in {args.split}), "
          f"{'bf16' if args.bf16 else 'fp32'}, model cfg {model.cfg}")

    rows = []
    for base, other in zip(probes, others):
        other = {**other, "id": f"{other['id']}~other"}
        rows.append(probe(model, tok, M, base, other, cfg, device, args.bf16))
    tol = 1e-2 if args.bf16 else 1e-3
    summary = {}
    print(f"\n| perturbation | mean max|Δp| | max |Δp| | pass (≤ {tol:g}) |\n|---|---|---|---|")
    for key in rows[0]:
        v = np.array([r[key] for r in rows])
        summary[key] = {"mean": float(v.mean()), "max": float(v.max()), "pass": bool(v.max() <= tol)}
        print(f"| {key} | {v.mean():.2e} | {v.max():.2e} | {'yes' if v.max() <= tol else 'NO'} |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"n": len(rows), "bf16": args.bf16, "tol": tol,
                                              "summary": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
