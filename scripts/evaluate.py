"""Fit per-decision temperatures on calib, report test metrics (design §5.6, §6, M3).

    python scripts/evaluate.py --model runs/micro-a-s0                          # in-domain test, writes calibration.json
    python scripts/evaluate.py --model runs/micro-a-s0 --test-data data_heldout --name heldout
    python scripts/evaluate.py --model runs/micro-a-s0 --ood-calib data_ood     # writes calibration.ood.json
    python scripts/evaluate.py --model runs/micro-a-s0 --test-data data_ksweep --k 1 --name k1   # B-k1 / H2
    python scripts/evaluate.py --model runs/micro-a-s0 --paraphrase 0 0 --name para_q0_o0      # G7 / H4

Temperatures always come from <data>/calib.jsonl (or --ood-calib) and are applied unchanged to
the test set. --save-preds writes per-group predictions for scripts/compare.py.
"""

import argparse
import json
import random
import time
from pathlib import Path

import _common  # noqa: F401
import torch
from _common import device_arg

from jevcore.backbones.modernbert import MicroJev
from jevcore.calibration import fit_temperature
from jevcore.data.augment import paraphrase
from jevcore.data.builders import chunk_packs
from jevcore.packing import PackConfig
from jevcore.report import decision_report, read_jsonl, render_table, save, split_by_name, write_jsonl
from jevcore.scoring import score_packs


def transform(packs, args):
    if args.k:
        rng = random.Random(0)
        packs = [c for p in packs for c in chunk_packs(p, args.k, rng)]
    if args.paraphrase:
        q, o = (None if x < 0 else x for x in args.paraphrase)
        packs = [paraphrase(p, q, o) for p in packs]
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data", help="folder with calib.jsonl (and test.jsonl)")
    ap.add_argument("--test-data", help="folder with the test.jsonl to report on (default: --data)")
    ap.add_argument("--ood-calib", help="folder with calib.jsonl from an OOD source -> calibration.ood.json")
    ap.add_argument("--name", default="results")
    ap.add_argument("--results-dir", help="default: <model>/eval")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--k", type=int, help="k-sweep: split packs into chunks of k segments (relevance only)")
    ap.add_argument("--paraphrase", type=int, nargs=2, metavar=("Q", "O"),
                    help="held-out question / option wording index (-1 keeps the training form)")
    ap.add_argument("--no-row-packing", action="store_true")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="don't write calibration files")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    device = device_arg(args.device)
    model, tok, M, saved = MicroJev.load(args.model, device)
    cfg = PackConfig.from_model_cfg(model.cfg, args.max_len)
    cfg.half_window = model.half_window
    score = lambda packs, timings=None: score_packs(  # noqa: E731
        model, tok, M, packs, cfg, device, args.tokens_per_batch,
        row_packing=not args.no_row_packing, bf16=not args.fp32, timings=timings)

    calib_dir = Path(args.ood_calib or args.data)
    calib = split_by_name(score(transform(read_jsonl(calib_dir / "calib.jsonl"), args)))
    test_packs = transform(read_jsonl(Path(args.test_data or args.data) / "test.jsonl"), args)
    timings: list[float] = []
    t0 = time.perf_counter()
    scored = score(test_packs, timings)
    wall = time.perf_counter() - t0
    test = split_by_name(scored)

    report = {}
    for name in sorted(test):
        if name not in calib:
            print(f"skip {name}: no calibration examples")
            continue
        report[name] = decision_report(calib[name], test[name], name=name)
    ms = 1000 * sum(timings) / max(len(test_packs), 1)
    title = f"Micro-Jev — `{args.model}` on `{args.test_data or args.data}` ({args.name})"
    table = render_table(title, report) + (
        f"\n\nforward {ms:.1f} ms/pack (batched), wall {wall:.1f}s for {len(test_packs)} packs")
    print(table)

    if not args.no_save and not args.k and not args.paraphrase:
        fname = "calibration.ood.json" if args.ood_calib else "calibration.json"
        temps = {n: fit_temperature(*calib[n]) for n in calib}
        # custom decisions use one shared temperature: the mean over the builtin ones until
        # the paraphrase-calib split exists (§8)
        temps.setdefault("custom", sum(temps.values()) / len(temps))
        (Path(args.model) / fname).write_text(json.dumps(temps, indent=2), encoding="utf-8")
        print(f"saved {fname}")
    out = Path(args.results_dir or Path(args.model) / "eval")
    save(report, out / f"{args.name}.json")
    (out / f"{args.name}.md").write_text(table + "\n", encoding="utf-8")
    if args.save_preds:
        temps = {n: r["temperature"] for n, r in report.items()}
        by_id = {p["id"]: p for p in test_packs}
        preds = []
        for s in scored:
            if s["label"] < 0 or s["name"] not in temps:
                continue
            p = torch.softmax(s["logits"] / temps[s["name"]], 0)
            orig = by_id[s["pack"]].get("meta", {}).get("orig_seg")   # k-sweep chunks
            seg = orig[s["seg"]] if orig and s["seg"] >= 0 else s["seg"]
            # key = (original pack id, decision, original segment): same as to_nano_rows
            preds.append({"pack": s["pack"].split("#")[0], "name": s["name"], "seg": seg,
                          "label": s["label"], "pred": int(p.argmax()), "probs": p.tolist()})
        write_jsonl(preds, out / f"{args.name}.preds.jsonl")
    print(f"saved {args.name}.json/.md to {out}")


if __name__ == "__main__":
    main()
