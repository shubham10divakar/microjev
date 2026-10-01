"""Baselines on identical items (design §6.3).

    # B-nano: re-score Nano v1.0 on the Nano-format rows exported by prepare_packed.py
    python scripts/baselines.py nano --data data --nano v1.0 --name test
    python scripts/baselines.py nano --data data_heldout --calib-data data --nano v1.0 --name heldout

    # B-pair: ModernBERT-base per-pair cross-encoder = Nano's architecture and trainer on the
    # same items. Train it with nano_jev's own script on the exported rows:
    python ../nano_jev/scripts/train.py --data data/nano --base answerdotai/ModernBERT-base \
        --out runs/b-pair --epochs 3 --lr 5e-5 --max-length 2048 --batch-size 8
    python scripts/baselines.py nano --data data --nano runs/b-pair --name b-pair-test

B-k1 (each chunk alone) is evaluate.py --k 1; B-rerank / B-llm reuse nano_jev/scripts/baselines.py
on data/nano rows.

Writes <out>/<name>.json/.md (same table as evaluate.py) and <name>.preds.jsonl keyed by
(pack, name, seg) for scripts/compare.py.
"""

import argparse
import sys
from pathlib import Path

import _common  # noqa: F401
import torch
from _common import device_arg, nano_path

from jevcore.report import decision_report, read_jsonl, render_table, save, write_jsonl


def nano_scorer(model_ref, device, max_length):
    """Load a Nano-architecture model once; return rows -> list of logit tensors."""
    sys.path.insert(0, str(nano_path()))
    from nanojev import model as NM
    from nanojev.decider import Decider
    nd = Decider.from_pretrained(model_ref, device=device)
    return lambda rows: NM.score(nd.model, nd.tok, rows, max_length, device)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nano", help="score a Nano-architecture model on <data>/nano rows")
    n.add_argument("--data", default="data", help="folder with nano/test.jsonl")
    n.add_argument("--calib-data", help="folder with nano/calib.jsonl (default: --data)")
    n.add_argument("--nano", default="v1.0", help="Nano version, Hub id or local folder")
    n.add_argument("--name", default="nano")
    n.add_argument("--out", default="results/baselines")
    n.add_argument("--max-length", type=int, default=512)
    n.add_argument("--device")
    args = ap.parse_args()

    device = device_arg(args.device)
    calib_rows = read_jsonl(Path(args.calib_data or args.data) / "nano" / "calib.jsonl")
    test_rows = read_jsonl(Path(args.data) / "nano" / "test.jsonl")
    score = nano_scorer(args.nano, device, args.max_length)
    report, preds = {}, []
    for dec in sorted({r["decision"] for r in test_rows}):
        c = [r for r in calib_rows if r["decision"] == dec]
        t = [r for r in test_rows if r["decision"] == dec]
        cz, tz = torch.stack(score(c)), torch.stack(score(t))
        cy = torch.tensor([r["label"] for r in c])
        ty = torch.tensor([r["label"] for r in t])
        report[dec] = decision_report((cz, cy), (tz, ty), name=dec)
        temp = report[dec]["temperature"]
        for r, z in zip(t, tz):
            p = torch.softmax(z / temp, 0)
            preds.append({"pack": r["pack_id"], "name": dec, "seg": r["seg"], "label": r["label"],
                          "pred": int(p.argmax()), "probs": p.tolist()})
    table = render_table(f"Nano architecture — `{args.nano}` on `{args.data}`", report)
    print(table)
    out = Path(args.out)
    save(report, out / f"{args.name}.json")
    (out / f"{args.name}.md").write_text(table + "\n", encoding="utf-8")
    write_jsonl(preds, out / f"{args.name}.preds.jsonl")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
