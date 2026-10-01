"""Paired bootstrap between two systems on identical items (design §6.6).

    python scripts/compare.py runs/micro-a-s0/eval/test.preds.jsonl results/baselines/nano.preds.jsonl

Items are matched by (pack, decision, segment). A difference is claimed only when the 95% CI
excludes 0.
"""

import argparse
import sys

import _common  # noqa: F401
import numpy as np

from jevcore.calibration import paired_bootstrap
from jevcore.report import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=10_000)
    args = ap.parse_args()

    key = lambda r: (r["pack"], r["name"], r["seg"])  # noqa: E731
    a = {key(r): r for r in read_jsonl(args.a)}
    b = {key(r): r for r in read_jsonl(args.b)}
    shared = a.keys() & b.keys()
    print(f"{len(a)} vs {len(b)} items, {len(shared)} shared")
    if not shared:
        sys.exit("no shared items: were both built from the same packs?")
    print("| decision | n | acc A | acc B | A − B | 95% CI | significant |\n|---|---|---|---|---|---|---|")
    for name in sorted({k[1] for k in shared}):
        ks = sorted(k for k in shared if k[1] == name)
        ca = np.array([a[k]["pred"] == a[k]["label"] for k in ks])
        cb = np.array([b[k]["pred"] == b[k]["label"] for k in ks])
        r = paired_bootstrap(ca, cb, args.n)
        print(f"| {name} | {len(ks)} | {ca.mean():.3f} | {cb.mean():.3f} | {r['diff']:+.3f} "
              f"| [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {'yes' if r['significant'] else 'no'} |")


if __name__ == "__main__":
    main()
