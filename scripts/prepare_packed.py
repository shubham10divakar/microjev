"""Download datasets and write packed JSONL files (design §5.1, M1).

    python scripts/prepare_packed.py --preset default --out data          # train / calib / test
    python scripts/prepare_packed.py --heldout --out data_heldout         # MuSiQue + VitaminC test
    python scripts/prepare_packed.py --ood --out data_ood                 # 500 MuSiQue *train* (OOD calibration)
    python scripts/prepare_packed.py --ksweep --out data_ksweep           # MuSiQue, all ~20 paragraphs (G6, H2)
    python scripts/prepare_packed.py --length-stats data/train.jsonl      # rendered length percentiles

Every split is also exported in Nano format (<out>/nano/*.jsonl) so Nano v1.0 and the B-pair
control are scored on identical items.
"""

import argparse
import random
from collections import Counter

import _common  # noqa: F401
from _common import load_config

from jevcore.data.builders import build_heldout, build_phase_a, musique_packs, to_nano_rows
from jevcore.report import read_jsonl, write_jsonl
from jevcore.schema import labeled_groups, validate

PRESETS = {
    # hotpot_* counts questions (each gives a "full" and a "minus-one-gold" pack)
    "smoke": dict(hotpot_train=200, squad_train=200, mnli_train=200,
                  hotpot_eval=50, squad_eval=50, mnli_eval=50),
    "default": dict(hotpot_train=6000, squad_train=6000, mnli_train=8000,     # = Nano v1.0
                    hotpot_eval=500, squad_eval=500, mnli_eval=500),
}
HELDOUT = dict(musique=1000, vitaminc=1000)


def summary(name, packs):
    counts = Counter((n, lab) for p in packs for n, _, lab in labeled_groups(p))
    print(f"{name}: {len(packs)} packs, {sum(counts.values())} labelled groups")
    for (n, lab), c in sorted(counts.items()):
        print(f"    {n:<11} {lab:<3} {c}")


def dedupe_nano(rows):
    """Train minus-one-gold packs repeat (passage, label) relevance rows of their full pack;
    the Nano-format view (B-pair training) keeps each distinct row once."""
    seen, out = set(), []
    for r in rows:
        k = (r["decision"], r["question"], r["state"], r["label"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def write(out, splits: dict):
    for name, packs in splits.items():
        for p in packs:
            validate(p)
        write_jsonl(packs, f"{out}/{name}.jsonl")
        write_jsonl(dedupe_nano(r for p in packs for r in to_nano_rows(p)), f"{out}/nano/{name}.jsonl")
        summary(name, packs)


def length_stats(path, tokenizer, max_lens=(2048, 4096, 8192)):
    import numpy as np
    from transformers import AutoTokenizer

    from jevcore.packing import PackConfig, add_markers, render
    tok = AutoTokenizer.from_pretrained(tokenizer)
    M = add_markers(tok)
    packs = read_jsonl(path)
    lens = np.array([len(render(p, tok, M, PackConfig(max_len=10**6))) for p in packs])
    print(f"{path}: {len(lens)} packs, rendered length (no truncation)")
    for q in (50, 90, 95, 99, 100):
        print(f"    p{q:<3} {int(np.percentile(lens, q))}")
    for m in max_lens:
        print(f"    > {m}: {(lens > m).mean() * 100:.2f}% truncated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="default", choices=PRESETS)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=str(_common.ROOT / "configs/micro_base.yaml"))
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--ood", action="store_true", help="500 MuSiQue train questions for OOD T")
    ap.add_argument("--ksweep", action="store_true", help="MuSiQue dev with all paragraphs")
    ap.add_argument("--eval-distractors", type=int, default=1,
                    help="distractors per eval pack (1 = Nano's composition)")
    ap.add_argument("--squad-relevance", action="store_true", help="ablation: SQuAD relevance labels")
    ap.add_argument("--length-stats", metavar="JSONL")
    ap.add_argument("--tokenizer", default="answerdotai/ModernBERT-base")
    args = ap.parse_args()

    if args.length_stats:
        length_stats(args.length_stats, args.tokenizer)
        return
    if args.heldout:
        write(args.out, {"test": build_heldout(HELDOUT, args.seed, args.eval_distractors)})
    elif args.ood:
        packs = musique_packs(500, random.Random(args.seed), args.seed, train=True,
                              n_distract=args.eval_distractors)
        write(args.out, {"calib": packs})
    elif args.ksweep:
        packs = musique_packs(HELDOUT["musique"], random.Random(args.seed), args.seed,
                              n_distract=None)
        write(args.out, {"test": packs})
    else:
        cfg = load_config(args.config)
        assert cfg["data"]["phase"] == "A", "only phase A is implemented"
        write(args.out, build_phase_a(PRESETS[args.preset], args.seed, args.eval_distractors,
                                      args.squad_relevance))


if __name__ == "__main__":
    main()
