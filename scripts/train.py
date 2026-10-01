"""Train Micro-Jev (design §5, M2).

    python scripts/train.py --config configs/micro_base.yaml --data data --out runs/micro-a-s0
    python scripts/train.py --data data --out runs/overfit --overfit 200        # M2 sanity: loss < 0.05
    python scripts/train.py --data data --out runs/smoke --max-steps 20 --dev-size 50
    python scripts/train.py --data data --out runs/a1 --set model.mask=full model.position_restart=false   # ablation A1

The best epoch by macro dev NLL (on a slice of calib) is saved to --out.
"""

import argparse
import json
import random
from pathlib import Path

import _common
import yaml
from _common import device_arg, load_config

from jevcore.backbones.modernbert import MicroJev
from jevcore.data.augment import AugConfig
from jevcore.packing import PackConfig
from jevcore.report import read_jsonl
from jevcore.schema import SCHEMA_VERSION
from jevcore.trainer import TrainConfig, train


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """--set section.key=value (value parsed as YAML)."""
    for s in sets or []:
        path, _, value = s.partition("=")
        node = cfg
        *parents, key = path.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[key] = yaml.safe_load(value)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_common.ROOT / "configs/micro_base.yaml"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--dev-size", type=int, default=1500, help="calib packs used for dev NLL")
    ap.add_argument("--overfit", type=int, metavar="N",
                    help="train on N train packs, no augmentation, constant LR; dev = same packs")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr_backbone=3e-5")
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    tcfg = cfg["train"]
    if args.seed is not None:
        tcfg["seed"] = args.seed
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.max_steps is not None:
        tcfg["max_steps"] = args.max_steps
    tc = TrainConfig.from_dict(tcfg)
    device = device_arg(args.device)
    if device == "cpu":
        tc.bf16 = False

    train_packs = read_jsonl(Path(args.data) / "train.jsonl")
    calib = read_jsonl(Path(args.data) / "calib.jsonl")
    dev = random.Random(tc.seed).sample(calib, min(args.dev_size, len(calib)))
    aug = AugConfig.from_dict(cfg["data"].get("aug", {}))
    if args.overfit:
        train_packs = random.Random(tc.seed).sample(train_packs, args.overfit)
        dev, aug = train_packs, None
        tc.constant_lr, tc.warmup, tc.packs_per_step = True, 0.0, min(tc.packs_per_step, 8)
        tc.epochs = args.epochs or 30

    model, tok, M = MicroJev.from_base(cfg["model"]["backbone"], cfg["model"])
    pc = PackConfig.from_model_cfg(model.cfg, cfg["data"]["max_len_train"])
    pc.half_window = model.half_window

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"{len(train_packs)} train packs, {len(dev)} dev packs, device {device}, "
          f"max_len {pc.max_len}, tc={tc}")
    train(model, tok, M, train_packs, dev, pc, tc, out, device, aug,
          row_packing=cfg["data"].get("row_packing", True),
          extra_save={"schema_version": SCHEMA_VERSION, "max_len_train": pc.max_len,
                      "max_len_eval": cfg["data"].get("max_len_eval", 8192), "seed": tc.seed,
                      "overfit": args.overfit})


if __name__ == "__main__":
    main()
