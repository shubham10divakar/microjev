"""End-to-end smoke test of the scripts on CPU: tiny local model + synthetic packs, no network."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import sample_example, tiny_model

from jevcore.report import write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _packs(prefix, n):
    out = []
    for i in range(n):
        e = sample_example(n_seg=1 + i % 3)
        e["id"] = f"{prefix}{i}"
        e["state"]["header"] += f" {i % 20}"
        e["decisions"][0]["labels"] = [(i + k) % 3 for k in range(len(e["decisions"][0]["targets"]))]
        e["decisions"][1]["label"] = i % 2
        e["decisions"][2]["label"] = (i + 1) % 2
        e["meta"] = {"query": "are there birds that cannot fly ?", "claim": "penguins cannot fly ."}
        out.append(e)
    return out


def run(*args, cwd):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ws")
    model, tok, M = tiny_model()
    model.save(tmp / "tiny", tok, {"max_len_eval": 512})
    for split, n in (("train", 24), ("calib", 16), ("test", 16)):
        write_jsonl(_packs(split, n), tmp / "data" / f"{split}.jsonl")
    return tmp


def test_train_evaluate_invariance_bench(workspace):
    tmp, s = workspace, ROOT / "scripts"
    out = run(s / "train.py", "--data", tmp / "data", "--out", tmp / "run", "--device", "cpu",
              "--max-steps", "2", "--dev-size", "8",
              "--set", f"model.backbone={(tmp / 'tiny').as_posix()}", "data.max_len_train=512",
              "train.tokens_per_microbatch=4096", "train.packs_per_step=4", cwd=tmp)
    assert (tmp / "run" / "head.pt").exists(), out
    out = run(s / "evaluate.py", "--model", tmp / "run", "--data", tmp / "data", "--max-len", "512",
              "--fp32", "--device", "cpu", "--save-preds", cwd=tmp)
    assert "relevance" in out and (tmp / "run" / "calibration.json").exists()
    preds = (tmp / "run" / "eval" / "results.preds.jsonl").read_text().splitlines()
    assert len(preds) > 16
    out = run(s / "evaluate.py", "--model", tmp / "run", "--data", tmp / "data", "--max-len", "512",
              "--fp32", "--device", "cpu", "--k", "1", "--name", "k1", "--save-preds", cwd=tmp)
    k1 = [json.loads(x) for x in (tmp / "run" / "eval" / "k1.preds.jsonl").read_text().splitlines()]
    full = [json.loads(x) for x in preds]
    key = lambda r: (r["pack"], r["name"], r["seg"])  # noqa: E731
    assert {key(r) for r in k1} <= {key(r) for r in full}       # k-sweep maps back to pack segments
    out = run(s / "compare.py", tmp / "run" / "eval" / "results.preds.jsonl",
              tmp / "run" / "eval" / "k1.preds.jsonl", "--n", "200", cwd=tmp)
    assert "relevance" in out
    out = run(s / "invariance.py", "--model", tmp / "run", "--data", tmp / "data", "--n", "4",
              "--max-len", "512", "--device", "cpu", cwd=tmp)
    assert out.count("| yes |") == 6, out
    out = run(s / "bench_latency.py", "--model", tmp / "run", "--device", "cpu", "--k", "1", "2",
              "--runs", "2", "--warmup", "1", "--chunk-tokens", "30", cwd=tmp)
    assert "| 2 |" in out
