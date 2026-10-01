"""Design §6.4 / M0: a decision's probabilities don't depend on what it is packed with.

Runs on a tiny random ModernBERT (CPU, fp32). The same check on trained weights is
scripts/invariance.py.
"""

import copy
import random

import pytest
import torch
from conftest import sample_example, tiny_model

from jevcore.collate import collate
from jevcore.packing import PackConfig, pack_row, render

TOL = 1e-4   # M0 exit criterion (fp32, random weights)


def probs_of(model, tok, M, examples, cfg, rows_of=None):
    """{(example id, decision name, seg): {option: p}} from one batch."""
    rendered = [render(e, tok, M, cfg) for e in examples]
    ids = [e["id"] for e in examples]
    groups = rows_of or [[i] for i in range(len(examples))]
    rows = [pack_row([rendered[i] for i in g], [ids[i] for i in g]) for g in groups]
    b = collate(rows, tok.pad_token_id, cfg)
    with torch.no_grad():
        z = model(**b)
    p = torch.softmax(z, -1)
    by_id = {e["id"]: e for e in examples}
    out = {}
    for n in range(len(b["names"])):
        ex = by_id[b["g_example"][n]]
        dec = ex["decisions"][b["g_dec"][n]]
        out[(ex["id"], dec["name"], b["g_seg"][n])] = dict(zip(dec["options"], p[n].tolist()))
    return out


def max_diff(a, b):
    keys = a.keys() & b.keys()
    assert keys, "no shared groups"
    return max(abs(a[k][o] - b[k][o]) for k in keys for o in a[k])


def variants(base):
    """The §6.4 perturbations, each as a list of examples sharing ids with `base`."""
    rng = random.Random(0)
    out = {}
    out["alone"] = []
    for d in base["decisions"]:
        e = copy.deepcopy(base)
        e["decisions"] = [copy.deepcopy(d)]
        e["id"] = f"{base['id']}"
        out["alone"].append(e)
    rev = copy.deepcopy(base)
    rev["decisions"] = rev["decisions"][::-1]
    out["reversed"] = rev
    perm = copy.deepcopy(base)
    for d in perm["decisions"]:
        order = list(range(len(d["options"])))
        rng.shuffle(order)
        d["options"] = [d["options"][i] for i in order]
    out["options_permuted"] = perm
    extra = copy.deepcopy(base)
    extra["decisions"].insert(1, {"name": "extra", "kind": "choice", "scope": "global",
                                  "question": "is it time sensitive ?",
                                  "options": ["yes", "no", "low", "high"]})
    out["extra_decision"] = extra
    return out


@pytest.fixture(params=["sdpa", "eager"])
def setup(request):
    return tiny_model(attn=request.param)


def test_invariance_random_weights(setup):
    model, tok, M = setup
    cfg = PackConfig(max_len=512, half_window=model.half_window)
    base = sample_example()
    ref = probs_of(model, tok, M, [base], cfg)
    v = variants(base)

    alone = {}
    for e in v["alone"]:
        alone.update(probs_of(model, tok, M, [e], cfg))
    diffs = {"alone": max_diff(ref, alone)}
    for name in ("reversed", "options_permuted", "extra_decision"):
        diffs[name] = max_diff(ref, probs_of(model, tok, M, [v[name]], cfg))

    # Row-packed after another example, and batched next to a longer row (padding).
    other = sample_example(n_seg=5, long=True)
    other["id"] = "other"
    packed = probs_of(model, tok, M, [other, base], cfg, rows_of=[[0, 1]])
    diffs["row_packed"] = max_diff(ref, packed)
    padded = probs_of(model, tok, M, [base, other], cfg)
    diffs["batched_padded"] = max_diff(ref, padded)

    assert all(d <= TOL for d in diffs.values()), diffs
    # sanity: the outputs are not trivially uniform
    assert max(max(p.values()) - min(p.values()) for p in ref.values()) > 1e-2


def test_no_mask_breaks_invariance():
    """H3 sanity: without the mask (A1) packing changes outputs, so the test above has teeth."""
    model, tok, M = tiny_model(mask="full", position_restart=False)
    cfg = PackConfig(max_len=512, mask="full", position_restart=False,
                     half_window=model.half_window)
    base = sample_example()
    ref = probs_of(model, tok, M, [base], cfg)
    rev = probs_of(model, tok, M, [variants(base)["reversed"]], cfg)
    assert max_diff(ref, rev) > 1e-3


def test_no_nan_with_padding():
    model, tok, M = tiny_model()
    cfg = PackConfig(max_len=512, half_window=model.half_window)
    short = sample_example(n_seg=1)
    short["id"] = "short"
    long = sample_example(n_seg=5, long=True)
    rows = [pack_row([render(short, tok, M, cfg)]), pack_row([render(long, tok, M, cfg)])]
    b = collate(rows, tok.pad_token_id, cfg)
    with torch.no_grad():
        h = model.encode(b["input_ids"], b["position_ids"], b["full_mask"], b["local_mask"])
    assert torch.isfinite(h).all()
