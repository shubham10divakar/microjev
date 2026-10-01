"""M0 on the real backbone (downloads answerdotai/ModernBERT-base, ~600 MB).

    MICROJEV_NETWORK_TESTS=1 python -m pytest tests/test_network.py -v
"""

import pytest
import torch
from conftest import network_enabled, sample_example

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(not network_enabled(), reason="set MICROJEV_NETWORK_TESTS=1")]

BASE = "answerdotai/ModernBERT-base"


def test_config_matches_design():
    """Design §2 table: check every value against the downloaded config."""
    from transformers import AutoConfig, AutoTokenizer
    c = AutoConfig.from_pretrained(BASE)
    assert (c.num_hidden_layers, c.hidden_size, c.num_attention_heads) == (22, 768, 12)
    assert c.max_position_embeddings == 8192
    assert c.global_attn_every_n_layers == 3 and c.local_attention == 128
    assert c.vocab_size == 50368
    tok = AutoTokenizer.from_pretrained(BASE)
    assert len(tok) <= c.vocab_size        # spare rows may hold the markers


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_invariance_pretrained(attn):
    from jevcore.backbones.modernbert import MicroJev
    from jevcore.invariance import probe
    from jevcore.packing import PackConfig
    torch.manual_seed(0)
    model, tok, M = MicroJev.from_base(BASE, {"attn": attn})
    for p in model.head.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.5)
    model.eval()
    cfg = PackConfig(max_len=2048, half_window=model.half_window)
    other = sample_example(n_seg=5, long=True)
    other["id"] = "other"
    diffs = probe(model, tok, M, sample_example(long=True), other, cfg)
    assert all(d <= 1e-4 for d in diffs.values()), diffs
