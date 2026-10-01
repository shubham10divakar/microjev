## Micro-Jev — `runs/micro-a-s0b` on `data_heldout` (heldout)

| decision | n | T | acc | macro-F1 | AUROC / QWK | NLL raw → cal | ECE raw → cal | AURC | acc @20% esc. |
|---|---|---|---|---|---|---|---|---|---|
| grounded | 1000 | 1.54 | 0.640 | 0.632 | 0.699 | 0.727 → 0.658 | 0.131 → 0.068 | 0.293 | 0.675 |
| relevance | 4655 | 1.25 | 0.618 | 0.567 | 0.363 | 1.322 → 1.154 | 0.192 → 0.158 | 0.291 | 0.663 |
| sufficient | 1040 | 1.56 | 0.720 | 0.715 | 0.819 | 0.667 → 0.560 | 0.145 → 0.090 | 0.150 | 0.770 |

forward 8.5 ms/pack (batched), wall 33.9s for 3000 packs
