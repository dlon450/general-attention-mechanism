#!/usr/bin/env python3
"""Non-gameability gate: at the headline slice (alpha=1), NO cheap content/count statistic may exceed
a pre-registered threshold (55%) over chance(50%). If any does, the task leaks and results are void.
Run: python tests/test_nongameability.py  (exits nonzero on failure)."""
import sys

import numpy as np

sys.path.insert(0, "/data/users/dereklong/scratch/general-attention-mechanism")
from task_consensus import (Cfg, prototypes, gen_batch, base_item_majority,
                            base_surface_count, oracle_robust)

THRESH = 55.0


def main():
    cfg = Cfg(); rng = np.random.default_rng(0); mu = prototypes(cfg, rng)
    B = gen_batch(4000, cfg, np.random.default_rng(42), alpha=1.0, gamma=0.8, mu=mu)
    acc = lambda p: 100.0 * (p == B["y"]).mean()
    im, sc, orc = acc(base_item_majority(B)), acc(base_surface_count(B)), acc(oracle_robust(B))
    print(f"alpha=1, gamma=0.8, chance=50%: item-majority={im:.1f} surface-count={sc:.1f} "
          f"| oracle(provenance)={orc:.1f}")
    ok = True
    for name, v in [("item_majority", im), ("surface_count", sc)]:
        if v > THRESH:
            print(f"  FAIL: cheap baseline {name}={v:.1f} > {THRESH} (task is gameable)"); ok = False
    if orc < 90.0:
        print(f"  FAIL: provenance oracle={orc:.1f} < 90 (provenance not informative enough)"); ok = False
    if ok:
        print(f"  PASS: cheap baselines <= {THRESH} (non-gameable) and provenance oracle >= 90.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
