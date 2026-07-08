#!/usr/bin/env python3
"""Redundancy fingerprint: does REAL code have the within-context redundancy structure
our mechanism exploits? Compares real Python code windows vs the synthetic redundancy /
needle tasks vs CIFAR image patches, on two intuitive metrics per context window:

  pair_redun  = fraction of token PAIRS that are near-duplicates (>0.9 cosine, or identical
                token for code)  -> "how redundant is a typical window"
  max_clique  = size of the largest near-duplicate cluster / L
                -> "how big is the dominant redundant clump"

Note metric caveat: code uses exact-token identity; continuous domains use cosine>0.9.
Both answer "how much of a window is near-duplicate." CPU only.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import numpy as np

L = 128
PY_ROOT = "/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/MuLoCo/lm-evaluation-harness"
CIFAR = "/data/users/dereklong/scratch/general-attention-mechanism/data/cifar-10-batches-py"
TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\S")


def code_windows(root, max_files=400, max_windows=800):
    files = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.endswith(".py"):
                files.append(os.path.join(dp, f))
        if len(files) >= max_files:
            break
    wins = []
    for p in files:
        try:
            toks = TOK.findall(open(p, encoding="utf-8", errors="ignore").read())
        except Exception:
            continue
        for i in range(0, len(toks) - L, L):
            wins.append(toks[i : i + L])
            if len(wins) >= max_windows:
                return wins
    return wins


def token_redun(win):
    c = Counter(win)
    same_pairs = sum(n * (n - 1) for n in c.values())
    pair_redun = same_pairs / (L * (L - 1))
    max_clique = max(c.values()) / L
    return pair_redun, max_clique


def cont_redun(X, thr=0.9):  # X: (L, d)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 0.0)
    high = S > thr
    pair_redun = high.sum() / (L * (L - 1))
    max_clique = high.sum(axis=1).max() / L  # largest near-dup neighborhood
    return pair_redun, max_clique


def synth_redundancy_window(rng, C=10, d=32, max_sig=5, max_dec=30, n_bg=30, ns=0.3, nb=1.0):
    n_sig = rng.integers(2, max_sig + 1); m_dec = rng.integers(2, max_dec + 1)
    protos = rng.standard_normal((C, d))
    y, yp = 0, 1
    sig = protos[y] + ns * rng.standard_normal((n_sig, d))
    base = protos[yp] + ns * rng.standard_normal(d)
    dec = np.tile(base, (m_dec, 1))                          # identical clique
    bg = nb * rng.standard_normal((n_bg, d))
    X = np.concatenate([sig, dec, bg], 0)
    if X.shape[0] < L:
        X = np.concatenate([X, nb * rng.standard_normal((L - X.shape[0], d))], 0)
    return X[:L]


def synth_needle_window(rng, d=32, noise=1.2):
    dprotos = rng.standard_normal((4, d))
    idx = rng.integers(0, 4, L)
    X = dprotos[idx] + noise * rng.standard_normal((L, d))    # 4 redundant distractor protos
    X[rng.integers(0, L)] = rng.standard_normal(d) + noise * rng.standard_normal(d)  # needle
    return X


def cifar_windows(root, n=200):
    import pickle
    with open(os.path.join(root, "data_batch_1"), "rb") as f:
        arr = pickle.load(f, encoding="latin1")["data"]  # (10000, 3072)
    imgs = arr.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    out = []
    for k in range(n):
        im = imgs[k]  # (3,32,32)
        # 4x4 patches -> 64 patches x 48
        p = im.reshape(3, 8, 4, 8, 4).transpose(1, 3, 0, 2, 4).reshape(64, 48)
        out.append(p)
    return out  # each (64,48); L for cifar = 64


def summarize(name, pairs):
    pr = np.array([p[0] for p in pairs]); mc = np.array([p[1] for p in pairs])
    print(f"{name:24}{pr.mean():>12.3f}{mc.mean():>14.3f}   (n={len(pairs)})")


def main():
    rng = np.random.default_rng(0)
    print(f"{'domain':24}{'pair_redun':>12}{'max_clique/L':>14}")
    print("-" * 54)
    # real code
    cw = code_windows(PY_ROOT)
    summarize("real Python code", [token_redun(w) for w in cw])
    # synthetic redundancy
    summarize("synthetic redundancy", [cont_redun(synth_redundancy_window(rng)) for _ in range(400)])
    # synthetic needle
    summarize("synthetic needle", [cont_redun(synth_needle_window(rng)) for _ in range(400)])
    # cifar patches
    try:
        summarize("CIFAR image patches", [cont_redun(p) for p in cifar_windows(CIFAR)])
    except Exception as e:
        print("CIFAR:", e)
    print("\n(higher = more within-context redundancy; our method needs this to be present)")


if __name__ == "__main__":
    main()
