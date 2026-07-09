#!/usr/bin/env python3
"""Precompute TinyLlama embeddings for surface-diverse PARAPHRASE copies of each misinfo doc.
Each variant drops a random fraction of tokens (surface differs -> evades a bag-of-words dedup)
while keeping the same claim (semantically redundant). The poisoning experiment injects these as
duplicate poison passages. Saves ramdocs_para_emb.npz: variants[gid] (V,2048) keyed by a flat
misinfo id, plus the (qi,di) map. Also reports paraphrase copy-to-copy cosine vs copy-to-legit."""
from __future__ import annotations

import json
import re

import numpy as np
import torch

from ramdocs_embed import MODEL, PATH, embed_texts   # reuse loader/pooler

import os
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoModel, AutoTokenizer

OUT = "/data/users/dereklong/scratch/general-attention-mechanism/ramdocs_para_emb.npz"
_tok = re.compile(r"[A-Za-z0-9]+")
N_VAR = 8
DROP = 0.5


def paraphrase(text, rng):
    toks = _tok.findall(text)
    if len(toks) < 6:
        return text
    keep = [t for t in toks if rng.random() > DROP]
    return " ".join(keep) if len(keep) >= 3 else " ".join(toks[:3])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()

    rows = [json.loads(l) for l in open(PATH)]
    rng = __import__("random").Random(0)
    texts, ids = [], []          # flatten all variants; id = f"{qi}_{di}"
    for qi, r in enumerate(rows):
        for di, d in enumerate(r["documents"]):
            if d["type"] != "misinfo":
                continue
            for _ in range(N_VAR):
                texts.append(paraphrase(d["text"], rng))
                ids.append(f"{qi}_{di}")
    print(f"{len(set(ids))} misinfo docs x {N_VAR} variants = {len(texts)} texts", flush=True)
    emb = embed_texts(texts, tok, model, device)         # (n,2048)
    # group by id
    order = {}
    for i, gid in enumerate(ids):
        order.setdefault(gid, []).append(i)
    keys = list(order)
    arrs = {gid: emb[order[gid]] for gid in keys}
    np.savez(OUT, keys=np.array(keys), **{f"v_{gid}": arrs[gid] for gid in keys})
    print(f"saved {OUT}: {len(keys)} misinfo groups", flush=True)

    # diagnostic: copy-copy vs copy-legit cosine (centered+rmtop10, matching the experiment)
    d0 = np.load("ramdocs_emb.npz"); demb = d0["demb"]
    mu = demb.mean(0)
    U, S, Vt = np.linalg.svd(demb - mu, full_matrices=False)
    def proc(E):
        m = E - mu; m = m - (m @ Vt[:10].T) @ Vt[:10]
        return m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-9)
    legit = proc(demb)
    cc, cl = [], []
    for gid in keys[:80]:
        V = proc(arrs[gid])
        for a in range(len(V)):
            for b in range(a + 1, len(V)):
                cc.append(float(V[a] @ V[b]))
        cl += [float(V[0] @ legit[j]) for j in range(0, len(legit), 37)]
    print(f"paraphrase copy-copy cos: {np.mean(cc):.3f} | copy-legit cos: {np.mean(cl):.3f} | "
          f"gap {np.mean(cc)-np.mean(cl):+.3f}")


if __name__ == "__main__":
    main()
