#!/usr/bin/env python3
"""Precompute REAL semantic embeddings for RAMDocs docs/questions with the locally-cached
TinyLlama-1.1B (mean-pooled last hidden state, L2-normalized). Saves to ramdocs_emb.npz and runs
a sanity check: are docs that assert the SAME answer more cosine-similar than docs asserting
DIFFERENT answers? (If not, the encoder is too weak to support a 'semantic redundancy' claim.)"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoModel, AutoTokenizer

PATH = "/home/dereklong/mil_data/ramdocs/RAMDocs_test.jsonl"
OUT = "/data/users/dereklong/scratch/general-attention-mechanism/ramdocs_emb.npz"
# locally-staged TinyLlama-1.1B @105B (Llama decoder, hidden 2048) — mean-pooled sentence embedding
MODEL = "/data/users/dereklong/fbsource/fbcode/ads_ceh/adaptive_diloco/assets/models/tinyllama-1.1b-step-50k-105b"


@torch.no_grad()
def embed_texts(texts, tok, model, device, bs=16, maxlen=256):
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(device)
        h = model(**enc).last_hidden_state                     # (B,T,H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)      # masked mean-pool
        v = torch.nn.functional.normalize(v, dim=-1)
        out.append(v.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()

    rows = [json.loads(l) for l in open(PATH)]
    # flat list of all doc texts + questions, with an index map back to (q, doc)
    doc_texts, q_texts, meta = [], [], []
    for qi, r in enumerate(rows):
        q_texts.append(r["question"])
        for di, d in enumerate(r["documents"]):
            doc_texts.append(d["text"])
            meta.append((qi, di))
    print(f"{len(rows)} questions, {len(doc_texts)} docs -> embedding with {MODEL}", flush=True)
    demb = embed_texts(doc_texts, tok, model, device)
    qemb = embed_texts(q_texts, tok, model, device)
    np.savez(OUT, demb=demb, qemb=qemb, meta=np.array(meta))
    print(f"saved {OUT}: docs {demb.shape}, questions {qemb.shape}", flush=True)

    # sanity: within-Q same-answer vs different-answer cosine
    same, diff = [], []
    idx = 0
    per_q = {}
    for (qi, di) in meta:
        per_q.setdefault(qi, []).append(idx); idx += 1
    for qi, r in enumerate(rows):
        ids = per_q[qi]
        ans = [r["documents"][di]["answer"].lower().strip() for (_, di) in [meta[j] for j in ids]]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                c = float(demb[ids[a]] @ demb[ids[b]])
                (same if ans[a] == ans[b] and ans[a] != "unknown" else diff).append(c)
    print(f"SANITY same-answer cos: {np.mean(same):.3f} (n={len(same)}) | "
          f"diff-answer cos: {np.mean(diff):.3f} (n={len(diff)}) | "
          f"gap {np.mean(same)-np.mean(diff):+.3f}")


if __name__ == "__main__":
    main()
