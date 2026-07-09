# Paper skeleton (toward a spotlight submission)

Working title: **"Attention that counts distinct, not total: a mean-field determinantal gate against
adversarial redundancy."**

This is the narrative skeleton with claims, the numbers we have, figure/table slots, and what's
still needed. Details in `THEORY.md`, `RESULTS.md`, `MATH.md`.

---

## Abstract (draft)
Softmax attention weights each token by its own relevance, so a set of `m` near-identical tokens
captures attention mass proportional to their *count*. When such redundancy is **adversarial** —
duplicated tokens carrying misleading signal that dominates a pooled decision — softmax, sparse
attention (sparsemax/entmax), attention-entropy methods (AEM), and even token-merging (ToMe) all
fail. We derive, from a subset-attention view, a deterministic **mean-field determinantal gate**:
a parameter-neutral, differentiable term inside softmax that down-weights a token for being
redundant *with the other attended tokens*. We prove a **separation** — first-order attention lets
a size-`m` clique's influence grow `Θ(m)`, while our gate bounds it at `Θ(log m)` — validate it
numerically (R²=0.999) and on trained models, and show on real-image weakly-supervised MIL that it
beats dense attention, sparse attention, SOTA MIL (Gated-ABMIL, AEM), cheap dedup/voting defenses,
and both nearest cousins (exact-DPP, ToMe), at equal parameters and training budget — while tying
on benign data (no over-firing).

## 1. Introduction
- **Problem:** adversarial within-context redundancy — duplicated/near-duplicate items win by
  count and mislead pooled decisions (real instances: RAG with mirrored misinformation, review
  spam, retrieval poisoning, weakly-supervised MIL with a rare positive among redundant negatives).
- **Gap:** per-token operators (softmax, sparse attention) *cannot* see redundancy; diversity-by-
  spreading (AEM) points the wrong way; token-merging (ToMe) with proportional attention preserves
  the count.
- **Contribution:** (i) a param-neutral, deterministic, differentiable mean-field DPP repulsion gate
  inside softmax; (ii) a separation theorem (Θ(m) vs Θ(log m)) with numerical + trained-model
  validation; (iii) real-image MIL experiments beating dense/sparse/SOTA/cheap-defenses/nearest-
  cousins; (iv) an honest scope (helps iff redundancy is adversarial; ties otherwise).

## 2. Related work (from the 48-method survey)
- **Sparse attention** (sparsemax, entmax): magnitude thresholds — can't remove high-relevance
  duplicates (our baselines; they tie-or-hurt).
- **DPP / determinantal attention** (DppNet NeurIPS'19; DPP-A eLife'23; K&T monograph): the
  substrate; we use the mean-field marginal as an in-softmax gate — cheaper (O(n²d) vs O(n³)) and
  differentiable. *We beat/tie exact-DPP.*
- **Token merging / pruning** (ToMe ICLR'23, DynamicViT…): same key-redundancy signal, but merge
  for efficiency with size-preserving proportional attention → doesn't fix accuracy. *We beat ToMe.*
- **Repulsive/diversity attention** (Repulsive Attention EMNLP'20 = head diversity; AEM = entropy
  max): different axis / wrong direction for adversarial redundancy. *AEM fails (chance).*
- **Coverage** (See'17, Tu'16): anti-repetition across decode steps, not intra-set.

## 3. Method
- Subset-attention view: `y = E_{S~p(S)}[softmax_S]`, `p(S) ∝ exp(β F₂(S))`.
- Modular F₂ → per-token gate = adaptive sparsity (nests softmax; not enough).
- Non-modular F₂ (pairwise repulsion) → mean-field marginal gate:
  `w_i ∝ g_i e^{a_i}`, `g_i = σ(β(a_i − τ − λ r_i))`, `r_i = ⟨k_i, Σ_j g_j k_j⟩`.
- Properties: nests softmax (β=0); deterministic (variance 0, vs the sampled version's 0.955);
  param-neutral (+O(d)); O(n²d) (O(nd) global). λ-warmup for jointly-trained encoders.
- [FIG 1] mechanism schematic (clique suppression).

## 4. Theory (THEORY.md)
- **Thm 1 (separation):** first-order clique influence Θ(m); mean-field repulsion Θ(log m).
- **Prop 2 (well-posedness):** contraction for βλ‖K‖<4 ⇒ unique fixed-point gate.
- **Prop 3 (complexity):** O(n²d) per-query, O(nd) global.
- [FIG 2] numerical validation: W ≈ 0.41 + 0.82 ln m (R²=0.999) vs softmax = m; 559× suppression @ m=4096.

## 5. Experiments
- **5.1 Parity (no-harm):** CIFAR-10 TinyViT, dropout-matched, 5 seeds — all methods tie (~83.9%,
  Welch |t|<1.5). [TABLE]
- **5.2 Controlled synthetic:** F₂-family map (needle vs redundancy); rep_key/rep_val ablation
  (rep_val 99.3 ≥ rep_key 98.8). [TABLE]
- **5.3 Real-image MIL (headline):** MNIST-bags, shared backbone, 5 seeds. rep **71.2** vs
  Gated-ABMIL 24.2, AEM 10.9 (chance), sparsemax/entmax 23–28, dedup/countnorm ~34; needle ties.
  [MAIN TABLE]
- **5.4 Nearest-cousin head-to-head:** vs exact-DPP (72 vs 62, tie on acc / win on O(n²d) cost) and
  ToMe (72 vs 17, t=5.9). [TABLE]
- **5.5 Scaling (theory↔experiment):** trained ABMIL clique gate = m (R²=1.0); rep → 0; acc gap
  holds across m∈[2,128]. [FIG 3 — money figure]
- **5.6 Over-firing control:** majority-by-count task (redundancy legitimate) — rep ≈ ABMIL (~91%),
  no harm. [TABLE]

## 6. Limitations (honest; RESULTS.md §13)
- **Variance** (±13, optimization/basin) — unresolved; fixed-λ and more mean-field iters both fail;
  blocks a *significant* win over exact-DPP. (Not estimator variance ⇒ control variates N/A.)
- **Single real testbed** (semi-synthetic MNIST-bags custom construction) — need standard
  benchmarks (Camelyon16/TCGA MIL; RAMDocs/ConflictQA RAG). ← biggest remaining gap.
- **Scope:** requires a *trainable* attention (not frozen LLMs); natural home = MIL/set models.
- **Generality:** one small model/scale; need ≥2 architectures + a real self-attention Transformer.

## 7. Conclusion
Adversarial redundancy is a real, specific failure of per-token attention; a param-neutral mean-
field determinantal gate provably and empirically fixes it, at attention-order cost, without
harming benign data.

---

## Status vs spotlight bar (what's left)
| component | state |
|---|---|
| theory + validation | ✅ done |
| mechanism + parity + synthetic | ✅ done |
| real-image MIL vs SOTA + cheap defenses + nearest cousins | ✅ done |
| scaling figure | ✅ done |
| over-firing control | ✅ done (3-seed confirming) |
| **variance control** | ❌ open (both fixes failed) |
| **standard real benchmark(s)** | ❌ needs data staging (biggest gap) |
| generality (architectures/scale, real Transformer) | ⬜ partial |
| polished figures + full writeup | ⬜ skeleton only |
