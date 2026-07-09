# When Does Attention Need to Count Distinct? A Mean-Field Determinantal Gate and Its Operating Regime

*Draft — methods/theory framing. Details/numbers in `THEORY.md`, `MATH.md`, `RESULTS.md`.*

---

## Abstract

Softmax attention scores each item by its own relevance, so a set of `m` near-identical items
captures pooled mass proportional to their **count**. We ask when this matters and what to do about
it. We derive, from a subset-attention view, a **mean-field determinantal gate**: a deterministic,
differentiable, parameter-neutral term inside the softmax numerator that down-weights an item for
being redundant *with the rest of the attended set*. The gate **unifies** existing pooling rules —
softmax, per-token/sparse gates (sparsemax, entmax), and the determinantal-point-process (DPP)
marginal all arise as special cases — and comes with a **separation theorem**: under a redundant
clique of size `m`, any *first-order* (per-item) rule lets the clique's influence grow `Θ(m)`, while
the gate caps it at `Θ(log m)`. We validate the law numerically (`W ≈ 0.41 + 0.82 ln m`, R²=0.999)
and on trained models. A single high learning rate on the gate's scalars makes its strength **learned
and init-independent**, which also removes the optimization variance of the sampled precursor.
Finally, we **characterize the operating regime precisely**, with instrumented evidence: the gate is
an *outlier-keeper* — it helps exactly when the *misleading* content is the redundant majority, and
is neutral-to-harmful when the *useful* signal is itself redundant. Across four real domains (MUSK,
Camelyon16, conflicting-evidence RAG, and Byzantine-robust federated learning) we show this regime is
narrow in practice, and we explain — with per-item weight measurements — why. The result is a clean
account of *when diversity-aware pooling is the right inductive bias, and when it is not.*

## 1. Introduction

Per-item attention (softmax and its sparse variants) is *first-order*: an item's weight depends only
on its own score. A direct consequence is **count-domination** — `m` near-duplicate items collect
`Θ(m)` of the pooled mass regardless of how much *distinct* information they carry. When those
duplicates are misleading (a redundant clique voting for the wrong answer), no first-order rule can
suppress them, because their count enters linearly and cannot be removed by any per-item function of
the score.

We study a second-order alternative and its limits. Contributions:

1. **A unifying mean-field determinantal gate** (§3): a deterministic, differentiable, parameter-
   neutral term inside softmax that nests softmax (β=0), per-token/sparse gates (λ=0), and the DPP
   marginal (the mean-field limit) as special cases, at attention-order cost `O(n²d)` (`O(nd)` global).
2. **A separation theorem** (§4): first-order clique influence is `Θ(m)`; the gate's is `Θ(log m)` —
   an exponential reduction in effective multiplicity — with a *falsifiable, measured* prediction
   `W_D ∼ log m`.
3. **An adaptive-λ training rule** (§5.3): a separate high learning rate on the gate scalars makes
   the repulsion strength learned and init-independent, and collapses the variance of the sampled
   precursor.
4. **A precise operating-regime characterization** (§5.4–5.5): with per-item weight measurements we
   show the gate is an *outlier-keeper* — it helps iff the misleading content is the redundant
   majority, and we map, across four real domains, exactly where that holds and where it does not.

We are deliberately explicit that this is a **methods/theory** contribution with a **narrow but
sharply-characterized** empirical regime, not a new state of the art on a standard benchmark.

## 2. Related work

- **Sparse attention** (sparsemax [Martins'16], entmax [Peters'19]): magnitude thresholds on the
  *score*; they zero low-score items but keep a high-score redundant clique. First-order ⇒ subject to
  the `Θ(m)` bound (§4). *In our experiments they tie or hurt on redundancy.*
- **Determinantal attention** (DppNet [NeurIPS'19]; DPP-A [eLife'23]; Kulesza & Taskar): the
  substrate. Our gate is the **mean-field marginal** of the DPP used *inside* softmax — cheaper
  (`O(n²d)` vs `O(n³)`) and differentiable. *We match exact-DPP on accuracy at lower cost.*
- **Token merging / pruning** (ToMe [ICLR'23], DynamicViT): identical key-similarity signal, but
  merges for efficiency with size-weighted (count-preserving) attention. *We beat ToMe on redundancy.*
- **Diversity/repulsive attention** (Repulsive Attention [EMNLP'20] = between-head; attention-entropy
  maximization): a different axis, or the wrong direction (spreading toward uniform = pure count).
- **Robust aggregation** (Krum, coordinate-median, trimmed-mean, centered-clipping, FoolsGold): §5.4
  situates the gate against these; the gate is an outlier-*keeper*, so it is *not* a robust aggregator
  (§5.5).

## 3. Method

**Subset-attention view.** Write attention as an expectation over attended subsets,
`y = E_{S∼p(S)}[softmax_S(a)·v]`, with `p(S) ∝ exp(β F₂(S))`. A **modular** `F₂` yields a per-token
gate (adaptive sparsity); a **non-modular** (pairwise) `F₂` yields a determinantal law. Collapsing
the sampler to its deterministic **mean-field marginal** gives a per-item gate multiplying the
softmax numerator:

```
a_i = q·k_i/√d
g_i = σ( β (a_i − τ(q) − λ · r_i) ),   r_i = ⟨k_i, Σ_j g_j k_j⟩/√d      (factored, O(n²d))
w_i = g_i e^{a_i} / Σ_j g_j e^{a_j},   y = Σ_i w_i v_i
```

**Unification (special cases).**
- `β = 0` ⇒ `g_i ≡ ½` ⇒ **exact softmax** (parity by construction).
- `λ = 0` ⇒ per-token gate = **adaptive sparsity** (sparsemax/entmax family in spirit).
- The full fixed point of `g` is the **DPP mean-field marginal** (a soft, differentiable DPP).

**Properties** (verified, §5.1): nests softmax to `7.5e-8`; deterministic (output variance `0`, vs
the sampled precursor's `0.955`); **parameter-neutral** (+`dim+2H` scalars, ~0.13% of an MHA layer);
differentiable in `β, τ, λ`. A short λ-warmup avoids corrupting a jointly-trained encoder at init.

## 4. Theory (see `THEORY.md`)

**Setup.** A pooling query over `n` distinct signal items and a clique of `m` identical copies (self-
affinity `s`). Correctness needs the clique's mass ratio `ρ < 1`.

- **Thm 1 (Separation).** *(a)* Any first-order gate `ν_i = φ(a_i)` gives `ρ = Θ(m)` — the count
  enters linearly and is unremovable by any `φ`; the clique dominates once it is a constant factor
  larger than the signal. *(b)* The mean-field gate gives clique mass `W = Σ_{i∈D} g_i = Θ(log m /
  βλs)`, hence `ρ_rep = Θ(log m / n)`; correct up to `m ≈ e^{Θ(n)}`.
- **Prop 2 (Well-posedness).** The mean-field map is a contraction for `βλ‖K‖_∞ < 4` ⇒ a unique,
  geometrically-convergent fixed-point gate.
- **Prop 3 (Complexity).** `O(n²d)` per-query (attention order; ~2× constant), `O(nd)` for a global
  gate. No `O(n³)` Gram.
- **Falsifiable prediction.** `W_D(m) ∼ m` (softmax) vs `∼ log m` (rep), directly measurable.

## 5. Experiments

### 5.1 Property verification & parity
`gated_attention.py`/`verify_gated_parity.py`: nests softmax (7.5e-8), deterministic (0.0), param
delta = `dim+2H` exactly. **CIFAR-10** TinyViT, 5 seeds, dropout-matched: MHA 83.95 vs gated+rep
84.00 (all Welch |t|<1.5) — **no harm on benign data.**

### 5.2 Controlled adversarial redundancy (where it wins) + the scaling law
Two self-contained testbeds where a misleading clique dominates by count.
- **Synthetic redundancy** (3 seeds): rep_val **99.26**, rep_key 98.80 vs entmax 96.21, modular
  96.87, dense 95.27, sparsemax 85.81. Ordering: **repulsion ≫ gate/entmax ≫ dense ≫ sparsemax.**
- **Real-image MNIST-bags MIL** (shared backbone, 5 seeds): rep **71.2** vs Gated-ABMIL 24.2, AEM
  10.9 (chance), sparsemax/entmax 23–28, cheap dedup/vote ~34; nearest cousins: exact-DPP 62 (tie on
  acc, win on `O(n²d)` cost), ToMe 17. Needle (no redundancy): ties.
- **Scaling law [key figure]:** trained ABMIL clique gate grows `∝ m` (R²=1.0); rep → 0. Measured
  `W ≈ 0.41 + 0.82 ln m` (R²=0.999) — the predicted `log m` law; 559× suppression at m=4096.
- **Generality:** the redundancy win holds across width (dim 32–256, Δ +3.4–5.3) in a self-attention
  Transformer (second architecture).

### 5.3 Adaptive λ (learnable strength; resolves variance)
The sampled precursor and fixed-λ variants had high seed variance (±12–13). The lever is the λ
**learning rate**: a separate high LR on `(λ,τ,β)` makes λ **learned and init-independent** (converges
to ~0.30 from either init on adversarial data) — adversarial accuracy **85.7 ±0.5** (up from 71), and
on a redundancy-is-signal task variance collapses (0.917 **±0.004**). Fixed-λ and more mean-field
iterations both fail; the right lever was optimization, not the estimator.

### 5.4 Operating regime across four real domains (honest scorecard)
Using real data + standard threat models where applicable:

| domain | setup | outcome for the gate |
|---|---|---|
| MUSK1/2 (MIL) | 10-fold CV | **tie** (no harm) |
| Camelyon16 (MIL) | Phikon feats, std split | **loses** at default (0.858 vs softmax 0.946 AUC); adaptive-λ → 0.917 (near-tie). It is a *redundancy-is-signal* needle task. |
| RAMDocs (RAG) | native + injected poisoning | native = tie (not count-dominated); under poisoning, **cheap dedup / learned relevance win**; gate hurts (over-suppresses redundant *correct* docs) |
| Byzantine-FL (CIFAR) | FedSGD, adaptive attack | **loses**; an attack crafted to evade it drops it below Krum |

**Takeaway:** no standard-benchmark win. The regime where the gate helps (misleading content is the
redundant majority *and* the useful signal is not itself redundant) is **narrow in practice**.

### 5.5 Why — the gate is an outlier-*keeper* (instrumented)
`byzantine_diag.py` measures the weight the gate assigns to honest vs adversarial items (uniform
0.0625). It suppresses adversaries only when they dominate the aggregate (large-magnitude attack:
w_adv ≈ 0.0002); against a *norm-bounded, diversified minority* it **amplifies** them (w_adv 0.08–0.12
> w_honest 0.04–0.06), because with honest items in the majority the aggregate points at the honest
consensus and the gate flags *that* as redundant. This is the mechanism, not a tuning artifact:
inverting the sign for robustness merely reproduces mean-shift/median. It also explains 5.4 — in real
tasks the useful signal is usually the (redundant) majority, exactly what the gate down-weights.

## 6. Scope & limitations (stated as a result)

> **The gate is a diversity/decorrelation operator: it down-weights the dense/aligned majority and
> keeps distinct/isolated items. It is therefore the right inductive bias iff the *misleading*
> content is the redundant majority, and the wrong one when the *useful* signal is the majority.**

- Requires a *trainable* attention (not a frozen-LLM inference patch); natural home = MIL / set models.
- Not a robust aggregator (outlier-keeper, not remover); §5.5.
- One small model scale; theory is for the clean clique model (extends qualitatively).

## 7. Conclusion

Count-domination is a real, provable failure mode of per-item attention. A mean-field determinantal
gate fixes it — provably (`Θ(m)→Θ(log m)`), unifying softmax/sparse/DPP, at attention-order cost,
learnable via a single LR, and without harming benign data. Its benefit is real but **precisely
bounded**: it helps when distinctness (not count) should decide, and we give the theory, the
measured scaling law, and instrumented evidence for exactly when that is — and is not — the case.

---

### Artifact / repro
`gated_attention.py`, `f2_sweep.py`, `mil_abmil.py`, `mil_mnist.py`, `scaling_figure.py`,
`theory_check.py`, `camelyon_mil.py`, `musk_mil.py`, `ramdocs_*.py`, `byzantine_check.py`,
`fl_byzantine.py`, `byzantine_diag.py`; full logs in `RESULTS.md`, proofs in `THEORY.md`.

### Status
Draft prose complete; needs: polished figures (mechanism schematic, `W∼log m` curve, scaling
accuracy), formal proof write-ups from `THEORY.md`, and a final pass on baselines/citations.
