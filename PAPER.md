# Counting Distinct, Not Total: Provenance-Aware Pooling for Consensus under Adversarial Duplication

*Draft — the honest, CI-backed framing. Numbers in `RESULTS.md` (§14–18), theory in `THEORY.md`,
pre-registration in `PREREG.md`, mechanism math in `MATH.md`.*

---

## Abstract

Softmax attention scores each item by its own relevance, so `m` near-identical items capture pooled
mass proportional to their **count**. When multiplicity is adversarial — a single source duplicated to
outvote independent sources — this is a failure mode, but fixing it is subtle: the *same* multiset of
`(query, key, value)` could be `m` copies from one source **or** `m` independent corroborating
sources, and the correct pooling differs. We prove and demonstrate a **no-free-lunch**: on a
distribution where content and count are swap-symmetric, **no pooling rule that sees only the item
multiset — softmax, sparse attention, or a Set Transformer — beats chance.** The missing ingredient is
**provenance** (a possibly-noisy same-source signal). Given it, we introduce a **source-aware gate** —
a deterministic mean-field pooling term (the descendant of a determinantal/repulsion attention) that
down-weights items redundant *within their same-origin group*, combined with a zero-initialized
residual so it starts as a strong low-data prior and relaxes to a flexible high-data solution. On a
pre-registered benchmark (10 seeds, paired-bootstrap CIs, frozen test), content-only attention is at
chance while our gate is (i) **far more sample-efficient** than feeding provenance to plain attention
(paired-CI wins of +6 to +30 points at ≤800 labels), (ii) **at parity at scale**, and (iii) **the most
robust to an adaptive adversary** (worst-case-over-attack 89.0 vs ≤85.1 for baselines), at **matched
parameters (+0.2%)** and ~1.7× forward cost. We are explicit about scope: this is an inductive-bias +
robustness result for consuming provenance, not a universal "better attention"; on standard benchmarks
without provenance (MUSK, Camelyon16, RAG poisoning, Byzantine-FL) a fixed diversity bias ties or loses
(§14), exactly as the no-free-lunch predicts.

## 1. Introduction

Per-item attention is *first-order*: an item's weight depends only on its own score, so a size-`m`
duplicate clique collects `Θ(m)` of the pooled mass. When that clique is a coordinated adversary
(Sybil/duplication attack), count-based pooling is fooled. But the deep obstacle is an **identifiability
gap**: multiplicity and corroboration produce the same item multiset, so *no function of the multiset
alone* can separate "one source shouting" from "many sources agreeing." Contributions:

1. **No-free-lunch (§4, §5.1):** a swap-symmetry construction on which any multiset-only pooling rule
   is exactly chance; confirmed empirically — softmax **and** a Set Transformer sit at chance.
2. **Provenance as the resolving signal, and a source-aware gate to use it (§3):** a deterministic
   mean-field pooling gate keyed on same-origin density, with a zero-init residual (rigid prior →
   flexible). It is the honest, corrected descendant of a determinantal/repulsion attention.
3. **A pre-registered, CI-backed win (§5):** far more sample-efficient than feeding provenance to
   plain attention, at parity at scale, and most robust to an adaptive adversary; matched params.
4. **Honest scope (§6):** without provenance no fixed rule can win; on standard no-provenance
   benchmarks a diversity bias ties/loses. We report those negatives.

## 2. Related work
- **Sparse attention** (sparsemax, entmax): per-item magnitude thresholds — first-order, subject to
  the no-free-lunch; at chance here.
- **Set models** (Set Transformer / ISAB+PMA, DeepSets): expressive multiset functions — but the
  no-free-lunch is information-theoretic, so a Set Transformer is *also* at chance without provenance.
- **Determinantal / repulsive attention** (DppNet, DPP-A, ToMe): diversity by key similarity. We use a
  deterministic mean-field marginal; §4/E0 corrects the true clone-invariant primitive (Θ(1)).
- **Robust aggregation / Sybil defense** (Krum, coord-median, centered-clipping, FoolsGold): related
  in spirit; our gate is provenance-conditioned pooling, not a coordinate-space robust mean (§14).
- **Provenance / attribution in RAG and trust**: motivates the same-source signal.

## 3. Method

**Deterministic mean-field pooling (lineage, not sampling).** The original idea drew subsets
`S∼p(S)∝exp(βF₂(S))` and averaged (Monte-Carlo) — high-variance and non-differentiable. We collapse
it to its **deterministic mean-field marginal**: each item gets a closed-form inclusion gate `gᵢ`
multiplying the softmax numerator, `wᵢ ∝ gᵢ e^{aᵢ}` (no sampling; variance 0; differentiable).

**Source-aware gate (ours).** Relevance `aᵢ = q·kᵢ/√d`. From a (noisy) same-source graph `P`, compute
each item's **within-content-neighbourhood same-origin density** `densᵢ = Σⱼ Cᵢⱼ Pᵢⱼ` (`C` = content-
similarity weights). A same-source clique has high density; independent corroborators have low density.
The gate down-weights high density:
`gᵢ = σ( β(τ − λ·densᵢ) + MLP([densᵢ, degreeᵢ, aᵢ]) )`, with the **MLP zero-initialised** so training
*starts* as the rigid prior `σ(β(τ−λ·dens))` (maximal sample-efficiency) and *relaxes* into a flexible
function as data grows (recovers the ceiling). Parity: `λ=0, MLP=0 ⇒ softmax`. Cost: `O(n²d)` (attention
order); +O(scalars) params.

## 4. Theory (`THEORY.md`)
- **No-free-lunch (Thm):** on the swap-symmetric slice the `(V, source-id)` marginal is invariant under
  honest↔adversary role swap ⇒ any function of the multiset has Bayes error = chance; only provenance
  breaks the symmetry.
- **Corrected clone-invariance (E0, §15):** the shipped one-step gate drives clique mass → 0
  (over-suppression); the naive fixed point is Θ(log m) but ill-posed past its contraction limit; the
  **true DPP marginal is Θ(1)** — one effective vote per duplicate group. We drop the earlier
  "Θ(log m)" and "nests DPP" claims and use the corrected primitive.
- **Complexity:** `O(n²d)` per query (attention order), `O(nd)` for a global gate.

## 5. Experiments (pre-registered; `PREREG.md`, 10 seeds, paired-bootstrap CIs, frozen test)

Benchmark: consensus-under-adversarial-duplication (`task_consensus.py`) — two content clusters matched
in #items / #surface-ids / content spread, differing only in true-origin structure (honest = many
independent origins; Sybil = one origin over many surface ids), revealed via a noisy same-origin graph.
Non-gameability gate passes (cheap content/count baselines = chance; oracle 98%).

- **5.1 No-free-lunch (H1):** at α=1, `softmax` 48.6→50.2 and `set_transformer` 35→49 across
  n∈[200,6000] — content-only is at chance. Regular attention cannot win. [TABLE §18]
- **5.2 Sample-efficiency (H2, headline):** paired Δ(m2_prov_r − prov_concat) = **+29.6** [28.5,30.7] @
  n=200, **+24.7** @400, **+6.3** [3.6,8.7] @800 (CI-significant wins); parity at the ceiling (n≥3200);
  one negligible −0.6 dip @1600. Large win when labels are scarce, no cost at scale. [FIG: learning curve]
- **5.3 Robustness (H3, clean win):** worst-case-over-α (adaptive adversary) — m2_prov_r **89.0** vs
  prov_concat 85.1, relation_bias 51.3 (collapses at α=1), softmax 49.9. Most robust. [TABLE §18]
- **5.4 Ablation (bias–variance):** rigid gate (max sample-efficiency, capped ceiling) vs MLP gate
  (recovers ceiling, needs data) vs rigid+residual (best of both). [§17.1]
- **5.5 Latency / params:** +0.2% params, ~1.7× forward, fwd+bwd comparable. [TABLE §18]
- **5.6 OOD:** train small-clusters → test large-clusters ties (shift makes it easier); reported
  honestly as non-differentiating.

## 6. Scope & limitations (stated as results)
- **Provenance is required.** Without it, the no-free-lunch bites: on standard benchmarks with no
  same-source signal (MUSK tie, Camelyon16 lose→near-tie, RAMDocs poisoning loses to dedup,
  Byzantine-FL loses to Krum under adaptive attack; §11–14) a fixed diversity gate ties or loses — it
  is an *outlier-keeper*, useful only when the misleading content is the redundant majority (§14.2).
- **Not a high-data expressivity win.** At scale a flexible model matches us; the contribution is
  sample-efficiency + robustness (an inductive bias), not universal superiority.
- **Scale/scope:** one MVP model size and one task family; requires trainable attention.

## 7. Conclusion
Multiplicity vs. corroboration is an irreducible ambiguity for multiset pooling — provably unlearnable
from content alone. Given a (noisy) provenance signal, a deterministic source-aware pooling gate uses
it more sample-efficiently and more robustly than feeding it to generic attention, at matched cost —
while, without provenance, no fixed rule can help. The honest boundary is the result.

---

### Status / to finalize
- Figures: learning-curve with CIs, worst-case-over-α bar, mechanism schematic.
- Formalize the no-free-lunch and E0 corrections in `THEORY.md`.
- Scale check (larger model / a second task family) to test whether the sample-efficiency + robustness
  edge persists; citations/baselines pass.
### Artifacts
`task_consensus.py`, `consensus_models.py`, `bench_full.py`, `aggregate.py`, `clique_scaling.py`,
`tests/test_nongameability.py`, `PREREG.md`; full logs `RESULTS.md`, proofs `THEORY.md`.
