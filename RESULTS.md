# Results: from a broken sampled "general attention" to a working anti-redundancy attention

This document records the full investigation: diagnosing why the sampled subset-attention
underperformed, fixing it with a deterministic mean-field gate, and honestly characterizing
where the resulting mechanism does and does not beat standard attention — including
comparisons against published sparse-attention baselines.

All experiments are parameter-matched (differences are O(d) gate params, ~0.1–0.2%) and run
on a single MI350X box; synthetic tasks are self-contained.

---

## TL;DR

1. The original **Gibbs-sampled** subset attention underperformed dense attention because it
   was (a) ~95% forward-pass noise, (b) giving its learnable F2 parameters *zero* gradient,
   (c) sampling a biased, non-stationary subset, and (d) computing an objective that isn't
   softmax. All four confirmed empirically + by a 25-finding adversarial code audit (0 refuted).
2. The fix: collapse the sampler to its **deterministic mean-field marginal**, a per-token gate
   `g_i = σ(β(a_i − τ(q) − λ·r_i))` multiplying the softmax numerator. This nests softmax
   exactly (β=0), is deterministic (variance 0), differentiable, and parameter-neutral.
3. **On CIFAR-10 it ties** dense attention (parity, no win) — as do all F2 variants.
4. It **beats** dense attention **only where redundancy is *adversarial*** (redundant tokens
   carry misleading signal that dominates by count). There it beats dense **and** the published
   sparse-attention baselines (sparsemax, entmax-1.5) — by +4.0 acc / 4× lower eval loss.
5. The winning ingredient is the **pairwise repulsion term** (the "general"/non-modular part),
   not the per-token gate. Mere redundancy is *not* enough (CIFAR is the most redundant testbed
   yet ties) — it must be adversarial.
6. **Not useful for standard LLM pretraining** (benign redundancy; risks harming induction heads).
   Its home is inference-time adversarial-redundancy / coverage settings (RAG with conflicting
   duplicates, robustness, multi-doc coverage).
7. **Novelty is a recombination, not a new concept**: the closest prior art is DppNet, ToMe
   (identical key-similarity signal), DPP-A, and the entmax/sparsemax family.

---

## 1. Diagnosis of the original sampled version

Measured on the original `general_attention.py` (restricted_softmax F1, sampled F2):

| failure | evidence |
|---|---|
| Forward pass is noise-dominated | MC output relative std ≈ **0.955** at runs=6 (~540 runs needed for 0.1) |
| Learnable F2 params get **zero** gradient | `grad(tau)=None`, `grad(f2_neural)=None` for `restricted_softmax` (it `del count`s and pools the detached bool mask) |
| Sampler is biased | empty-init + few Gibbs steps → support ~10/65, not stationary |
| Objective ≠ softmax | `E_S[softmax_S] ≠ softmax`; only `full_set` recovers exact attention |

A 6-dimension adversarial audit produced **25 findings, 0 refuted** (also: stochastic eval,
`modular_dot` applying an undocumented τ, non-telescoping straight-through gradient, MHA
baseline init clobbered, `dot_repulsion` sign-flip in the sweep, verify-script only covering
`full_set`).

## 2. The fix — deterministic mean-field gated attention

`gated_attention.py :: GatedSoftmaxAttention`, derived in `MATH.md`:

```
a_i = q·k_i/√d
g_i = σ( β (a_i − τ(q) − λ · r_i) ),   r_i = Σ_j g_j ⟨k_i,k_j⟩/√d   (mean-field DPP marginal)
w_i = g_i·e^{a_i} / Σ_j g_j·e^{a_j}
y   = Σ_i w_i v_i
```

- **β=0 ⇒ g_i=½ ⇒ exact softmax** (starts at parity, departs only if it helps).
- **Deterministic** (variance 0, was 0.955); **differentiable** to β/τ/λ (were None).
- **Parameter-matched**: +`dim+2H` params (0.13% of MHA at d=192,H=3).
- `λ·r_i` (repulsion) is the distinctive **non-modular** term; λ=0 recovers a plain per-token gate.
- Repulsion computed in the **factored O(n²d)** form `r_i = ⟨k_i, Σ_j g_j k_j⟩` (not the O(n³) Gram).

Property proofs (`verify_gated_parity.py`): nests softmax to 7.5e-8; gate params differentiable;
deterministic to 0.0; param delta = `dim+2H` exactly.

## 3. CIFAR-10 — parity (no win), the real target

TinyViT, 100 epochs, dropout-matched baseline, 5 seeds:

| arm | val_acc | val_loss |
|---|---|---|
| MHA (dense) | 83.95 ±0.30 | 0.908 |
| gated β=0 (parity) | 83.72 ±0.14 | 0.914 |
| gated β=0.5 (live) | 83.90 ±0.19 | 0.908 |
| gated β=0.5 + repulsion (λ active) | 84.00 ±0.10 | 0.908 |

All Welch |t| < 1.5 → **no significant difference**. Also confirmed with a dedicated
`rep_key` (λ_init=1) run: 83.79 ±0.22 vs mha 83.68 ±0.19 (Δ+0.10, t=0.72). **CIFAR ties.**

## 4. Synthetic probes — where it wins, and why

Two self-contained tasks (`synthetic_needle.py`, `synthetic_redundancy.py`):
- **needle**: 1 signal token among many distractors (long-context dilution).
- **redundancy**: true class from `n_sig` diverse tokens vs a decoy class = **one token copied
  m times** (adversarial redundant clique); label = the class with more *distinct* tokens.

### 4a. F2 family map (which mechanism wins where)
`rep_key`/`rep_val` (pairwise repulsion) win on redundancy; per-token gates (`modular`,`card`)
help less; `submod` inert. On the needle only `rep_key` shows a small robust edge.

### 4b. Comparison vs published baselines (redundancy task, 3 seeds)

| method | acc±std | eval loss | Δacc vs dense |
|---|---|---|---|
| MHA (dense) | 95.27 ±1.36 | 0.110 | — |
| sparsemax [Martins '16] | 85.81 ±6.62 | 0.313 | **−9.45** |
| entmax-1.5 [Peters '19] | 96.21 ±0.35 | 0.091 | +0.94 |
| modular gate (= our λ=0) | 96.87 ±0.20 | 0.078 | +1.60 |
| **rep_key (ours)** | 98.80 ±0.06 | 0.030 | +3.53 |
| **rep_val (ours)** | **99.26 ±0.09** | **0.020** | **+4.00** |

Our repulsion beats dense **and** both sparse-attention algorithms; the sparse family
(magnitude-based) can't remove high-relevance duplicates and can *hurt* (sparsemax −9.45; on
the needle entmax −4.10). Ordering: **repulsion ≫ per-token gate/entmax ≫ dense ≫ sparsemax**.

## 5. The key principle — adversarial vs benign redundancy

Within-context redundancy fingerprint (`code_fingerprint.py`):

| domain | pair-redundancy | max-clique/L | our result |
|---|---|---|---|
| real Python code | 0.033 | 0.101 | (would tie) |
| synthetic redundancy | 0.019 | 0.117 | **win** |
| CIFAR patches | **0.177** | **0.436** | tie |

**CIFAR is the *most* redundant yet ties; the synthetic task is the *least* redundant yet wins.**
Conclusion: what matters is not *how much* redundancy exists but whether it is **adversarial**
(misleading duplicates that dominate by count). Benign redundancy (repeated-but-correct tokens,
as in code/images/clean text) does not hurt dense attention, so anti-redundancy doesn't help.

## 6. Efficiency

Factored repulsion is O(n²d) (same order as attention), numerically identical to the Gram form
(6e-6). Per-step time vs our dense proxy: 5.9× / 8.8× / 12.6× at L=512/1024/2048 — repulsion adds
a flat ~1.4× over the gate; the remaining gap is the unfused-vs-flash-attention penalty shared by
all gated variants (fixable with a custom kernel).

## 7. Prior art & novelty (honest)

Surveyed 48 methods. Closest: **DppNet** (NeurIPS'19, DPP marginal from key similarity — closest
cousin), **ToMe** (ICLR'23, *identical* key-cosine redundancy signal, but merges for efficiency
with size-weighting that *preserves* count), **DPP-A** (eLife'23), **Repulsive Attention**
(EMNLP'20, between heads), **entmax/sparsemax** (= our modular/sparse baselines), coverage
mechanisms, slot attention. **Verdict: a novel recombination + clean mean-field derivation, not
a new concept.** Fair claim: *the mean-field limit unifying DppNet-style repulsion with ToMe's
key-redundancy signal, as a soft, param-neutral, differentiable per-query in-softmax gate that
keeps all tokens.* Not "first DPP/repulsion in attention" (DppNet/DPP-A) nor "first key
redundancy" (ToMe).

## 8. LLM pretraining — would this help? No.

- The pathology (adversarial redundancy) is absent from clean, corpus-deduped pretraining data
  (benign redundancy → tie, like CIFAR).
- Next-token prediction isn't a dominated-by-count decision.
- **Risk of harm**: penalizing key-redundancy could suppress **induction/copying heads** (which
  attend to repeated tokens for in-context learning).
- Practical: causal masking untested; breaks flash-attention fusion.
- Where it could help: inference-time RAG-with-conflicting-duplicates, long-context robustness,
  coverage generation — not pretraining.

## 9. Honest bottom line

| question | answer |
|---|---|
| Beat dense at equal params? | Yes — only on adversarial-redundancy tasks (+4.0 acc, 4× lower loss) |
| Beat published baselines (sparsemax/entmax)? | Yes, decisively; they tie-or-hurt |
| Win from "general attention" or gating? | The general (pairwise repulsion) part |
| CIFAR / real code / LLM pretraining? | Ties; pretraining could hurt (induction heads) |
| Where it wins | Adversarial redundancy / coverage (RAG conflicts, robustness, multi-doc) |

## 10. Reproduce

```bash
V=<python-with-torch>
# property proofs (CPU)
$V verify_gated_parity.py
# CIFAR parity (needs cifar-10-batches-py under ./data)
bash sweeps/run_parity_gated.sh
# synthetic + baseline comparison (GPU)
bash baseline_grid2.sh          # mha, sparsemax, entmax15, modular, rep_key, rep_val
bash redundancy_grid.sh         # redundancy-strength sweep, 5 seeds
$V code_fingerprint.py          # within-context redundancy of real code vs tasks vs CIFAR
```

Key files: `gated_attention.py` (mechanism), `MATH.md` (derivation), `f2_sweep.py` (F2 families +
sparsemax/entmax baselines + tasks), `synthetic_{needle,redundancy}.py`, `code_fingerprint.py`,
`train_vit_cifar.py` (`--attention gated [--gated-beta-init --gated-repulsion --gated-lambda-init]`).

---

## 11. Real-image validation: MNIST-bags MIL vs SOTA

Multiple-Instance-Learning on **real MNIST images** (`mil_mnist.py`, `mil_abmil.py`): a "bag"
of digit images with one bag label; a CLS/pooling query attends over per-image embeddings.
All methods share one backbone (encoder + gated-ABMIL scorer + head) so only the pooling
differs — parameter-matched. Redundancy bags = `n_sig` distinct class-y images + one class-y'
image copied `m` times (adversarial clique) + background; label = class with more DISTINCT
instances. Needle bags = 1 signal among N-1 diverse distractors. `--attn rep` uses a λ-warmup
(repulsion off for the first 40% of steps so the encoder learns first — active-λ-at-init
otherwise corrupts the encoder → chance).

**REDUNDANCY (5 seeds, 2500 steps, chance=10%):**

| method | acc mean±std | eval loss | Δ vs Gated-ABMIL |
|---|---|---|---|
| Gated-ABMIL [Ilse'18] (SOTA) | 24.16 ±1.16 | 2.159 | — |
| AEM [2024] (SOTA anti-concentration) | 10.91 ±1.94 | 2.289 | −13.25 (t=−11.7) |
| sparsemax | 28.20 ±1.67 | 2.068 | +4.05 |
| entmax-1.5 | 23.02 ±2.63 | 2.161 | −1.14 |
| dedup (cheap defense, keep 1/near-dup group) | 33.98¹ | 1.691 | +9.8 |
| count-norm voting (cheap defense) | 33.75¹ | 1.662 | +9.6 |
| **rep (OURS)** | **71.23 ±10.33** (79.7¹) | **0.911** | **+47.1 (t=9.1)** |

¹ single-seed (the 5-seed dedup grid was blocked by stuck uninterruptible-state GPU procs; the
gap is unambiguous regardless — same-seed rep 79.7 vs dedup 34.0 ≈ 2.3×).

Our worst seed (53.5) beats Gated-ABMIL's best (25.9). **NEEDLE (benign):** all tied ~39.8
(rep +0.16, t=0.24) — no false win.

Key notes:
- **The cheap defenses do NOT match us** (rebuts the #1 reviewer objection): dedup / count-norm
  voting reach only ~34% vs rep ~72–80%. They remove the *count* advantage, but the residual
  "pick the class with more distinct instances" is still hard for ABMIL pooling, and a single
  high-relevance decoy still wins per-instance attention; rep's soft, end-to-end repulsion
  handles both. (Caveat: the task label *is* distinct-count, so dedup is a natural oracle-ish
  defense — that it still underperforms by ~2× is the notable result.)
- **AEM fails (chance)** because entropy-maximization spreads attention onto the numerous clique
  — the wrong direction.
- rep has **high variance** (±10.3, λ-warmup-sensitive). This is a *custom* adversarial-redundancy
  construction, not a standard leaderboard task.

### 11.1 Scaling: theory ↔ experiment

The separation theorem (`THEORY.md`: first-order clique influence Θ(m); mean-field repulsion
Θ(log m)) predicts how the clique's gate mass scales with clique size m. Confirmed at both
operating points:

- **Numerical, fixed moderate λ** (`theory_check.py`): rep clique gate `W ≈ 0.41 + 0.82·ln m`
  (**R²=0.999**); first-order = m. → the log-m law, exactly.
- **Trained models, learned λ** (`scaling_figure.py`, 3 seeds):

  | m | ABMIL gate (Σν) | ABMIL acc | rep gate (Σg) | rep acc |
  |---|---|---|---|---|
  | 2 | 2.00 | 26% | 1.70 | 65% |
  | 8 | 8.00 | 26% | 0.86 | 67% |
  | 16 | 16.00 | 25% | 0.08 | 71% |
  | 32 | 32.00 | 26% | 0.00 | 71% |
  | 128 | 128.00 | 25% | 0.00 | 63% |

  - **ABMIL gate = m exactly** (R²=1.0 vs m) — first-order counts every copy → **Θ(m) confirmed**.
  - **rep gate → 0** — the trained model learns *aggressive* repulsion (stronger than the
    theoretical log-m; fully eliminates the clique by m≈16).
  - **Accuracy holds across all m**: rep ~63–71% vs ABMIL flat ~25% (m=4 = one noisy seed).

Net: first-order attention's clique influence is provably and empirically **Θ(m)** (unbounded);
repulsion **bounds/eliminates** it — log m at the theoretical operating point, →0 at the learned
one — and the accuracy gap holds at every clique size.

### 11.2 Head-to-head vs the nearest cousins (novelty)

Redundancy MIL, 3 seeds, shared backbone (`novelty_grid.sh`):

| method | acc ±std | eval loss |
|---|---|---|
| ToMe (proportional, ICLR'23) | 17.4 ±0.2 | 2.22 |
| Gated-ABMIL (SOTA) | 24.9 ±0.8 | 2.16 |
| ToMe-dedup (no size-weighting) | 34.6 ±1.5 | 1.66 |
| exact DPP marginal (DppNet / DPP-A lineage) | 61.8 ±0.9 | 0.96 |
| **rep (ours, mean-field DPP)** | **72.0 ±13.1** | 0.89 |

- **Beat ToMe decisively** (+54.6, t=5.9). ToMe-as-published (proportional attention) *fails*
  (17% < dense) — its `+log(size)` weighting preserves the count-domination we remove; even
  ToMe-without-size (dedup, 35%) can't solve the residual.
- **vs exact DPP marginal: accuracy is a statistical tie** (72 vs 62, Δ+10 but **t=1.09** — rep's
  variance is high: `[53.5, 79.7, 82.8]` vs DPP's `[60.5, 62.4, 62.5]`; rep wins 2/3 seeds). So the
  honest advantage over exact DPP is **O(n²d) mean-field vs O(n³) matrix-solve + full
  differentiability + parameter-neutrality**, *not* a significant accuracy gain. The exact DPP is a
  strong, stable baseline — which validates the DPP principle the method rests on.
- **Novelty positioning (defensible):** the mean-field DPP marginal realized as a param-neutral,
  differentiable, in-softmax gate — cheaper than exact DPP (DppNet/DPP-A) and, unlike ToMe, it
  *drops* the size-weighting so it actually removes count-domination.

### 11.3 Over-firing control (no harm when redundancy is legitimate)

A reviewer-critical check: does rep *hurt* when redundancy is legitimate signal? The
**majority-by-count** task (`--task majority`): two identical-copy cliques of classes y, y' with
sizes m1, m2; label = the *majority* class **by count** — so de-duplicating is the *wrong* thing.

| method | acc (majority task) |
|---|---|
| Gated-ABMIL | 90.7¹ |
| rep (ours) | 91.3¹ |

¹ single seed (3-seed confirming). rep **matches** ABMIL — it learns to *not fire* (λ→0) when
redundancy is the legitimate signal → **no over-firing cost.** Together with the needle and CIFAR
ties, this establishes: rep helps on *adversarial* redundancy and does not harm on *benign* or
*legitimate*-redundancy data.

Ablations recap (across the paper): rep_key vs rep_val (synthetic: rep_val 99.3 ≥ rep_key 98.8);
λ=0 (= the `modular` gate ≈ baseline, vs rep +47); warmup on/off (off → chance, active-λ corrupts
the encoder); variance-reduction (§13: fixed-λ and more mean-field iters both fail).

### 11.4 Generality (architecture + scale)

The win is not an artifact of one tiny model. In a real multi-head **self-attention Transformer**
(`f2_sweep`, a *distinct* architecture from ABMIL pooling), on synthetic adversarial redundancy,
the rep advantage holds across model width (3 seeds):

| dim | mha | rep_key | rep_val | Δ(rep_val−mha) |
|---|---|---|---|---|
| 32 | 94.5 | 98.0 | 98.1 | +3.6 |
| 64 | 94.5 | 98.7 | 99.0 | +4.5 |
| 128 | 93.9 | 98.8 | 99.2 | +5.3 |
| 256 | 95.6 | 98.6 | 99.0 | +3.4 |

So across the paper the effect is demonstrated over **2 architectures** (self-attention Transformer
+ ABMIL pooling) × **2 modalities** (Gaussian tokens + real MNIST images), robust across scale.

### 11.5 Standard real MIL benchmarks (MUSK, Camelyon16) — honest: a tie and a *negative*

Two canonical real MIL benchmarks that other MIL papers report on. Both have redundancy that is
*benign relative to the label*, so the honest expectation is no-harm/parity, not a win — and one
of them exposed a genuine failure mode (and its fix, §11.6).

**MUSK1 / MUSK2** (Dietterich et al. 1997), standard 10-fold CV, shared backbone (`musk_mil.py`):

| mode | MUSK1 acc | MUSK2 acc |
|---|---|---|
| abmil (Gated-ABMIL) | 87.9 ±9.2 | 89.2 ±7.0 |
| dedup | 87.9 ±10.5 | 84.2 ±9.1 |
| dpp (exact) | 87.9 ±9.2 | 89.2 ±5.4 |
| **rep (ours)** | **87.9 ±9.2** | **87.3 ±10.9** |

→ statistical tie / **no harm on the standard benchmark** (rep's larger MUSK2 spread is the same
λ-optimization variance addressed in §11.6). ABMIL matches its literature numbers, validating the
setup.

**Camelyon16** (Owkin Phikon features, standard fixed 269-train / 130-test split, `camelyon_mil.py`),
metric = test AUC (3 seeds). This is a **needle** task — a slide is positive iff *any* patch is
tumor — where the positive signal is *itself a redundant clique* (tumor regions span many similar
patches). So *suppressing* redundancy suppresses the evidence:

| mode | test AUC | test acc |
|---|---|---|
| abmil (plain softmax) | **0.946 ±.010** | 89.7 |
| dedup | 0.925 ±.006 | 88.5 |
| dpp (exact) | 0.921 ±.016 | 89.0 |
| **rep — default (λ-init 1, fixed LR)** | **0.858 ±.031** | 82.6 |

`rep` **loses** at the default config, and the damage is monotone in suppression aggressiveness
(abmil > dpp ≈ dedup > rep). Diagnostic: `rep` kept **learned λ ≈ 0.99** (≈ its init) — it did
*not* learn to turn itself off, unlike the clean over-firing control (§11.3). Root cause and fix
in §11.6. **Honest takeaway: our method is not a drop-in for arbitrary MIL — on redundancy-is-
signal tasks naïve repulsion hurts.**

### 11.6 λ is *learnable* — the optimization fix (and it resolves the variance gap)

The Camelyon negative traced to a single scalar (λ) getting a **weak gradient** under the shared
base LR + the late λ-warmup: λ barely moved from its init in *either* direction, so it behaved
like a hyperparameter, not a learned quantity. Fix: give the gate params (λ, τ, β) their **own
high learning rate** (`--lambda-lr`, a separate optimizer group; base backbone LR unchanged).

**λ becomes genuinely learned and init-independent on adversarial data.** Adversarial-redundancy
MIL (task=redundancy, 3 seeds), converging to λ ≈ 0.30 from *either* init:

| config | acc | learned λ |
|---|---|---|
| abmil | ~25 | — |
| rep, default (fixed LR) | 71 (needs λ-init 1) / 25 (λ-init 0.1) | stuck at init |
| **rep + high λ-LR, init 0.1** | **85.6 ±0.5** | 0.1 → **0.29** |
| **rep + high λ-LR, init 1.0** | **85.7 ±0.5** | 1.0 → **0.30** |

The win now (a) **beats the old fixed-λ result** (85.7 vs 71), (b) is **init-independent** (λ
converges to ~0.30 regardless of start), and (c) has **near-zero variance** (±0.5).

**And it recovers Camelyon** (redundancy-is-signal), test AUC:

| config | AUC | learned λ |
|---|---|---|
| rep, default (λ-init 1, fixed LR) | 0.858 ±.031 | 0.99 (stuck) |
| **rep + high λ-LR, init 1.0** | **0.917 ±.004** | 0.85 |
| rep + high λ-LR, init 0.1 | 0.916 ±.034 | wanders {0.13,0.23,1.19} |

Recommended single default = **λ-init 1.0 + high λ-LR**: stable on both regimes (adversarial 85.7
±0.5, Camelyon 0.917 ±0.004). Residual honesty: on Camelyon `rep` still lands slightly below plain
softmax (0.917 vs 0.946) — repulsion *cannot* help when redundancy carries the signal; the fix
just keeps the harm small and the variance tiny.

**This also resolves the top open limitation (§13, variance).** The ±12–13 seed variance was
largely an *under-optimized-λ* artifact: with a proper λ-LR the adversarial win is 85.7 ±0.5 and
Camelyon is 0.917 ±0.004. The earlier "fixed-λ fails / more MF-iters fail" conclusions stand — the
right lever was the λ *learning rate*, not a fixed λ or more inner iterations.

### 11.7 Real RAG conflicting-evidence data (RAMDocs) + poisoning — HONEST NEGATIVE

RAMDocs (Wang et al. 2025), 500 real questions with conflicting/duplicated/noisy retrieved docs.

- **Native RAMDocs is not our regime.** misinfo 0.61 vs correct 3.84 docs/Q; majority-vote-by-count
  is wrong in only 2% of Qs; the real difficulty is *ambiguity* (400/500 multi-gold). A native run
  ties — "conflicting-evidence RAG" ≠ "adversarial redundancy".
- **Retrieval-poisoning stress test** (inject `k` duplicate misinfo copies — the PoisonedRAG threat):
  - *Surface (hashed-BoW) space* (`ramdocs_poison.py`): `majority` collapses **90.7→0.0** (count-
    domination confirmed on real text); among *learned* attention rep is the most robust (Δ −7.8 vs
    softmax −10.9, dpp −11.6); **but hard `dedup` is immune (97.7) and beats rep** — the poison is
    exact-duplicate, trivially caught by a cosine threshold.
  - *Semantic space* (real TinyLlama-1.1B embeddings, centered + top-10 removed; `ramdocs_semantic.py`),
    **paraphrase** poison (50% token-drop → surface-diverse, copy-copy cos 0.76, copy-legit 0.00):
    softmax **88.4 flat** (learned relevance + poison-augmented training already defends),
    semantic-dedup **89.9–93.8 flat**, `majority` **→0.0**, and **rep is the WORST learned method
    (86.0)** — it over-suppresses the legitimately-redundant *correct* docs (multiple correct docs
    per Q = redundancy-is-signal, cf. Camelyon §11.5).
- **Verdict: on real RAG data we do NOT beat the cheap `dedup` defense.** Only the pure count-vote
  rule fails, and that is defended by either learned relevance or a one-line dedup. rep's advantage
  is confined to controlled adversarial-redundancy where the legitimate signal is *not* itself
  redundant — a conjunction absent from the standard real benchmarks tested.

### 11.8 Overall real-data scorecard (honest)

| real dataset | rep vs baselines |
|---|---|
| MUSK1 / MUSK2 (MIL) | tie (no harm) |
| Camelyon16 (MIL) | loses at default; λ-LR fix → *near*-tie, still < plain softmax |
| RAMDocs exact-dup poison | dedup immune & beats rep; rep best only among *soft* methods |
| RAMDocs paraphrase poison | loses to softmax AND dedup (over-suppresses legit redundancy) |

**We have no real-data win over cheap defenses.** rep's clean wins (§4, §11) are on controlled
adversarial-redundancy constructions. Honest positioning: this is a **methods/theory** contribution
(unified differentiable determinantal gate nesting softmax/sparse/DPP; separation theorem; adaptive-λ),
not an empirical-SOTA-win paper.

## 12. Why repulsion wins (mechanism)

Softmax, per-token gates, and sparse attention are all **first-order**: a token's weight
depends only on its own relevance `a_i`. So `m` near-identical tokens each get weight ∝ `e^{a_i}`
and **together grab `m·e^{a}` — mass grows with count.** No first-order operator escapes this:
sparsemax truncates *low*-relevance tokens but the clique is *high*-relevance (kept, and the
background it removes concentrates *more* mass on the clique); AEM pushes toward *uniform*,
which is pure count-voting → clique dominates → chance.

Repulsion is the only mechanism that reads **second-order** (token–token) structure:
`g_i = σ(β(a_i − τ − λ r_i))`, `r_i = ⟨k_i, Σ_j g_j k_j⟩`. A clique member is similar to all its
clones (and to the aggregate it dominates) → `r_i ≈ λ·m·‖k‖²` → its gate is suppressed **more as
the clique grows**, capping the clique's total mass to ≈ one distinct token. Distinct signal
tokens have small `r_i` → kept. Net: **repulsion converts voting-by-count into
voting-by-distinct-content** — the exact information a first-order operator structurally cannot
see. (rep_val targets redundancy in *values* = what corrupts the output; rep_key in the routing
keys; identical copies are redundant in both.)

## 13. Limitations & path to a strong (spotlight-grade) paper

Honest self-review — current state is a clean proof-of-concept, not yet spotlight:
- **Standard real benchmarks: now run, with an honest split verdict (§11.5).** MUSK1/MUSK2 → tie
  (no harm). Camelyon16 → naïve `rep` *loses* (0.858 vs 0.946 AUC) because it is a
  redundancy-is-*signal* needle task; fixed to 0.917 ±.004 by the λ-LR fix (§11.6) but still a hair
  below plain softmax. **Still missing: a real *win* on a standard benchmark** — the natural target
  is an adversarial-redundancy setting (RAMDocs/ConflictQA/PoisonedRAG, or YelpZip review-spam) with
  a trained reader/aggregator, since standard MIL redundancy is benign.
- **New scope caveat (redundancy-is-signal).** When the label *is* carried by a redundant clique
  (Camelyon tumor patches), any redundancy-suppressor (rep, dedup, DPP) hurts; rep needs the λ-LR
  fix to keep the harm small. The method helps only when redundancy is *adversarial*, not when it
  is the signal.
- **Cheap-defense baselines** (semantic dedup, self-consistency voting) must be beaten — done
  here on MNIST-bags (see §11), but must hold on real data.
- **Nearest cousins:** ✅ run (§11.2). We beat ToMe decisively; we *match* the exact DPP marginal
  on accuracy (win on compute/differentiability, not significance). A *significant* accuracy win
  over exact DPP is currently blocked by rep's variance (below).
- **Fragility (was the top open issue — LARGELY RESOLVED by the λ-LR fix, §11.6):** the ±12–13
  seed variance was *optimization/basin* variance, **not** estimator variance (forward is
  deterministic → control-variate / antithetic / Rao-Blackwell do **not** apply). Earlier fixes
  failed for the right reason: (i) *fixed* non-learned λ hurts (38–42 vs 70); (ii) more mean-field
  iterations do nothing (70.9 ±13.4 vs 70.2 ±11.7). The actual lever was the λ **learning rate**:
  λ is a single scalar getting a weak gradient under the shared base LR + late warmup, so it never
  moved. Giving (λ, τ, β) their own high LR makes λ genuinely learned and **collapses the variance**
  — adversarial win 85.7 **±0.5** (up from 71), Camelyon 0.917 **±0.004**. Remaining: confirm the
  fix on the full MNIST-bags SOTA table (§11) and re-test whether it now yields a *significant* win
  over exact-DPP (previously blocked by this variance).
- **No theory:** want a first-order-can't / second-order-can *separation result*, mean-field
  fixed-point well-posedness (contraction), and the O(n·d) complexity, formalized.
- **Scope/applicability:** only works when you *train* the attention (not frozen LLMs), so RAG's
  natural home (frozen readers) is out; the honest deployable home is trainable MIL/set models.
- **Generality:** one tiny model, small scale; need multiple architectures/scales + a real
  self-attention Transformer result, and evidence of no-harm on benign data at scale.
- **Analysis depth:** attention visualizations of clique suppression, the redundancy-scaling
  curve (gap grows with clique size), λ ablations, over-firing control on benign-corroboration.

## 14. Byzantine-robust aggregation — a real fit (structure check PASSES)

The recurring blocker on real data was condition 5: truth is usually *corroborated* (redundant), so
rep suppresses it. Byzantine-robust aggregation *inverts* this cleanly: honest worker updates are
naturally **diverse** (non-IID shards), while colluding attackers send **correlated** updates to pull
the aggregate by count. Here suppressing the correlated cluster is exactly right, and honest signal
is not the redundant one — all five win-conditions hold.

**Structure check** (`byzantine_check.py`, robust-mean estimation, 30 seeds; error =
‖aggregate − μ‖/‖μ‖, lower better). rep = soft/global repulsion on the update set (down-weight
updates redundant with the gated aggregate). Baselines: mean, coordinate-median, trimmed-mean, Krum,
centered-clipping (CClip, Karimireddy'21), and **FoolsGold** (the nearest cousin — a Sybil defense
that down-weights *pairwise*-similar gradients).

- **Tight collusion** (σ_b=0.05): rep ≈ FoolsGold (tied; FoolsGold slightly better mid-range). Krum
  is *destroyed* (~6.0 — picks the dense malicious cluster); median/trimmed/CClip degrade with f.
- **Loose / evasive collusion** (σ_b=1–2, attackers diversify to evade detection): **rep dominates
  every baseline up to 50% Byzantine** — e.g. σ_b=2, f=0.4: rep **1.61** vs next-best 2.10 (CClip),
  FoolsGold 2.87 (collapses toward mean — attackers no longer pairwise-similar), Krum 5.7. Principle:
  FoolsGold keys on *pairwise* similarity (evadable); rep keys on redundancy *with the aggregate*
  (their collective pull survives individual diversification).
- **Honest failure boundary:** near-IID honest (σ_b spread ~0.2) *and* very low attack (f≤0.1) →
  rep slightly over-suppresses the honest consensus; and majority Byzantine (f≥0.6) → beyond standard
  threat models (Krum survives there by construction).

**Verdict:** first candidate to pass the pre-build structure check, against strong modern defenses,
in the realistic adaptive/evasive-collusion regime — and adjacent to distributed-training work.
Caveats before claiming a result: this is robust-*mean* simulation, not real FL; next step is a real
federated run (real gradients over rounds → final accuracy under attack), adaptive attacks designed
against rep, and a fuller defense suite (multi-Krum, bucketing, robust momentum).

### 14.1 Real federated CIFAR — the structure-check optimism does NOT survive (honest negative)

`fl_byzantine.py`: FedSGD, N=16 non-IID (Dirichlet α=0.5), 30% Byzantine, 200 rounds; server
aggregates real gradients each round with the full defense suite; attacks are omniscient. The
decisive `adaptive_rep` attack is norm-bounded (evades clipping) and diversified (evades FoolsGold's
pairwise-similarity), *crafted to evade rep*.

Final test acc (%), byz=5/16:

| defense | none | signflip | ipm | alie | **adaptive_rep** |
|---|---|---|---|---|---|
| mean | 44.1 | 10.0 | 37.2 | 19.0 | 10.0 |
| comedian | 33.7 | 18.8 | 12.1 | 19.7 | 17.3 |
| trimmed | 32.5 | 20.9 | 10.6 | 16.2 | 16.8 |
| krum | 32.0 | 10.5 | 14.1 | 10.5 | **32.2** |
| cclip | 42.8 | 11.4 | 38.3 | 18.6 | 17.8 |
| foolsgold | **44.2** | **43.1** | **40.6** | 40.0 | 10.0 |
| **rep (ours)** | 37.7 | 39.9 | 18.3 | **42.3** | 16.1 |

**Honest verdict: rep does NOT win in real FL.**
- **Clean penalty**: rep 37.7 vs mean/FoolsGold 44 — rep always represses redundancy, so it costs
  accuracy even with no attack; FoolsGold gracefully reduces to `mean` when there's no attack.
- **The adaptive attack crafted against rep succeeds** (rep 16.1, *below* Krum's 32.2) — the
  make-or-break test, failed. Diversified norm-bounded byz evade rep's redundancy signal.
- rep wins exactly one column (`alie`, 42.3) and is poor on `ipm` (18.3) and `adaptive_rep`.
- No defense dominates (Byzantine no-free-lunch): FoolsGold best on none/signflip/ipm; Krum uniquely
  survives `adaptive_rep`; rep best only on `alie`.

**Why the structure check (§14) was misleading**: the robust-mean simulation used a *fixed* evasive
attack (rep won); it did not include an *adaptive* attack that reacts to rep. Real FL + an adaptive
attack refutes the clean-win hypothesis. Lesson recorded: a pre-build structure check must include
adaptive adversaries.

**Final conclusion for the project**: no real-data win over cheap/standard defenses on ANY tested
domain (MUSK, Camelyon, RAMDocs, Byzantine-FL). rep is a genuine *methods/theory* contribution
(unified differentiable determinantal gate + Θ(m)→Θ(log m) separation + adaptive-λ) that wins in
controlled adversarial-redundancy, ties/loses to specialized methods on real data. Honest scope,
solid conference paper — not a spotlight empirical-win.

### 14.2 Why rep fails (instrumented) — it is an outlier-KEEPER, not an outlier-remover

`byzantine_diag.py` logs the weight rep actually assigns to honest vs Byzantine workers on real
CIFAR gradients (uniform = 1/16 = 0.0625; a good defense wants ~0 on byz):

| attack | rep w_honest | rep w_byz | verdict |
|---|---|---|---|
| none | 0.061 | 0.066 | mild clean penalty (up-weights the distinct, not the consensus) |
| signflip | 0.091 | 0.0002 | ✓ suppresses byz (they dominate the aggregate by magnitude) |
| alie | 0.090 | 0.003 | ✓ suppresses byz |
| ipm | 0.036 | **0.122** | ✗ AMPLIFIES byz |
| adaptive_rep | 0.055 | **0.080** | ✗ more weight on byz than honest |

rep's signal is "down-weight alignment with the aggregate", which flags attackers ONLY when they
dominate the aggregate (large magnitude / majority). A norm-bounded MINORITY attack pointing away
from the honest consensus makes the aggregate point at HONEST → rep flags honest as redundant and
KEEPS the attackers. Learning/adapting λ cannot fix this — the *sign* of the signal is wrong for
minority-attacker robustness.

**Crisp scope (the cleanest statement of the whole project, backed by weights):** rep is an
outlier-KEEPER (down-weights the dense/aligned majority, up-weights distinct/isolated points). It
wins iff the *misleading* content is the redundant majority (adversarial duplicates dominating by
count); it loses/hurts when the *useful* signal is the majority — the normal case in real tasks.
This is why every real dataset failed (truth = majority = what rep suppresses) and only constructed
adversarial-redundancy tasks won.

## 15. E0 — theory/implementation reconciliation (reviewer's central math claim: CONFIRMED)

`clique_scaling.py`: total gate mass W_m of a size-m identical clique (+ n=8 distinct signal keys),
under three gates (β=1, λ=0.5, τ=0):

| m | one-step (shipped) | fixed-point (THEORY.md) | true DPP marginal |
|---|---|---|---|
| 4 | 1.59 | 2.03 | 0.91 |
| 32 | 0.0008 | 5.34 | 0.99 |
| 256 | 0.000 | 12.5 | 1.00 |
| 4096 | 0.000 | 21.1 | 1.00 |

- **One-step gate (what `gated_attention.py` ships)** → W_m **collapses to 0** (the `m·e^{-cm}` shape:
  peaks at m≈4 then crashes). Our trained "clique gate → 0" observation validates *this one-step rule*,
  NOT the Θ(log m) theorem. It *over*-suppresses (a clique gets ~0 votes, not "one vote").
- **Fixed point (what THEORY.md proves)** → grows ~log-ish but the iteration does **not converge** past
  the contraction limit `βλ‖K‖_∞<4` (m≳7 here) → oscillates (m=128:6.2, 256:12.5, 512:9.8). The
  theorem is stated exactly where its own well-posedness fails.
- **True DPP marginal** (`K=L(L+I)⁻¹`, `g_i=K_ii`) → clean **Θ(1)≈1.0**: a size-m duplicate clique
  counts as **one effective item** — the actual clone-invariance ("one vote per cluster") you want.

**Corrections forced:** drop the "Θ(log m)" and "DPP special-case" claims (our `F₂=Σa−λΣ⟨k_i,k_j⟩`
is pairwise-inhibitory Ising, not `det(L_S)`; raw dot-product rewards *anti*-aligned pairs). The true
DPP is the correct clone-invariant primitive. **But** even Θ(1) is a *fixed* diversity bias → it still
discounts legitimate corroboration (the provenance-free no-free-lunch, §14.2). Beating regular
attention requires a *context/provenance-aware* rule that decides per example whether multiplicity
should count — the next experiment (E1), benchmarked against 2-layer attention / Set Transformer /
attention+source-features.

## 16. E1a — the consensus benchmark is validated (non-gameable + provenance window found)

`task_consensus.py`: binary latent truth; two content clusters MATCHED in #items, #distinct
surface-ids, and per-item content spread (exchangeable in (V,S)), differing ONLY in true-origin
structure (honest = n independent origins; Sybil = 1 origin relabeled across n surface-ids), revealed
via a NOISY same-origin graph (edge accuracy γ). α = Sybil/honest content spread.

Gameability + window probe (3000 ex/cell, binary chance = 50%):

| slice | item-maj | surf-count | oracle (provenance, robust) |
|---|---|---|---|
| **α=1, γ=1.0** | 49.6 | 49.6 | 100.0 |
| **α=1, γ=0.8** | 49.7 | 49.7 | 98.3 |
| **α=1, γ=0.7** | 48.9 | 48.9 | 94.2 |
| **α=1, γ=0.6** | 48.6 | 48.6 | 84.4 |
| α=0.5, γ=0.8 | 38.1 | 38.1 | 98.4 |
| α=2.0, γ=0.8 | 81.0 | 81.0 | 95.2 |

**α=1 is the swap-symmetric hard slice**: content/count baselines sit at chance, and only the
provenance channel separates truth — the empirical no-free-lunch for any (V,S)-only reader. The
robust oracle (within-cluster mean of the noisy graph, which averages out edge noise) is near-perfect
and graceful in γ. Headline cell for the mechanism study: **α=1, γ=0.8** (oracle 98%, clearly
non-trivial noise, cheap = chance). This validates the testbed before building mechanisms/baselines.

## 17. E1b — the consensus mechanism study (P1/P2): a real, honestly-scoped win

Shared k=1 self-attention encoder + swappable pooling head -> C logits (predict latent truth),
matched params. `consensus_models.py`, headline cell alpha=1, gamma=0.8, frozen test, 3 seeds.

**P1 (no-free-lunch, CONFIRMED).** Content-only models sit at chance — regular attention *provably
cannot* win when content/count are swap-symmetric:

| arm | test acc | params |
|---|---|---|
| softmax (content) | 49.2 ±1.2 | 35,908 |
| Set Transformer (ISAB+PMA, content) | 49.1 ±0.3 | 40,388 |

**P2 (provenance). Accuracy at n=6000:** all provenance arms win (92–96%), but on RAW accuracy our
mechanism is NOT best — a trivial 1-feature degree-concat beats it:

| arm (provenance) | test acc |
|---|---|
| prov_concat (degree feature + softmax) | 95.9 ±0.3 |
| relation_bias (Pgraph as attn bias) | 94.2 ±0.8 |
| m2_prov (ours) | 92.2 ±0.3 |

**P2 sample-efficiency (the WIN).** m2_prov's inductive bias dominates in the low-data regime:

| n_train | prov_concat | relation_bias | **m2_prov** |
|---|---|---|---|
| 200 | 52.0 | 45.0 | **80.2** |
| 400 | 59.1 | 46.6 | **83.7** |
| 800 | 84.4 | 51.8 | **87.6** |
| 1600 | 94.0 | 82.5 | 89.9 |
| 3200 | 95.4 | 91.8 | 91.4 |

m2_prov reaches 80% with ~200 bags where the baselines are at chance (~4x fewer bags to 80% than
prov_concat); baselines catch up/overtake only at high data. Classic useful-inductive-bias signature.

**Honest thesis (defensible, pre-registered):** (1) NO content-only rule — softmax OR Set Transformer
— beats chance on the mixed multiplicity-vs-corroboration distribution (empirical no-free-lunch);
(2) a noisy same-source provenance signal is necessary AND sufficient; (3) a source-aware pooling
*inductive bias* (down-weight within-content-neighbor same-origin density) is far more
SAMPLE-EFFICIENT at consuming that signal than feeding it to generic attention — while at high data
the bias becomes unnecessary (baselines match). NOT a raw in-distribution expressivity win.
MVP scale, 3 seeds, one cell — full protocol (>=10 seeds + paired CIs, worst-case-over-alpha, OOD-cell
extrapolation, latency) is the next step to lock it.

### 17.1 Fixing the ceiling: the source-aware gate Pareto-DOMINATES across all data budgets

The rigid m2_prov loses at high data only because its gate is 3 scalars (under-capacity), not because
the source-aware idea is wrong. Two expressive variants (alpha=1, gamma=0.8, 3 seeds, matched params):
- `m2_prov_x` = gate is an MLP on [within-neighbor same-origin density, degree, relevance].
- `m2_prov_r` = **rigid gate + ZERO-INIT residual MLP** (starts exactly as m2_prov, relaxes with data).

Test accuracy vs #training bags:

| n_train | prov_concat | m2_prov (rigid) | m2_prov_x (MLP) | **m2_prov_r (rigid+residual)** |
|---|---|---|---|---|
| 200 | 52.0 | 80.2 | 53.0 | **81.8** |
| 400 | 59.1 | 83.7 | 70.1 | **85.1** |
| 800 | 84.4 | 87.6 | 89.1 | **91.7** |
| 1600 | 94.0 | 89.9 | 93.6 | 94.0 |
| 3200 | 95.4 | 91.4 | 95.1 | 95.6 |
| 6000 | 95.9 | 92.2 | 96.2 | 95.9 |

**m2_prov_r >= prov_concat at EVERY data size** — +30 pts at n=200 (where the flexible baseline is at
chance), tapering to an exact tie at the ~96 ceiling. The zero-init residual gives the best of both:
the rigid density prior's sample-efficiency AND the flexible ceiling. m2_prov_x confirms the ceiling
is recoverable (96.2 >= prov_concat 95.9) but needs data to train its MLP (loses the extreme-low-data
head start). Bias-variance is a knob; the residual dials it automatically.

**Headline claim (honest, this cell, MVP scale):** content-only attention (softmax AND Set Transformer)
is at chance (no-free-lunch); given a noisy same-source provenance signal, a source-aware pooling gate
(rigid density prior + zero-init residual) **Pareto-dominates** a flexible provenance-attention baseline
across the full data-budget spectrum at matched params — large gains when labels are scarce, no cost at
scale. Answers "why lose at high data": the rigid ceiling, now fixed. Next: full protocol to lock
(>=10 seeds + paired CIs, worst-case-over-alpha incl adaptive, OOD-cell extrapolation, latency).

## 18. E1c — FULL PROTOCOL (10 seeds, paired CIs, pre-registered): the locked result

Pre-registration in `PREREG.md` (primary = balanced test acc @ alpha=1, gamma=0.8; frozen test seed
777; H1-H4 + kill criteria). Non-gameability gate PASSES (item-maj 50.2, surf-cnt 50.2, oracle 98.2).
Arms: shared k=1 attn encoder + swappable head, matched params (within 0.2%). `bench_full.py` +
`aggregate.py`, 10 seeds.

**H1 no-free-lunch — CONFIRMED.** Content-only at/below chance at every size: softmax 48.6->50.2,
set_transformer 35->49. Regular attention (and Set Transformer) cannot beat chance.

**H2 low-data win — CONFIRMED (parity at scale).** Learning curve (test acc, mean±std):

| arm | n=200 | n=400 | n=800 | n=1600 | n=3200 | n=6000 |
|---|---|---|---|---|---|---|
| softmax | 48.6 | 48.9 | 49.9 | 49.5 | 49.5 | 50.2 |
| set_transformer | 35.1 | 39.8 | 43.7 | 45.8 | 47.6 | 49.1 |
| prov_concat | 52.2 | 60.6 | 85.1 | 94.4 | 95.5 | 95.7 |
| relation_bias | 45.0 | 47.1 | 51.3 | 79.8 | 92.4 | 94.3 |
| m2_prov (rigid) | 80.6 | 84.6 | 87.6 | 90.0 | 91.6 | 92.9 |
| m2_prov_x (MLP) | 54.3 | 70.9 | 87.7 | 93.6 | 94.9 | 96.2 |
| **m2_prov_r (ours)** | **81.8** | **85.3** | **91.4** | 93.7 | 95.5 | 95.8 |

paired Δ(m2_prov_r − prov_concat), 95% bootstrap CI: n=200 +29.6 [28.5,30.7]; n=400 +24.7 [23.0,26.5];
n=800 +6.3 [3.6,8.7] (all wins); n=1600 −0.6 [−1.0,−0.1] (baseline +0.6, marginal); n=3200 −0.0; n=6000
+0.1 (ties). => large win when labels scarce, parity at the ceiling, one negligible 0.6-pt dip.

**H3 robustness (worst-case-over-alpha = adaptive adversary, n=800) — CLEAN WIN:**

| arm | α=0.5 | α=1 | α=2 | worst |
|---|---|---|---|---|
| softmax | 81.1 | 49.9 | 97.0 | 49.9 |
| prov_concat | 86.3 | 85.1 | 96.8 | 85.1 |
| relation_bias | 82.9 | 51.3 | 96.6 | 51.3 |
| **m2_prov_r** | 89.0 | 91.4 | 96.6 | **89.0** |

m2_prov_r worst-case (89.0) beats regular attention (49.9), cheap prov-concat (85.1), and relation-bias
(collapses to 51.3 at the hard slice). Most robust to the adaptive adversary.

**H4 OOD (train n_orig[3,6] -> test [8,10]):** tie/non-differentiating (shift makes it easier; all
~98-99.7; m2_prov_r 99.7 = prov_concat 99.7).

**Latency @ L=64:** m2_prov_r +84 params (0.2%), ~1.7x fwd, fwd+bwd comparable.

**LOCKED VERDICT:** content-only attention is at chance (no-free-lunch); given a noisy same-source
provenance signal, our source-aware gate (rigid density prior + zero-init residual) is (i) far more
SAMPLE-EFFICIENT (paired-CI win, +6 to +30 pts at n<=800), (ii) at parity at scale, and (iii) the MOST
ROBUST to an adaptive adversary (worst-case-over-alpha 89.0 vs <=85.1), at matched params and ~1.7x
fwd cost. Honestly scoped: not a raw high-data expressivity win; the contribution is inductive bias +
robustness for consuming provenance. (MVP model scale; single task family.)

## 19. Phase diagram over alpha (honest characterization; n=800, 5 seeds)

Wide alpha-sweep (`bench_full.py --exp B`, α = Sybil/honest content spread; chance=50):

| α | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 | 8.0 | worst |
|---|---|---|---|---|---|---|---|---|---|---|
| softmax (reg attn) | 92.1 | 80.9 | 50.4 | 91.4 | 97.0 | 98.6 | 99.1 | 99.4 | 99.5 | 50.4 |
| relation_bias | 95.8 | 83.4 | 52.2 | 88.4 | 96.5 | 98.6 | 99.0 | 99.3 | 99.5 | 52.2 |
| prov_concat | 94.2 | 86.1 | 86.2 | 92.0 | 96.9 | 98.6 | 99.0 | 99.3 | 99.4 | 86.1 |
| m2_prov | 88.1 | 88.3 | 87.5 | 93.6 | 96.4 | 98.2 | 98.7 | 98.9 | 98.6 | 87.5 |
| m2_prov_r (ours) | 90.3 | 88.7 | 90.6 | 94.2 | 96.6 | 98.3 | 98.7 | 98.8 | 98.6 | 88.7 |

**Reading:** regular attention is 80–99.5% everywhere EXCEPT a narrow valley at α≈1 (coordinated
duplication / swap-symmetry), where it craters to chance (50.4); relation_bias craters too (52.2)
despite provenance. Our gate has NO valley (flat 88–99), giving the best worst-case-over-α (88.7 =
adaptive-adversary robustness). Honest cost: at the easy extremes (α=0 tight dups; α≥3 diffuse) our
gate is marginally lower (e.g. 98.6 vs softmax 99.5 @ α=8) — a small peak-accuracy trade for removing
the blind spot. Defensible claim: NOT "beats attention"; rather "attention has a catastrophic blind
spot at coordinated duplication, which a provenance-aware gate removes at negligible cost elsewhere"
(quantified as best worst-case-over-α). Caveat unchanged: synthetic probe; real-world prevalence of
the α≈1 regime is unproven (real benchmarks §14 sit at α≠1-like content-informative points -> tie/lose).
