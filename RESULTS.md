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
- **Single, custom, semi-synthetic testbed.** Need wins on ≥2 *standard* real benchmarks
  (Camelyon16/TCGA MIL with official ABMIL/DSMIL/TransMIL/DTFD; or RAMDocs/ConflictQA with a
  trained reader).
- **Cheap-defense baselines** (semantic dedup, self-consistency voting) must be beaten — done
  here on MNIST-bags (see §11), but must hold on real data.
- **Nearest cousins:** ✅ run (§11.2). We beat ToMe decisively; we *match* the exact DPP marginal
  on accuracy (win on compute/differentiability, not significance). A *significant* accuracy win
  over exact DPP is currently blocked by rep's variance (below).
- **Fragility (top open issue — variance-reduction attempted, UNRESOLVED):** high seed variance
  (±12–13). It is *optimization/basin* variance (each variant has ~1 bad seed ~45–50 vs the rest
  ~70–83), **not** estimator variance — the forward is deterministic, so control-variate /
  antithetic / Rao-Blackwell techniques do **not** apply. Two principled fixes **failed**
  (`variance_grid.sh`): (i) fixed non-learned λ *hurts* accuracy (38–42 vs 70) and doesn't reduce
  variance (the model needs to adapt λ per input); (ii) more mean-field iterations (3 vs 1) give
  no change (70.9 ±13.4 vs 70.2 ±11.7). This variance is what blocks a *significant* win over the
  exact-DPP baseline. Untried levers: longer training / warmup-schedule sweep / LR &
  regularization / restart-and-select. A genuine open limitation, documented honestly.
- **No theory:** want a first-order-can't / second-order-can *separation result*, mean-field
  fixed-point well-posedness (contraction), and the O(n·d) complexity, formalized.
- **Scope/applicability:** only works when you *train* the attention (not frozen LLMs), so RAG's
  natural home (frozen readers) is out; the honest deployable home is trainable MIL/set models.
- **Generality:** one tiny model, small scale; need multiple architectures/scales + a real
  self-attention Transformer result, and evidence of no-harm on benign data at scale.
- **Analysis depth:** attention visualizations of clique suppression, the redundancy-scaling
  curve (gap grows with clique size), λ ablations, over-firing control on benign-corroboration.
