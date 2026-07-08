# Phase A — deterministic mean-field gated attention (parity-first)

## Why the sampled version failed (all confirmed, empirically + by audit)

1. **Forward pass was ~95% noise.** Monte-Carlo relative std ≈ 0.955 at `runs=6`; you'd
   need ~540 runs to reach rel-std 0.1. Hits even non-learnable F2.
2. **Learnable F2 got zero gradient.** With `f1=restricted_softmax`, `tau_fn`/`f2_neural`
   grads are literally `None` — the module `del count`s and pools with the detached bool
   `mask`, ignoring the straight-through `sum_v`/`count_f`. The "policy" was never trained.
3. **Sampler was biased** (empty init + few Gibbs steps → support ~10/65, not stationary).
4. **Objective mismatch:** `E_S[softmax-within-S] ≠ softmax` (over-normalization within
   random subsets → different, peakier attention, not obviously better).

Plus audit extras: stochastic eval, `modular_dot` silently applies an undocumented `tau`,
straight-through gradient doesn't telescope, MHA baseline init was clobbered,
`dot_repulsion` sweep sign was attraction, `verify_exact_attention.py` only tested `full_set`.

## The fix: collapse the sampler to a deterministic mean-field functional

`gated_attention.py :: GatedSoftmaxAttention`

    a_i = q·k_i/sqrt(d)
    g_i = sigmoid( beta * (a_i - tau(q) - lambda * r_i) )     # inclusion marginal / gate
    w_i = g_i * exp(a_i) / sum_j g_j * exp(a_j)               # gated softmax
    y   = sum_i w_i v_i

- **beta = 0 ⇒ g_i = 0.5 ⇒ the constant cancels ⇒ w = softmax EXACTLY.** Starts as dense
  attention; departs only if it lowers the loss.
- **Deterministic** (no sampling) → forward variance is exactly 0.
- **Differentiable** to all gate params (fixes the None-gradient bug).
- **Param-matched:** fused qkv + out_proj identical to `nn.MultiheadAttention`; gate adds
  only `dim + 2H` params (≈0.13%).
- `r_i` (repulsion) is the Phase-B DPP hook — the one bias softmax structurally can't
  express (down-weight a key for being redundant with the other selected keys).

## Proven (CPU, `verify_gated_parity.py`)

    [1] nests softmax:   max_abs_err = 7.5e-8         (copied MHA weights, beta=0)
    [2] gate grads:      beta/tau/lambda nonzero at beta>0; tau frozen at beta=0 (sticky parity)
    [3] deterministic:   err across RNG seeds = 0.0    (vs sampled rel-std 0.955)
    [4] param match:     +198 params = dim+2H = 0.134% of MHA (D=192,H=3)

## Sticky-parity risk (from the design panel)

At `beta=0`, `d gate / d tau ∝ beta = 0`, and `beta=0` is an attractive fixed point → the
layer can stay dense forever ("match but never beat"). Use `--gated-beta-init 0.0` for the
parity proof, and a small positive value (`0.3–1.0`) for win experiments so the gate is
live from step 0.

## Run

    # Parity proof (should tie tuned dense MHA, 5 seeds, dropout-matched):
    bash sweeps/run_parity_gated.sh
    # Live-gate win attempt on CIFAR (expected: parity; real win needs large-L):
    GATED_BETA_INIT=0.5 bash sweeps/run_parity_gated.sh
    # Phase-B DPP repulsion:
    GATED_BETA_INIT=0.5 GATED_REPULSION=1 bash sweeps/run_parity_gated.sh

## Honest expectation (adversarial design panel)

- Parity on CIFAR: ~95%, mechanical.
- A *win* on CIFAR (L=65) at equal params: ~10–20% — near-worst-case; expect a tie.
- A *win* on a large-L / distractor-heavy / redundant-context benchmark: ~30–45%, and you
  must defend novelty vs entmax/sparsemax (gated softmax) and dense-CRF mean-field (DPP).
- Decisive go/no-go before spending compute: a **synthetic sparse-signal task** (K
  informative tokens among ≥65 distractors). If it can't win there, it won't win anywhere.
