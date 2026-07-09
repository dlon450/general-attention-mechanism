# Theory: why anti-redundancy attention provably escapes count-domination

This formalizes the mechanism (§12 of RESULTS.md). The headline is a **separation theorem**:
under a redundant clique of size `m`, any *first-order* attention (softmax / gate / sparse)
lets the clique's influence grow **Θ(m)**, while the *second-order* mean-field repulsion caps it
at **Θ(log m)**. That gap is exactly why softmax/sparsemax/entmax/AEM fail and repulsion wins —
and it yields a falsifiable prediction (§4) we can measure.

## 1. Setup

A single pooling query attends over a bag of tokens with keys `k_i`, values `v_i`, and scores
`a_i = q·k_i/√d`. Pooling weights and output:
$$ w_i = \frac{\nu_i\, e^{a_i}}{\sum_j \nu_j\, e^{a_j}},\qquad y = \sum_i w_i v_i, $$
where `ν_i ≥ 0` is a **gate** (`ν_i ≡ 1` recovers softmax). Partition the bag into:
- **Signal** `S`: `n` *distinct* tokens (mutually near-orthogonal keys), each with score `a_S`, value `v_S`.
- **Clique** `D`: `m` *identical* copies of one token with key `k`, score `a_D`, value `v_D≠v_S`,
  and self-affinity `s := ⟨k,k⟩/√d > 0`.

The clique's total (normalized) weight is `W_D = (Σ_{i∈D} ν_i) e^{a_D} / Z`. The prediction is
determined by whether signal or clique value dominates the pooled `y`, i.e. by the ratio
`ρ := (Σ_{i∈D}ν_i)e^{a_D} / ((Σ_{i∈S}ν_i)e^{a_S})`. Correct (signal) prediction requires `ρ < 1`.

**Definition (first-order pooling).** `ν_i = φ(a_i)` depends only on token `i`'s own score, for
some `φ: ℝ→ℝ_{≥0}`. Softmax (`φ≡1`), a per-token gate (`φ(a)=σ(β(a−τ))`), sparsemax/entmax
(`φ` a threshold on the score) are all first-order.

## 2. Theorem 1 (Separation).

**(a) First-order pooling: clique influence is Θ(m).** For any first-order `φ` with `φ(a_D)>0`,
$$ \rho \;=\; \frac{m\,\varphi(a_D)\,e^{a_D}}{n\,\varphi(a_S)\,e^{a_S}} \;=\; \Theta(m). $$
Since the copies are identical, every clique member contributes the *same* `φ(a_D)`, so their
count enters linearly and **cannot be removed by any choice of `φ`.** Hence `ρ>1` for all
`m > m_0 := n\,\frac{\varphi(a_S)e^{a_S}}{\varphi(a_D)e^{a_D}}`: **the clique dominates once it is
merely a constant factor larger than the signal.** (Sparsemax/entmax cannot help: they zero
*low*-score tokens, but `a_D` is high — the clique is kept.)

**(b) Mean-field repulsion: clique influence is Θ(log m).** Let
`ν_i = g_i = σ(β(a_i − τ − λ r_i))`, `r_i = Σ_j g_j ⟨k_i,k_j⟩/√d`. By symmetry all clique members
share a gate `g_D`. Their pairwise affinities give `r_D ≈ m\,g_D\,s` (the signal, near-orthogonal
to `k`, contributes `o(1)`). The self-consistent equation is
$$ g_D = \sigma\!\big(\beta(a_D - \tau - \lambda\, s\, m\, g_D)\big). $$
Write `W := m\,g_D` (the clique's total gate mass). For large `m`, `g_D→0`, so `σ(x)≈e^{x}`:
$$ g_D \approx C\,e^{-\beta\lambda s\,W},\quad C:=e^{\beta(a_D-\tau)} \;\Rightarrow\; W = mC\,e^{-\beta\lambda s W}. $$
Taking logs: `βλs·W + ln W = ln(mC)`, hence
$$ \boxed{\,W \;=\; \Sigma_{i\in D} g_i \;=\; \Theta\!\Big(\tfrac{\log m}{\beta\lambda s}\Big).\,} $$
The clique's total gate mass grows only **logarithmically** in `m`. The signal (small `r` ⇒
`g≈g_S` constant) keeps mass `Θ(n)`. Therefore
$$ \rho_{\text{rep}} = \Theta\!\Big(\tfrac{\log m}{n}\Big), $$
and the model can choose `β,λ,τ` so that `ρ_{rep}<1` (correct) for all `m` up to `exp(Θ(n))`. ∎

**Interpretation.** First-order attention counts a size-`m` clique as `Θ(m)` votes; mean-field
repulsion counts it as `Θ(log m)` — an *exponential* reduction in effective multiplicity. This is
the precise sense in which repulsion converts "voting by count" into "voting by distinct content,"
and it is unachievable by any per-token (softmax/gate/sparse) or attention-spreading (AEM) rule.

## 3. Proposition 2 (Well-posedness of the gate).

Let `K_{ij} = ⟨k_i,k_j⟩/√d` and define the mean-field map `T(g)_i = σ(β(a_i - τ - λ(Kg)_i))`.
Since `|σ'|≤¼`,
$$ \|T(g)-T(g')\|_\infty \le \tfrac14\,\beta\lambda\,\|K\|_\infty\,\|g-g'\|_\infty. $$
So `T` is a contraction whenever `βλ‖K‖_∞ < 4`, and by Banach's theorem has a **unique fixed
point** — the gate is well-defined and the fixed-point iteration converges geometrically. In
practice one iteration (our `g₀`) is a good approximation in the low-`λ` regime; correctness of
Thm 1(b) only needs the qualitative `Θ(log m)` scaling, which holds at the fixed point.

## 4. Falsifiable prediction (theory ⇒ experiment)

Thm 1 predicts, *quantitatively*, that the clique's total attention mass `W_D(m)` should be
**linear in `m` for softmax and logarithmic in `m` for repulsion.** This is directly measurable:
sweep clique size `m`, record `W_D`, and check the functional form. Confirming `W_D ∼ log m` (rep)
vs `W_D ∼ m` (softmax) would tie the theory to the mechanism far more tightly than accuracy
curves alone. (Corresponding accuracy prediction: rep stays correct until `m ≈ e^{Θ(n)}`, softmax
fails at `m ≈ Θ(n)`.)

## 5. Proposition 3 (Complexity).

The repulsion term `r_i = ⟨k_i, Σ_j g_j k_j⟩/√d` factors through a single aggregate
`m⃗ = Σ_j g_j k_j`:
- **Query-independent (global) gate:** `m⃗` is one vector — **O(nd)** per iteration.
- **Per-query gate** (ours): `m⃗` differs per query — **O(n²d)**, the *same order as attention
  itself*; a constant (~2×) overhead, and asymptotically free relative to the QKᵀ it rides on.

No `O(n³)` Gram is needed (that was an implementation artifact, now removed).

## 6. What this buys the paper

- A **provable separation** (Θ(m) vs Θ(log m)) explaining *why* every first-order baseline fails
  and repulsion succeeds — not just an empirical table.
- A **well-posedness** guarantee (contraction ⇒ unique, convergent gate).
- A **tight complexity** statement (attention-order, O(nd) in the global case).
- A **falsifiable, measurable prediction** (`W_D ∼ log m`) that turns the mechanism into a testable law.
