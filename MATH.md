# The math: `gated`, `gated_rep`, and "redundancy"

This explains, from first principles, the three attention variants compared in the
experiments and why only one of them handles *redundancy*. Math is written in LaTeX
(`$...$` inline, `$$...$$` display) — view in a Markdown renderer with math support
(VS Code preview, GitHub, Jupyter, etc.).

---

## 0. Notation — ordinary attention

For a query $q$ and tokens $i = 1,\dots,L$ with keys $k_i$ and values $v_i$:

$$
a_i \;=\; \frac{q^\top k_i}{\sqrt{d}} \quad\text{(relevance score)},
\qquad
w_i \;=\; \frac{e^{a_i}}{\sum_{j} e^{a_j}} \quad\text{(softmax weight)},
\qquad
y \;=\; \sum_{i} w_i\, v_i .
$$

Each token is scored **independently** — $a_i$ depends only on token $i$ — and then the
scores are normalized together. This independence is the crux of everything below.

---

## 1. The "general attention" view (the framework)

Instead of attending to all tokens, choose a **subset** $S \subseteq \{1,\dots,L\}$, run
softmax *restricted to $S$*, and average over subsets:

$$
y \;=\; \mathbb{E}_{S \sim p(S)}\!\left[\; \sum_{i \in S} \operatorname{softmax}_S(a)_i \, v_i \right],
\qquad
p(S) \;\propto\; \exp\!\big(\beta\, F_2(S)\big).
$$

- $F_2(S)$ is a **set-scoring function**: how good is subset $S$?
- $\beta > 0$ is an inverse temperature.
- Ordinary attention is the special case $F_2(S) = \big(S = \{1,\dots,L\}\big)$ (always take the full set).

Sampling $S$ is expensive and high-variance (this is what the original repo did via Gibbs
sampling, and it was ~95% noise). Instead we use the **marginal inclusion probability**

$$
g_i \;=\; \Pr[\, i \in S \,] \;\in\; (0,1),
$$

and a deterministic **mean-field** output:

$$
\boxed{\;
w_i \;=\; \frac{g_i\, e^{a_i}}{\sum_{j} g_j\, e^{a_j}},
\qquad
y \;=\; \sum_i w_i\, v_i .
\;}
$$

So $g_i$ is a **gate** multiplying the softmax numerator. If all $g_i$ are equal (e.g.
$g_i = \tfrac12$), the constant cancels and this is **exactly ordinary softmax**. Everything
now hinges on how $g_i$ is computed — which is determined by the choice of $F_2$.

---

## 2. `gated` — a *modular* $F_2$ (gate depends on each token alone)

Take the simplest set function, a **modular** one (a sum over members):

$$
F_2(S) \;=\; \sum_{i \in S} \big(a_i - \tau(q)\big).
$$

Because the energy is a sum of independent per-token terms, subset membership factorizes
into **independent** Bernoulli variables, and the exact marginal is a sigmoid:

$$
\boxed{\; g_i \;=\; \sigma\!\big(\beta\,(a_i - \tau(q))\big). \;}
$$

- $\tau(q)$ — a learned **per-query threshold**; $\beta$ — a learned **sharpness**.
- $g_i$ depends **only on token $i$'s own score** $a_i$. Relevant tokens ($a_i > \tau$) get
  $g_i \to 1$ (kept); irrelevant ones get $g_i \to 0$ (dropped).
- This is **adaptive sparsity** — a soft, learned top-$k$. It is essentially the
  entmax / sparsemax family of sparse attention.
- $\beta = 0 \Rightarrow g_i = \tfrac12 \Rightarrow$ exact softmax (so it starts at parity
  with `mha` and departs only if that lowers the loss).

**Code** (`gated_attention.py`): `gate = sigmoid(beta * (a - tau))`, then
`w = exp(a) * gate / sum`.

**Limitation:** since $g_i$ uses only $a_i$, two tokens with the same relevance get the same
gate — *regardless of how many near-duplicates exist.* Remember this.

---

## 3. `gated_rep` — a *non-modular* $F_2$ (gate depends on the other tokens)

Add a **pairwise repulsion** term — the anti-redundancy / DPP-style part:

$$
F_2(S) \;=\; \sum_{i \in S} a_i \;-\; \lambda \sum_{i < j \,\in\, S} \frac{\langle k_i, k_j\rangle}{\sqrt d}.
$$

The new term **subtracts a cost for every pair of *similar* selected tokens**
($\langle k_i, k_j\rangle$ large $\Rightarrow$ similar keys). Its mean-field marginal is:

$$
\boxed{\;
g_i \;=\; \sigma\!\Big(\beta\big(a_i - \tau(q) - \lambda\, r_i\big)\Big),
\qquad
r_i \;=\; \sum_{j} g_j \, \frac{\langle k_i, k_j\rangle}{\sqrt d}.
\;}
$$

- $r_i$ measures how **redundant** token $i$ is with *all the other attended tokens*
  (each weighted by its own gate $g_j$).
- If token $i$ is redundant with many attended tokens, $r_i$ is large $\Rightarrow$ $g_i$ is
  pushed **down** $\Rightarrow$ the token is suppressed.
- $\lambda \ge 0$ (learned) is the repulsion strength. **$\lambda = 0$ recovers `gated`
  exactly**, so `gated_rep` is a strict superset.
- This is **non-modular**: $g_i$ now couples to the entire set through $r_i$. That coupling
  is precisely what ordinary softmax and the per-token gate **cannot** express.

**Code** (`gated_attention.py`): `r = g0 @ (K Kᵀ / √d)` (the `matmul(g0, kk)` line), then
`gate_logit = (a − τ) − λ·r`.

---

## 4. What "redundancy" means, and why only `gated_rep` handles it

**Redundancy** = tokens whose keys are similar or duplicated: $\langle k_i, k_j\rangle$
large. In the experiment, the decoy class is **one vector copied $m$ times**, i.e. $m$
tokens with *identical* keys — maximal redundancy.

Here is the exact reason a per-token gate fails but repulsion succeeds. Suppose:

- $n$ **signal** tokens (true class $y$), *distinct*, each with score $a$;
- $m$ **decoy** tokens (wrong class $y'$), *identical copies*, each also with score $a$
  (equally relevant per token).

### Softmax and `gated`: mass scales with count

Each token gets the same weight $g\,e^{a}$ (same score $a$, same gate $g$), so the **total**
attention mass on each class scales with its **count**:

$$
\text{mass}(y) \propto n\, g\, e^{a},
\qquad
\text{mass}(y') \propto m\, g\, e^{a},
\qquad
\frac{\text{mass}(y')}{\text{mass}(y)} = \frac{m}{n}.
$$

If $m > n$ (a big redundant clique), the decoy **dominates the output** $\Rightarrow$ the
model predicts $y'$ (wrong). The per-token gate cannot fix this: it multiplies *every* token
by the same $g$, leaving the ratio $m/n$ unchanged. **Counting duplicates is impossible with
per-token scoring.**

### `gated_rep`: repulsion caps the clique

Each identical decoy is similar to all $m-1$ of its clones, so its redundancy term is large:

$$
r_i \;\approx\; g \cdot m \cdot \frac{\lVert k \rVert^2}{\sqrt d}
\qquad\Longrightarrow\qquad
g_{\text{decoy}} \;=\; \sigma\!\Big(\beta\big(a - \tau - \lambda\, m\, \tfrac{\lVert k\rVert^2}{\sqrt d}\big)\Big)
\;\xrightarrow[\;m \text{ large}\;]{}\; 0 .
$$

The whole clique of $m$ copies is squeezed down to roughly **one token's worth of mass**.
The signal tokens are *distinct* ($\langle k_i, k_j\rangle$ small $\Rightarrow$ $r_i$ small),
so they are kept. The ratio flips:

$$
\frac{\text{mass}(y')}{\text{mass}(y)} \;\approx\; \frac{1}{n}
\qquad\Longrightarrow\qquad
\text{signal wins} \;\Longrightarrow\; \text{prediction } y \ \text{(correct)}.
$$

This matches the measurements: `gated ≈ mha` (cannot de-duplicate), while `gated_rep` cuts
eval cross-entropy 2–5×, and the learned $\lambda$ stayed $\approx 1.1$ (the model *chose* to
use repulsion).

---

## 5. The three rungs, side by side

| method | gate $g_i$ | extra capability |
|---|---|---|
| `mha` | $g_i = 1$  (⇒ plain softmax) | weight by relevance only |
| `gated` | $\sigma\!\big(\beta(a_i - \tau)\big)$ — **per-token** | + drop low-relevance tokens (sparsity) |
| `gated_rep` | $\sigma\!\big(\beta(a_i - \tau - \lambda\, r_i)\big)$ — **set-coupled** | + suppress tokens **redundant with the others** |

Ordering by expressive power: `mha` $\subset$ `gated` $\subset$ `gated_rep`
(each recovers the previous at $\beta{=}0$ or $\lambda{=}0$).

The jump `gated` → `gated_rep` — adding the $\lambda\, r_i$ coupling — is the **only** part
that is *not* expressible as "reweight each token on its own." That coupling is the
distinctive "general attention" mechanism, and it is the part that wins on redundancy.

---

## 6. Which mechanism wins where (empirical summary)

| regime | winning lever | is it just gating? |
|---|---|---|
| CIFAR-10 (L=65, natural images) | neither (both idle) | — → **parity** |
| needle / long-context dilution | modular gate `gated` (adaptive sparsity) | **yes** — known entmax family |
| redundancy pathology | **`gated_rep` repulsion** (non-modular $F_2$) | **no** — the distinctive part |

Each lever earns its keep only in the regime whose pathology it targets, at equal parameter
count ($+O(d)$ gate params, ~0.1–0.2%).
