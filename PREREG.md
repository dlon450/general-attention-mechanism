# Pre-registration — consensus-under-adversarial-duplication mechanism study

Registered before scoring the frozen test set. Task, models, and protocol are fixed in
`task_consensus.py`, `consensus_models.py`, `bench_full.py` at the committing git hash.

## Primary metric (single, headline)
Balanced test accuracy on the FROZEN test set (generator seed 777) of the consensus task at the
**headline cell α=1, γ=0.8**, chance = 50%. Frozen test is never used for any selection; validation
(separate seeds) selects the checkpoint step.

## Arms (matched: shared k=1 self-attention encoder + swappable pooling head; params within ~0.2%)
- Content-only (see only V,S): `softmax` (regular attention), `set_transformer` (ISAB+PMA).
- Provenance (fed the SAME noisy graph): `prov_concat` (degree feature concat), `relation_bias`
  (graph as attention bias).
- Ours: `m2_prov` (rigid density-prior gate), `m2_prov_x` (MLP gate), `m2_prov_r` (rigid + zero-init
  residual — the primary mechanism arm).

## Hypotheses (fixed before test contact)
- **H1 (no-free-lunch):** both content-only arms sit at chance (≤ 55%) at α=1 — regular attention
  cannot beat chance when content/count are swap-symmetric.
- **H2 (Pareto-dominance, PRIMARY):** `m2_prov_r` ≥ best provenance baseline (`prov_concat`) at every
  training-set size, with the paired-bootstrap 95% CI on Δ(m2_prov_r − prov_concat) **> 0** in the
  low-data regime (n ≤ 800) and **≥ 0 (CI includes 0 is acceptable = tie)** at the ceiling.
- **H3 (robustness):** `m2_prov_r` worst-case-over-α ≥ every provenance baseline's worst-case-over-α;
  no weaponization (never below `softmax`).
- **H4 (OOD):** report train-n_orig∈[3,6] → test-n_orig∈[8,10] extrapolation; success = m2_prov_r OOD
  drop ≤ baselines' (not required for the headline claim; exploratory).

## Success criterion for "beats regular attention (honestly)"
H1 holds AND H2 holds (paired CI > 0 at n ≤ 800, ≥ 0 at ceiling) AND latency overhead reported.

## Kill / narrowing criteria (stated in advance)
- If a content-only arm exceeds 55% at α=1 → the task leaks; re-open non-gameability (do NOT proceed).
- If `prov_concat` ≥ `m2_prov_r` at low data (CI) → no sample-efficiency win; report the honest negative.
- If `m2_prov_r` drops below `softmax` at any α → robustness failure; report it.

## Seeds & statistics
≥10 seeds; paired Δ per seed (same data-gen + init seed index); paired-bootstrap 95% CI (10k
resamples). Frozen test scored once per (arm, seed). Non-gameability asserted by
`tests/test_nongameability.py` (cheap content/count baselines ≈ chance at α=1).
