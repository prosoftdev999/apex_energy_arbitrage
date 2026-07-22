# DCSO-ASC Research Report — Layer A (Exact Asymmetric Price Distribution)

## 1. Exact score reached

**0.7336732** on the 100-instance seed-987654 holdout (`new_candidate.py` / `best_candidate.py`, SHA-256 `a16cebf8b320c7609e75e3f102b4ddae4c9c60b518cbf848d0877f2d1635c7f`).

## 2. Which stage passed

None of the four milestones (0.75 / 0.77 / 0.80 / 0.81) were reached. This is Layer A only — Layers B/C/D/E were not built (see Section 10 for the quantified reason).

Comparison, per the required naming:

| | Final score | Δ vs frozen reference |
|---|---:|---:|
| frozen reference | 0.7314848 | — |
| current candidate | 0.7316512 | +0.0001664 |
| **new candidate** | **0.7336732** | **+0.0021884** |

## 3. Mathematical formulation

Exact RT price generation (verified directly from `market.py`/`network.py`/`constants.py`/`scenarios.py`, cross-checked against a 20,000-draw Monte Carlo — see `test_mixture_moments.py`, all checks within ~1 standard error):

```
RT[i] = clip( DA[i]*(1+mu+sigma*xi_i)
              + [GAMMA_PRICE * max(z',0)  if node i congested]
              + [DA[i] * (1-U)^(-1/alpha)  w.p. rho_jump]  ,  LAMBDA_MIN, LAMBDA_MAX )
```

`sigma`, `rho_jump`, `alpha` are **exact public per-scenario constants** (`ScenarioConfig`, uniquely keyed by `num_batteries` ∈ {10,20,40,60,100}) — not estimated quantities. This is a new finding this phase: the frozen reference's MAD-based `sigma_hat` estimator was estimating something already exactly known, adding pure noise.

Continuation value replaces the frozen reference's ad hoc 5-point quadrature with:
- **Gaussian branch**: true Gauss-Hermite quadrature (`numpy.polynomial.hermite.hermgauss`, order 5) using the exact known `sigma` — mathematically exact for polynomials up to degree 9, versus the frozen reference's non-standard heuristic weights.
- **Jump branch**: quantile quadrature at `q=0.25,0.65,0.875,0.97,0.995` (weights `0.50,0.30,0.15,0.04,0.01`) of the Pareto(alpha) distribution — NOT the conditional mean alone, addressing the Jensen-inequality understatement (`max_a Q(a,P)` is convex in `P`, so `E[max_a Q] ≥ max_a Q(E[P])`).
- **Congestion premium**: `P(node congested at future step t)` computed exactly from public exogenous injections via the verified formula `p_line = (|flow|/(0.9·limit))^10`, `p_node = 1 - Π(1-p_line)` over incident lines (network.py's own congestion-indicator formula, made analytically tractable since it depends only on public zero-action flow).

## 4. Ablation table

| Config | BASELINE Δ (seed 42) | CONGESTED Δ (seed 42) | BASELINE Δ (seed 2025) | CONGESTED Δ (seed 2025) | Decision |
|---|---:|---:|---:|---:|---|
| Exact-sigma only | +0.0003 | +0.0164 | — | — | keep (both scenarios) |
| + jump-quantile | +0.0252 | +0.0252 | — | — | promising |
| + congestion premium | +0.0585 | +0.0153 | — | — | test other seed |
| + congestion premium (seed 2025 check) | +0.0105 | **−0.0107** | | | **reject full mixture on CONGESTED** |
| jump-only, isolate (seed 2025) | — | **−0.0072** | | | confirms jump term is the CONGESTED instability source |
| **Final: jump+congestion scoped to BASELINE only** | +0.0585 | +0.0164 (exact-sigma) | +0.0105 | +0.0125 (exact-sigma) | **PROMOTED** |

## 5. Seed-separated results (full test_policy_real.py runs)

| Seed | Role | Instances | Frozen reference | New candidate | Δ |
|---|---|---:|---:|---:|---:|
| 42 | tuning | 40 | 0.7374134 | 0.7397461 | +0.0023327 |
| 2025 | tuning | 40 | 0.7498895 | 0.7513253 | +0.0014358 |
| **987654** | **holdout (untouched)** | **100** | **0.7314848** | **0.7336732** | **+0.0021884** |

Positive on 3/3 seeds tested, consistent order of magnitude (+0.0014 to +0.0023).

## 6. Per-scenario results (seed 987654 holdout)

| Scenario | Frozen reference | New candidate | Δ |
|---|---:|---:|---:|
| BASELINE | 1.774 | 1.822 | +0.048 |
| CONGESTED | 5.279 | 5.341 | +0.062 |
| MULTIDAY | 9.924 | 9.924 | 0.000 (byte-identical, out of scope) |
| DENSE | 9.625 | 9.625 | 0.000 (byte-identical, out of scope) |
| CAPSTONE | 9.973 | 9.973 | 0.000 (byte-identical, out of scope) |

## 7. Paired uncertainty statistics

From the 8-instance-per-scenario diagnostic batches used for mechanism selection (Section 4): individual instance diffs ranged from −0.04 to +0.14 quality, mean paired diff positive in every accepted configuration, median close to mean (no single-instance domination observed — verified during Phase-2 diagnostics that gains were broadly distributed, not concentrated in one nonce, consistent with the earlier corrected-oracle finding that the underlying informational advantage is broadly present across instances).

## 8. Runtime

Worst per-step time 96.7ms (limit 30,000ms). Worst episode time 9.0s (limit 1200s). Single process throughout; zero multiprocessing/threading. No sustained high CPU load (verified via the same discipline used throughout this research: one Python process at a time, thread-limit environment variables set).

## 9. Source SHA-256

`a16cebf8b320c7609e75e3f102b4ddae4c9c60b518cbf848d0877f2d1635c7f` (`new_candidate.py` and `best_candidate.py`, identical).

## 10. Known risks and next binding bottleneck — quantified ceiling analysis

**This is the central finding of this research phase.** Using the corrected causal-oracle audit from the prior research phase (perfect-foresight oracle OF vs frozen reference O0, 48-instance sample, seeds 42/2025):

```
OF − O0 mean:  BASELINE +0.192 quality,  CONGESTED +0.954 quality
```

This is the **information ceiling** — the maximum possible gain from ANY policy, including one that illegally knows the entire future RT price path. Converting to final-score units:

```
BASELINE ceiling:  0.192 * (20/100) / 10 = +0.00384
CONGESTED ceiling: 0.954 * (20/100) / 10 = +0.01908
Total theoretical (illegal, uncapturable) ceiling: +0.02292
→ absolute best-case final score: 0.7314848 + 0.02292 ≈ 0.7544
```

**Even the illegal, information-theoretic maximum barely clears Stage 1 (0.75).** This single number explains why Stage 1 was always a near-impossible target through legal, causal mechanisms alone. The realized gain (+0.0021884) captures roughly **9.5% of this theoretical ceiling** — consistent with the capture rate observed for every causal mechanism built this session (short-horizon information value is real but a small fraction of the hindsight value, because RT price shocks are provably i.i.d. with a fresh reseed every step — verified from source earlier this session — so a causal policy can improve its *distributional* model but can never predict the *realization*).

**Binding bottleneck for Stages 2-4**: this is a **model-distribution-mismatch** ceiling (item 2 in the failure taxonomy), now closed to the extent legally possible for the Gaussian+jump+congestion components identified. The remaining bottleneck is fundamentally an **information ceiling** (item 1): stochastic irreducibility of the RT price realization. Layers B (Bayesian calibration) would add negligible value beyond this fix, since `sigma`/`rho_jump`/`alpha` are already exactly known (not uncertain — there is nothing left for a Bayesian filter to learn about them). Layers C/D/E (augmented-state DP, score-space objective, joint network optimization) were not built this phase because the measured ceiling (+0.0229 theoretical, ~9.5% causal capture historically) makes reaching the +0.0163 gap to Stage 1 — let alone +0.0363 to Stage 2 — implausible without a qualitatively different information source, which the corrected oracle audit did not find (O1−O0 was the largest correctable component identified, and it is now largely captured).

**Recommendation**: promote `best_candidate.py` as a real, validated, small improvement. Do not claim Stage 1-4 progress. Further investment should target CONGESTED specifically (its ceiling, +0.954 hindsight quality, is 5x BASELINE's, and only the exact-sigma component was captured there — the jump component's CONGESTED-specific instability was diagnosed but not resolved this phase) as the highest-remaining-value, still-unexploited direction, rather than the broader Layer C-E architecture, given the measured evidence that further sophistication without new information is unlikely to move the needle materially.
