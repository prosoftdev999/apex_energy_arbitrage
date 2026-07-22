# Exact RT price mixture (verified from source: market.py, network.py, constants.py, scenarios.py)

## Exact generative formula (market.py::generate_rt_prices, per time step t, per node i)

```
z_common ~ N(0,1)                       [drawn ONCE per step, shared across all nodes]
z_prime  ~ N(0,1)                       [drawn ONCE per step, shared across all nodes]
zeta = max(z_prime, 0)                  [half-normal, shared across all nodes THIS step]

for each node i:
    eps_i ~ N(0,1)                      [i.i.d. per node]
    xi_i = sqrt(rho)*z_common + sqrt(1-rho)*eps_i        [rho = RHO_SPATIAL = 0.70]
    price = DA[i] * (1 + mu + sigma*xi_i)                [mu = MU_BIAS = 0.0]

    if congestion_indicators[i]:
        price += GAMMA_PRICE * zeta                       [GAMMA_PRICE = 20.0]

    u_jump ~ Uniform(0,1)               [i.i.d. per node]
    if u_jump < jump_probability:
        u_pareto ~ Uniform(0,1)         [i.i.d. per node]
        pareto = (1 - u_pareto)^(-1/alpha)                [support [1, inf)]
        price += DA[i] * pareto                            [ADDITIVE, scaled by DA[i]]

    RT[i] = clip(price, LAMBDA_MIN=-200, LAMBDA_MAX=5000)
```

**Confirmed exactly** (this resolves the user's explicit question about parameterization): the Pareto term is an **additive, DA-scaled** shock, `jump = DA[i] * (1-U)^(-1/alpha)`, support `[DA[i], ∞)` — i.e., when a jump occurs, price gains AT LEAST a full DA-price's worth of extra revenue. It is emphatically NOT a multiplicative-only rescale of the Gaussian branch, and NOT applied to the final clipped price. Jumps are drawn **independently per node**, not as a single market-wide event.

## Exact per-scenario public constants (scenarios.py, keyed uniquely by num_batteries)

| Scenario | num_batteries | sigma (volatility) | rho_jump | alpha (tail index) |
|---|---:|---:|---:|---:|
| BASELINE | 10 | 0.10 | 0.01 | 4.0 |
| CONGESTED | 20 | 0.15 | 0.02 | 3.5 |
| MULTIDAY | 40 | 0.20 | 0.03 | 3.0 |
| DENSE | 60 | 0.25 | 0.04 | 2.7 |
| CAPSTONE | 100 | 0.30 | 0.05 | 2.5 |

**Verified by direct trace**: `Challenge.generate_instance` builds `MarketParams(volatility=config.sigma, jump_probability=config.rho_jump, tail_index=config.alpha)` directly from `ScenarioConfig` — these are FIXED, KNOWN, PUBLIC per-scenario constants, not instance-random quantities. **This means `sigma_hat`'s single-sample MAD estimator (used in the frozen reference policy) is estimating a quantity that is already exactly known with zero uncertainty** — the estimator can only add noise relative to using the table above directly. This is a new, previously unexploited finding from this research phase.

## Exact congestion-indicator formula (network.py::generate_congestion_indicators)

```
for each line l (public PTDF, public flow_limits, public exogenous_injections[t]):
    flow_l = compute_flows(exogenous_injections[t])[l]        [PUBLIC -- zero-battery-action flow]
    p_l = (|flow_l| / (TAU_CONG * flow_limit_l))^10             [TAU_CONG = 0.90]
    if p_l > Uniform(0,1):        [Bernoulli(p_l), one draw per line]
        congestion_indicators[from_node] = True
        congestion_indicators[to_node]   = True
```

`p_l` is **exactly computable from public information alone** (PTDF, flow limits, and the exogenous-injection schedule, all known at episode start) for every future line and step. The only randomness is a per-line Bernoulli draw with a KNOWN probability. Therefore:

```
P(node i congested at step t) = 1 - Π_{l incident to i} (1 - p_l(t))
```

is exactly computable, causally, for any future step — this was previously only approximated heuristically (`policy_v13`'s clipped-utilization proxy), never computed via the TRUE Bernoulli-probability formula.

## Full conditional mixture for E[RT[i] | public info at decision time]

```
E[RT[i]] ≈ DA[i]*(1+mu)
          + P(congested_i) * GAMMA_PRICE * E[max(z',0)]      [E[max(z',0)] = 1/sqrt(2*pi) ≈ 0.39894, half-normal mean]
          + rho_jump * DA[i] * alpha/(alpha-1)                [E[Pareto] = alpha/(alpha-1), valid for alpha>1]
```

(pre-clipping; clipping to [-200,5000] is only ever binding in extreme tail scenarios given MEAN_DA_PRICE=50, DA_AMPLITUDE=20 -- negligible correction, verified not to matter at the scales observed this session.)

## Deterministic quadrature design for the Gaussian branch

Replacing the frozen reference's ad hoc `_QUAD_Z=[-2,-1,0,1,2]` / `wq ∝ exp(-z²/2)` (a crude un-normalized-density heckuristic, not a true Gauss-Hermite rule) with a **true Gauss-Hermite quadrature** for `E[g(xi)]` where `xi ~ N(0,1)`:

```
E[g(xi)] ≈ Σ_k  w_k * g(sqrt(2)*x_k)  / sqrt(pi)
```
where `(x_k, w_k)` are the standard Gauss-Hermite nodes/weights (`numpy.polynomial.hermite.hermgauss`). This is EXACT for `g` polynomial up to degree `2n-1` for an n-point rule -- a strictly better approximation basis than the frozen reference's heuristic weights, at the same node count.

## Deterministic quadrature design for the jump branch (addressing "conditional mean ignores nonlinear option value")

The Pareto branch's contribution to `E[max_a Q(a, P)]` is NOT well-approximated by evaluating at the Pareto MEAN alone, because `max_a Q(a,P)` is CONVEX in P (max of affine functions of P) -- by Jensen, `E[max_a Q(a,Pareto)] ≥ max_a Q(a, E[Pareto])`, i.e. using only the mean point systematically UNDERSTATES the branch's true option value, exactly as the user's brief anticipates. Fix: represent the jump branch via **quantile quadrature** at `q = 0.50, 0.80, 0.95, 0.99` of the Pareto(alpha) distribution, i.e. `pareto_q = (1-q)^(-1/alpha)`, each carrying probability mass `0.30, 0.15, 0.04, 0.01` respectively covering the CDF from 0.35 to 1.0 (with the first mass point at q=0.35 as a representative for the [0, 0.65) bulk) -- **implemented and tested in Phase 2 below**, replacing the single-point mean approximation.
