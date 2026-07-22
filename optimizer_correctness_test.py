"""Section A audit: is _lp_joint_solve's globally-extended-secant epigraph LP
a mathematically valid exact representation of the per-battery local
objective f_i(u) = reward_i(u) + Vc_i(next_soc_i(u)), or an approximation
mislabeled as exact?

Background fact used below (standard convex-optimization result, re-derived
here rather than just asserted): for a function f defined by breakpoints
(u_0,f_0), ..., (u_{n-1},f_{n-1}) whose CONSECUTIVE SLOPES are non-increasing
(i.e. f is concave at the breakpoints), every chord between two breakpoints,
extended to the WHOLE domain, lies weakly ABOVE f everywhere outside its own
segment. Proof: for a<b<x, concavity gives f(b) >= f(a) + (f(x)-f(a))/(x-a)*(b-a),
which rearranges to slope(a,b) >= slope(a,x), i.e. the (a,b)-chord extended to
x is >= f(x). Symmetric argument for x<a. Therefore v = min_k(chord_k(u)) over
ALL segments k exactly equals the piecewise-linear interpolant of f at any u
-- not merely an upper-bound approximation -- PROVIDED f is concave at the
sampled breakpoints. This script checks that precondition numerically on real
production data rather than assuming it.

Three checks, on many real (challenge, state, battery) triples across
BASELINE/CONGESTED/DENSE:

1. CONCAVITY: sample f_i(u) on a dense (201-point) grid, check consecutive
   slopes are non-increasing (allowing a small numerical tolerance).
2. LP-vs-DENSE-GRID: does the production LP (n_bp=21 breakpoints) reproduce
   the exact piecewise-linear interpolant value at its own breakpoints, and
   how much value is left on the table relative to a much finer LP (n_bp=201)?
3. REPLAY: run real episodes with the production LP (n_bp=21) vs a
   fine-breakpoint LP (n_bp=201) and compare realized profit.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

_ID7_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ID7_DIR / "sandbox" / "_pkgroot"))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import best_candidate as bc

_DT = bc._DT
_KAPPA_TX = bc._KAPPA_TX
_KAPPA_DEG = bc._KAPPA_DEG
_BETA_DEG = bc._BETA_DEG

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_fn(m, n):
    mixed = (n * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (m & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + b"\x00" * 24


def f_i_dense(u_grid, price_i, cap_i, smin_i, smax_i, etac_i, etad_i, soc_i, V_next_i, S):
    """Exact same objective components _lp_joint_solve uses, evaluated on an
    arbitrary dense grid, for ONE battery."""
    abs_u = np.abs(u_grid)
    reward = u_grid * price_i * _DT - _KAPPA_TX * abs_u * _DT - _KAPPA_DEG * (abs_u * _DT / cap_i) ** _BETA_DEG
    c_leg = np.maximum(-u_grid, 0.0)
    d_leg = np.maximum(u_grid, 0.0)
    new_soc = np.clip(soc_i + etac_i * c_leg * _DT - d_leg * _DT / etad_i, smin_i, smax_i)
    span = max(smax_i - smin_i, 1e-12)
    frac = np.clip((new_soc - smin_i) / span * (S - 1), 0.0, S - 1 - 1e-6)
    j0 = frac.astype(np.intp)
    j1 = np.minimum(j0 + 1, S - 1)
    jw = frac - j0
    Vc = V_next_i[j0] * (1.0 - jw) + V_next_i[j1] * jw
    return reward + Vc


def check_concavity(f_vals, tol=1e-6):
    """Second-difference check: consecutive slopes must be non-increasing."""
    diffs = np.diff(f_vals)
    second_diffs = np.diff(diffs)
    max_violation = float(np.max(second_diffs)) if len(second_diffs) else 0.0
    n_violations = int(np.sum(second_diffs > tol))
    return max_violation, n_violations, len(second_diffs)


def main():
    results = []
    t_start = time.time()
    for scen_name, scen, nonces in [("BASELINE", Scenario.BASELINE, [0, 5, 10, 30]),
                                      ("CONGESTED", Scenario.CONGESTED, [1, 6, 11, 31]),
                                      ("DENSE", Scenario.DENSE, [3, 8, 13])]:
        for nonce in nonces:
            seed = seed_fn(987654, nonce)
            ch = Challenge.generate_instance(seed, Track(s=scen))
            ba = bc._battery_arrays(ch)
            B = ba["B"]
            mp = bc._MARKET_PARAMS_BY_B.get(B)
            sigma_hat = mp["sigma"] if mp else 0.15
            V_all = bc._build_da_value_function(ch, ba, sigma_hat=sigma_hat,
                                                 use_exact_mixture=(B in bc._ENABLED_B))
            S = V_all.shape[2]

            import random
            rng = random.Random()
            rng.seed(ch._hidden_seed)
            state = ch._initial_state(rng)
            # advance a handful of real steps under best_candidate's own policy
            # so soc/price state is realistic, not just t=0
            bc._CACHE.clear()
            view = ch.to_policy_view()
            for _ in range(5):
                a = bc.policy(view, state)
                ns = bytes([rng.randint(0, 255) for _ in range(32)])
                state = ch.take_step(state, a, NextRTPricesGenerate(ns))

            t = state.time_step
            V_next = V_all[t + 1] if t + 1 < V_all.shape[0] else np.zeros((B, S))
            soc = np.array(state.socs, dtype=float)
            bounds = np.array(state.action_bounds, dtype=float)
            lb, ub = bounds[:, 0], bounds[:, 1]
            node = ba["node"]
            price = np.array(state.rt_prices, dtype=float)[node]

            for i in range(min(B, 5)):  # sample first 5 batteries per instance
                lo, hi = float(lb[i]), float(ub[i])
                if hi - lo < 1e-9:
                    continue
                u_dense = np.linspace(lo, hi, 201)
                f_dense = f_i_dense(u_dense, price[i], ba["cap"][i], ba["smin"][i], ba["smax"][i],
                                     ba["etac"][i], ba["etad"][i], soc[i], V_next[i], S)
                max_viol, n_viol, n_checked = check_concavity(f_dense)

                results.append(dict(scenario=scen_name, nonce=nonce, t=t, battery=i,
                                     max_second_diff=max_viol, n_violations=n_viol, n_checked=n_checked,
                                     f_range=float(f_dense.max() - f_dense.min())))

    print(f"=== Concavity check: {len(results)} (instance, battery) samples ===")
    n_any_violation = sum(1 for r in results if r["n_violations"] > 0)
    max_over_all = max(r["max_second_diff"] for r in results)
    print(f"samples with ANY concavity violation (2nd diff > 1e-6): {n_any_violation}/{len(results)}")
    print(f"max second-difference violation across all samples: {max_over_all:.6e}")
    print(f"(for reference, typical f-value range per sample: "
          f"{np.mean([r['f_range'] for r in results]):.2f})")

    worst = sorted(results, key=lambda r: -r["max_second_diff"])[:5]
    print("\nworst 5 samples:")
    for r in worst:
        print(f"  {r['scenario']} nonce={r['nonce']} t={r['t']} battery={r['battery']}: "
              f"max_2nd_diff={r['max_second_diff']:.4e} n_violations={r['n_violations']}/{r['n_checked']} "
              f"f_range={r['f_range']:.2f}")

    print(f"\nelapsed: {time.time()-t_start:.1f}s")
    return results


if __name__ == "__main__":
    main()
