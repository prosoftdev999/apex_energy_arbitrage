"""Diagnostic: does policy_v11's per-step joint LP (_lp_joint_solve) have
alternative optimal solutions (degenerate optima)? If ties exist, is there
room to break them in favor of future battery value using a second-stage
lexicographic LP (stage 1: maximize reward+Vc; stage 2, restricted to the
stage-1-optimal objective value, maximize a distinct secondary criterion)?

Method for detecting degeneracy (the only way that actually proves it,
rather than just guessing from reduced costs which HiGHS's revised-simplex
output does not reliably expose across degenerate bases): re-solve the
IDENTICAL LP (same A_ub/b_ub/bounds) with objective replaced by the ORIGINAL
objective plus a small random linear perturbation, and separately with a
principled secondary objective (sum of marginal SOC value / costate at the
resulting state). If the action vector changes while sum(v_i) stays within
solver tolerance of the original optimum, that step had multiple optima.

This monkeypatches policy_v11._lp_joint_solve so it runs against REAL
episodes (production challenge/state), recording diagnostics as a side
effect, with zero duplication of policy logic besides the LP matrix build
(unavoidably copied here to add the second-stage solve).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch
import policy_v11 as v11

_DT = v11._DT
_KAPPA_TX = v11._KAPPA_TX
_KAPPA_DEG = v11._KAPPA_DEG
_BETA_DEG = v11._BETA_DEG
_LP_BREAKPOINTS = v11._LP_BREAKPOINTS

STATS = {
    "n_steps": 0,
    "n_degenerate_random": 0,
    "n_degenerate_costate": 0,
    "n_batteries_changed_random": 0,
    "n_batteries_changed_costate": 0,
    "n_batteries_total": 0,
    "costate_gain_sum": 0.0,   # sum of secondary-objective improvement when degenerate
    "z_gap_max": 0.0,
    "costate_nonzero_count": 0,
    "costate_abs_sum": 0.0,
}
TOL_Z = 1e-6   # relative tolerance for "same optimum"


def _build_lp_matrices(soc, price, cap, smin, smax, etac, etad, lb, ub, V_next, S,
                        node, ptdf_np, limits_np, exo, slack, n_bp=_LP_BREAKPOINTS):
    B = len(soc)
    L = ptdf_np.shape[0]
    nvar = 2 * B
    bounds = [(float(lb[i]), float(ub[i])) for i in range(B)] + [(None, None)] * B
    c_obj = np.zeros(nvar)
    c_obj[B:] = -1.0

    A_ub_rows = []
    b_ub = []
    # also record, per battery, the SOC value each breakpoint maps to, so we
    # can build a costate (dVc/dSOC) secondary objective after stage 1 solves.
    bp_new_soc = []
    for i in range(B):
        lo, hi = float(lb[i]), float(ub[i])
        us = np.linspace(lo, hi, n_bp)
        abs_us = np.abs(us)
        reward = us * price[i] * _DT - _KAPPA_TX * abs_us * _DT - _KAPPA_DEG * (abs_us * _DT / cap[i]) ** _BETA_DEG
        c_leg = np.maximum(-us, 0.0)
        d_leg = np.maximum(us, 0.0)
        new_soc = np.clip(soc[i] + etac[i] * c_leg * _DT - d_leg * _DT / etad[i], smin[i], smax[i])
        bp_new_soc.append(new_soc)
        span_i = max(smax[i] - smin[i], 1e-12)
        frac = np.clip((new_soc - smin[i]) / span_i * (S - 1), 0.0, S - 1 - 1e-06)
        j0 = frac.astype(np.intp)
        j1 = np.minimum(j0 + 1, S - 1)
        jw = frac - j0
        Vc = V_next[i, j0] * (1.0 - jw) + V_next[i, j1] * jw
        fs = reward + Vc

        added = False
        for k in range(n_bp - 1):
            du = us[k + 1] - us[k]
            if abs(du) < 1e-12:
                continue
            m = (fs[k + 1] - fs[k]) / du
            b_k = fs[k] - m * us[k]
            row = np.zeros(nvar)
            row[B + i] = 1.0
            row[i] = -m
            A_ub_rows.append(row)
            b_ub.append(b_k)
            added = True
        if not added:
            row = np.zeros(nvar)
            row[B + i] = 1.0
            A_ub_rows.append(row)
            b_ub.append(float(fs[0]))

    base_flow = v11._flows_np(exo, np.zeros(B), node, slack, ptdf_np)
    n_epi_rows = len(A_ub_rows)
    for l in range(L):
        row_hi = np.zeros(nvar)
        row_lo = np.zeros(nvar)
        for i in range(B):
            row_hi[i] = ptdf_np[l, node[i]]
            row_lo[i] = -ptdf_np[l, node[i]]
        A_ub_rows.append(row_hi)
        b_ub.append(float(limits_np[l] - base_flow[l]))
        A_ub_rows.append(row_lo)
        b_ub.append(float(limits_np[l] + base_flow[l]))

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)
    return A_ub, b_ub, bounds, c_obj, bp_new_soc


def _costate(V_next, soc_val, smin, smax, S, i):
    """dVc/dSOC at soc_val for battery i, from the SAME piecewise-linear grid
    the primary LP already uses -- a distinct (finer-grained, per-battery
    marginal) criterion, not a new forecast or value model."""
    span = max(smax[i] - smin[i], 1e-12)
    frac = np.clip((soc_val - smin[i]) / span * (S - 1), 0.0, S - 1 - 1e-06)
    j0 = int(frac)
    j1 = min(j0 + 1, S - 1)
    if j1 == j0:
        return 0.0
    dV = V_next[i, j1] - V_next[i, j0]
    dsoc = span / (S - 1)
    return float(dV / dsoc)


_RNG = np.random.RandomState(20260713)


def diag_lp_joint_solve(soc, price, cap, smin, smax, etac, etad, lb, ub, V_next, S,
                         node, ptdf_np, limits_np, exo, slack, n_bp=_LP_BREAKPOINTS):
    """Correct degeneracy test: perturb the objective by an infinitesimal
    secondary term (lexicographic weighted-sum form, same feasible region,
    no extra constraint row) and check whether the argmax moves. By LP
    sensitivity theory, a sufficiently small epsilon cannot change which
    basis is optimal UNLESS the unperturbed problem already had multiple
    optimal bases (a tied/degenerate face) -- this avoids the earlier bug
    where constraining only sum(v_i) let individual v_i's float below their
    true chord value while another battery compensated."""
    B = len(soc)
    A_ub, b_ub, bounds, c_obj, bp_new_soc = _build_lp_matrices(
        soc, price, cap, smin, smax, etac, etad, lb, ub, V_next, S, node, ptdf_np, limits_np, exo, slack, n_bp)

    res1 = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res1.success:
        return None
    u1 = res1.x[:B]
    z1 = -res1.fun  # sum(v_i) at stage-1 optimum

    STATS["n_steps"] += 1
    STATS["n_batteries_total"] += B

    eps = 1e-6  # relative to typical |c_obj|=1 per battery; small enough not
    # to change a non-degenerate optimal basis, large enough for HiGHS to
    # resolve above its own internal tolerance.

    # --- Probe A: random linear perturbation on u_i, same feasible region ---
    c_rand = c_obj.copy()
    c_rand[:B] += _RNG.uniform(-1.0, 1.0, size=B) * eps
    res_rand = linprog(c_rand, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res_rand.success:
        u_rand = res_rand.x[:B]
        z_rand = -(c_obj @ res_rand.x)
        same_z = abs(z_rand - z1) <= max(1e-6, 1e-6 * abs(z1))
        changed = np.abs(u_rand - u1) > 1e-6
        if same_z and np.any(changed):
            STATS["n_degenerate_random"] += 1
            STATS["n_batteries_changed_random"] += int(np.sum(changed))

    # --- Probe B: principled secondary objective = maximize sum of costate
    # (dVc/dSOC) at the resulting SOC -- among stage-1-optimal actions,
    # prefer ones that push SOC further in the direction the (already
    # validated) continuation value says is good. ---
    c_costate = np.zeros(2 * B)
    us_grid_cache = {}
    for i in range(B):
        if u1[i] < 0:  # charging
            dsoc_du = etac[i] * _DT
        else:
            dsoc_du = -_DT / etad[i]
        us_i = np.linspace(lb[i], ub[i], n_bp)
        k_near = int(np.argmin(np.abs(us_i - u1[i])))
        lam = _costate(V_next, bp_new_soc[i][k_near], smin, smax, S, i)
        c_costate[i] = -lam * dsoc_du  # minimize -(lam*dsoc_du*u_i) == maximize lam*dsoc

    STATS["costate_nonzero_count"] += int(np.sum(np.abs(c_costate[:B]) > 1e-9))
    STATS["costate_abs_sum"] += float(np.sum(np.abs(c_costate[:B])))
    norm = max(np.max(np.abs(c_costate)), 1e-12)
    c_cs_scaled = c_obj.copy()
    c_cs_scaled[:B] += (c_costate[:B] / norm) * eps
    res_cs = linprog(c_cs_scaled, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res_cs.success:
        u_cs = res_cs.x[:B]
        z_cs = -(c_obj @ res_cs.x)
        same_z = abs(z_cs - z1) <= max(1e-6, 1e-6 * abs(z1))
        changed = np.abs(u_cs - u1) > 1e-6
        if same_z and np.any(changed):
            STATS["n_degenerate_costate"] += 1
            STATS["n_batteries_changed_costate"] += int(np.sum(changed))
            gain = float(np.sum(c_costate[:B] * (u_cs - u1)))  # true secondary-obj improvement
            STATS["costate_gain_sum"] += gain

    return u1


def run_episode(ch):
    v11._CACHE.clear()
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    for _ in range(ch.num_steps):
        a = v11.policy(view, state)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return state.total_profit


def main():
    orig = v11._lp_joint_solve
    v11._lp_joint_solve = diag_lp_joint_solve
    try:
        for scen_name, scen, nonces in [("BASELINE", Scenario.BASELINE, list(range(0, 20, 2))),
                                          ("CONGESTED", Scenario.CONGESTED, list(range(1, 21, 2)))]:
            for nonce in nonces:
                seed = bch.seed_from_master_nonce(42, nonce)  # dev seed, never used for tuning
                ch = Challenge.generate_instance(seed, Track(s=scen))
                run_episode(ch)
            print(f"[{scen_name}] cumulative steps={STATS['n_steps']}")
    finally:
        v11._lp_joint_solve = orig

    print("\n=== LP degeneracy diagnostic (dev seed 42, BASELINE+CONGESTED) ===")
    n = STATS["n_steps"]
    print(f"total LP-solved steps: {n}")
    print(f"total batteries across those steps: {STATS['n_batteries_total']}")
    print(f"random-perturbation degenerate steps: {STATS['n_degenerate_random']} "
          f"({100*STATS['n_degenerate_random']/max(n,1):.1f}%)  "
          f"batteries changed: {STATS['n_batteries_changed_random']}")
    print(f"costate-objective degenerate steps: {STATS['n_degenerate_costate']} "
          f"({100*STATS['n_degenerate_costate']/max(n,1):.1f}%)  "
          f"batteries changed: {STATS['n_batteries_changed_costate']}  "
          f"total secondary-obj gain: {STATS['costate_gain_sum']:.6f}")
    print(f"costate nonzero battery-steps: {STATS['costate_nonzero_count']} / {STATS['n_batteries_total']} "
          f"({100*STATS['costate_nonzero_count']/max(STATS['n_batteries_total'],1):.1f}%)  "
          f"mean |costate*dsoc_du|: {STATS['costate_abs_sum']/max(STATS['n_batteries_total'],1):.6f}")


if __name__ == "__main__":
    main()
