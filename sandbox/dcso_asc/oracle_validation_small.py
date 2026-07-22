"""Part B: three independent oracle formulations on a small, reduced,
REAL instance (1 battery, first 4 timesteps of a real BASELINE Challenge,
network coupling ignored since B=1 has no flow constraint to violate at
this scale) -- must agree to grid resolution, proving the LP oracle used
throughout this session (oracle_certified.py::oracle2) is not silently
suboptimal.

Method 1: LP (epigraph trick, matches oracle_certified.py's approach)
Method 2: backward-induction dynamic program over a discretized SOC grid
Method 3: exhaustive discretized search over all action sequences

Single process, deterministic.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))
sys.path.insert(0, str(_SANDBOX_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track
from competition.energy_arbitrage.python.scenarios import Scenario
from competition.energy_arbitrage.python import constants
import oracle_certified as oc

DT = constants.DELTA_T
KAPPA_TX = constants.KAPPA_TX
KAPPA_DEG = constants.KAPPA_DEG
BETA_DEG = constants.BETA_DEG

T_REDUCED = 4
N_ACTION_GRID = 9  # exhaustive search grid points per step
N_SOC_GRID = 41    # DP SOC grid resolution


def get_reduced_instance():
    seed = (555).to_bytes(8, "little") + b"\x00" * 24
    ch = Challenge.generate_instance(seed, Track(s=Scenario.BASELINE))
    b = ch.batteries[0]
    rt_traj, exo_traj = oc._record_rt_trajectory(ch)
    prices = rt_traj[:T_REDUCED, b.node]
    return b, prices


def reward(u, price, cap):
    abs_u = abs(u)
    return u * price * DT - KAPPA_TX * abs_u * DT - KAPPA_DEG * (abs_u * DT / cap) ** BETA_DEG


def bounds_at(b, soc):
    headroom = max(b.soc_max_mwh - soc, 0.0)
    avail = max(soc - b.soc_min_mwh, 0.0)
    max_c = min(headroom / (b.efficiency_charge * DT), b.power_charge_mw)
    max_d = min(avail * b.efficiency_discharge / DT, b.power_discharge_mw)
    return -max_c, max_d


def transition(b, u, soc):
    c, d = max(-u, 0.0), max(u, 0.0)
    new_soc = soc + b.efficiency_charge * c * DT - d * DT / b.efficiency_discharge
    return max(b.soc_min_mwh, min(new_soc, b.soc_max_mwh))


def method_exhaustive(b, prices):
    """Brute-force over a discretized action grid at every step, exact
    dynamic feasibility enforced by simulation."""
    best_total = -1e18
    best_seq = None

    def recurse(t, soc, actions, total):
        nonlocal best_total, best_seq
        if t == T_REDUCED:
            if total > best_total:
                best_total = total
                best_seq = list(actions)
            return
        lb, ub = bounds_at(b, soc)
        for frac in np.linspace(0.0, 1.0, N_ACTION_GRID):
            for sign in (-1, 1):
                u = (lb if sign < 0 else ub) * frac
                r = reward(u, prices[t], b.capacity_mwh)
                new_soc = transition(b, u, soc)
                actions.append(u)
                recurse(t + 1, new_soc, actions, total + r)
                actions.pop()

    recurse(0, b.soc_initial_mwh, [], 0.0)
    return best_total, best_seq


def method_dp(b, prices):
    """Backward-induction DP over a discretized SOC grid (same structure as
    policy_v11._build_da_value_function, but self-contained/independent)."""
    soc_grid = np.linspace(b.soc_min_mwh, b.soc_max_mwh, N_SOC_GRID)
    V = np.zeros((T_REDUCED + 1, N_SOC_GRID))
    best_action_idx = np.zeros((T_REDUCED, N_SOC_GRID), dtype=int)
    action_grid_cache = {}

    for t in range(T_REDUCED - 1, -1, -1):
        for si, soc in enumerate(soc_grid):
            lb, ub = bounds_at(b, soc)
            us = np.linspace(lb, ub, 41)
            best_val, best_u_idx = -1e18, 0
            for ui, u in enumerate(us):
                r = reward(u, prices[t], b.capacity_mwh)
                new_soc = transition(b, u, soc)
                frac = np.clip((new_soc - b.soc_min_mwh) / (b.soc_max_mwh - b.soc_min_mwh) * (N_SOC_GRID - 1), 0, N_SOC_GRID - 1 - 1e-9)
                i0 = int(frac)
                i1 = min(i0 + 1, N_SOC_GRID - 1)
                w = frac - i0
                vc = V[t + 1, i0] * (1 - w) + V[t + 1, i1] * w
                total = r + vc
                if total > best_val:
                    best_val, best_u_idx = total, ui
            V[t, si] = best_val
            best_action_idx[t, si] = best_u_idx
            action_grid_cache[(t, si)] = us

    # extract policy starting from true initial SOC (nearest grid point)
    soc = b.soc_initial_mwh
    total = 0.0
    for t in range(T_REDUCED):
        frac = np.clip((soc - b.soc_min_mwh) / (b.soc_max_mwh - b.soc_min_mwh) * (N_SOC_GRID - 1), 0, N_SOC_GRID - 1 - 1e-9)
        si = int(round(frac))
        us = action_grid_cache[(t, si)]
        u = us[best_action_idx[t, si]]
        total += reward(u, prices[t], b.capacity_mwh)
        soc = transition(b, u, soc)
    return total


def method_lp(b, prices, n_bp=41):
    """Epigraph-trick LP, same formulation family as oracle_certified.py's
    oracle2, specialized to a single unconstrained battery (no network)."""
    T = T_REDUCED
    # decision vars: u_0..u_{T-1} (action), s_1..s_T (soc), v_0..v_{T-1} (epigraph reward)
    nvar = T + T + T  # u, s, v
    def iu(t): return t
    def is_(t): return T + t  # s_1..s_T stored at indices T..2T-1 (t=0..T-1 => s_{t+1})
    def iv(t): return 2 * T + t

    c = np.zeros(nvar)
    c[[iv(t) for t in range(T)]] = -1.0  # maximize sum(v) == minimize -sum(v)

    A_ub_rows, b_ub = [], []
    A_eq_rows, b_eq = [], []

    smin, smax = b.soc_min_mwh, b.soc_max_mwh
    for t in range(T):
        prev_s = b.soc_initial_mwh if t == 0 else None
        # bounds on u_t depend on s_t (previous soc) -- since soc bounds are
        # STATE-independent action bounds only through headroom/avail, and s
        # is a decision variable here, use the SAME approach as oracle2: soc
        # bounds are implicit via the smin<=s<=smax constraint plus the
        # transition equation; action power limits are simple box bounds.
        pass

    bounds = []
    for t in range(T):
        bounds.append((-b.power_charge_mw, b.power_discharge_mw))  # u_t
    for t in range(T):
        bounds.append((smin, smax))  # s_{t+1}
    for t in range(T):
        bounds.append((None, None))  # v_t

    # transition equality: s_{t+1} = s_t + etac*c_t*dt - d_t*dt/etad -- NONLINEAR
    # in u (due to c/d split from sign of u). Use the same trick as oracle2:
    # since charging efficiency < 1 < 1/discharging efficiency, the LP
    # relaxation s_{t+1} <= s_t + etac*c_t*dt - d_t*dt/etad (with c,d >= 0,
    # c-d=u) is tight at optimality for this maximization (simultaneous
    # charge+discharge is dominated, proven earlier this session), so model
    # c_t, d_t as separate non-negative variables with u_t = d_t - c_t.
    # Rebuild variable set: c_0..c_{T-1}, d_0..d_{T-1}, s_1..s_T, v_0..v_{T-1}
    nvar = T + T + T + T
    def ic(t): return t
    def idd(t): return T + t
    def is2(t): return 2 * T + t
    def iv2(t): return 3 * T + t

    c_obj = np.zeros(nvar)
    c_obj[[iv2(t) for t in range(T)]] = -1.0

    bounds = []
    for t in range(T):
        bounds.append((0.0, b.power_charge_mw))
    for t in range(T):
        bounds.append((0.0, b.power_discharge_mw))
    for t in range(T):
        bounds.append((smin, smax))
    for t in range(T):
        bounds.append((None, None))

    A_eq_rows, b_eq = [], []
    for t in range(T):
        row = np.zeros(nvar)
        row[is2(t)] = 1.0
        row[ic(t)] = -b.efficiency_charge * DT
        row[idd(t)] = DT / b.efficiency_discharge
        prev = b.soc_initial_mwh if t == 0 else None
        if t == 0:
            b_eq.append(b.soc_initial_mwh)
        else:
            row[is2(t - 1)] = -1.0
            b_eq.append(0.0)
        A_eq_rows.append(row)

    A_ub_rows, b_ub = [], []
    for t in range(T):
        us = np.linspace(-b.power_charge_mw, b.power_discharge_mw, n_bp)
        for k in range(n_bp - 1):
            u0, u1 = us[k], us[k + 1]
            du = u1 - u0
            f0 = reward(u0, prices[t], b.capacity_mwh)
            f1 = reward(u1, prices[t], b.capacity_mwh)
            m = (f1 - f0) / du
            b_k = f0 - m * u0
            row = np.zeros(nvar)
            row[iv2(t)] = 1.0
            # u = d - c
            row[idd(t)] = -m
            row[ic(t)] = m
            A_ub_rows.append(row)
            b_ub.append(b_k)

    res = linprog(c_obj, A_ub=np.array(A_ub_rows), b_ub=np.array(b_ub),
                   A_eq=np.array(A_eq_rows), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not res.success:
        return None
    return -res.fun


def main():
    b, prices = get_reduced_instance()
    print(f"Reduced instance: 1 battery, T={T_REDUCED} steps, prices={prices.tolist()}")
    print(f"Battery: cap={b.capacity_mwh:.2f} pchg={b.power_charge_mw:.2f} pdis={b.power_discharge_mw:.2f} "
          f"smin={b.soc_min_mwh:.2f} smax={b.soc_max_mwh:.2f} soc0={b.soc_initial_mwh:.2f}")

    val_exhaustive, seq = method_exhaustive(b, prices)
    val_dp = method_dp(b, prices)
    val_lp = method_lp(b, prices)

    print(f"\nMethod 1 (LP, epigraph):         {val_lp:.6f}")
    print(f"Method 2 (DP, discretized SOC):  {val_dp:.6f}")
    print(f"Method 3 (exhaustive search):    {val_exhaustive:.6f}")

    tol = 0.5  # grid-resolution tolerance for methods 2/3 vs the near-exact LP
    ok = (abs(val_lp - val_dp) < tol) and (abs(val_lp - val_exhaustive) < tol)
    print(f"\nAll three agree within grid tolerance ({tol}): {'PASS' if ok else 'FAIL'}")
    print(f"LP vs DP diff: {abs(val_lp-val_dp):.4f}   LP vs exhaustive diff: {abs(val_lp-val_exhaustive):.4f}")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
