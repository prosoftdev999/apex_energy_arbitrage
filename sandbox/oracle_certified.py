"""Certified perfect-foresight oracle suite (diagnostic only, not a policy).

Three oracles, each classified per the required taxonomy:

Oracle 1 (independent battery, network-blind, continuous, degradation dropped):
    CLASS B -- rigorous upper bound on the true network-blind optimum.
    Dropping a strictly non-negative cost term (degradation) from the
    objective can only raise the achievable optimal value for any given
    feasible schedule, hence raises the optimum too. LP, solved to certified
    "optimal" status by HiGHS (simplex/IPM optimality = primal-dual gap at
    machine precision by construction).

Oracle 2 (joint, network-feasible, continuous, degradation dropped):
    CLASS B -- rigorous upper bound on the true production-exact,
    network-feasible perfect-foresight optimum, by the same argument as
    Oracle 1, with the true PTDF flow constraints included as hard LP
    constraints (not a discrete grid, not a heuristic repair).

Oracle 3 (relaxed: network-blind AND degradation AND transaction cost dropped):
    CLASS B -- a looser, structurally simpler upper bound (revenue only,
    subject only to energy/power/SOC physics) requiring no argument beyond
    "removing cost terms and constraints cannot lower an optimum."

All three use the identical realized RT price trajectory (recorded once,
verified action-independent) and the identical PTDF/network data used by the
real simulator for that instance.

Key facts verified directly from source before writing this file:
  - PTDF's slack-bus column is exactly zero for every line (traced through
    Network._compute_ptdf's reduced-matrix construction), so the value used
    for the slack bus's own injection does not affect flow constraints at all.
  - Simultaneous charge>0 and discharge>0 in one step is never part of an
    optimal LP solution: for any (c,d) with overlap m=min(c,d)>0, replacing
    with (c-m,d-m) leaves revenue exactly unchanged ((d-c) is invariant),
    strictly lowers transaction cost (proportional to c+d), and weakly
    RAISES the resulting SOC (since eta_c<=1<=1/eta_d). So no explicit
    complementarity constraint is needed -- the optimal value is unaffected,
    and at optimality the LP's tx-cost-on-(c+d) formulation exactly equals
    production's tx-cost-on-|d-c| formulation, since one of c,d is zero.
  - conservative.py's _enforce_profit_floor inductively guarantees
    conservative's cumulative profit never goes negative, so
    baseline_profit = max(greedy, conservative) >= 0 always (can be near
    zero, explaining huge raw quality ratios, but never negative).
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import (
    Challenge, Track, NextRTPricesOverride, Solution,
)
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch

_DT = 0.25
_KAPPA_TX = 0.25
_KAPPA_DEG = 1.0
_BETA_DEG = 2.0
MASTER_SEED = 123


def _record_rt_trajectory(ch):
    """RT prices are independent of the policy's actions (the next-seed draw
    happens before the action-dependent state update in take_step), so they
    can be recorded once with a trivial zero-action stand-in."""
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    rt = np.zeros((ch.num_steps, ch.network.num_nodes))
    exo = np.zeros((ch.num_steps, ch.network.num_nodes))
    for t in range(ch.num_steps):
        rt[t] = state.rt_prices
        exo[t] = state.exogenous_injections
        a = [0.0] * ch.num_batteries
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        from competition.energy_arbitrage.python.challenge import NextRTPricesGenerate
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return rt, exo


def _battery_arrays(ch):
    cap = np.array([b.capacity_mwh for b in ch.batteries])
    pchg = np.array([b.power_charge_mw for b in ch.batteries])
    pdis = np.array([b.power_discharge_mw for b in ch.batteries])
    etac = np.array([b.efficiency_charge for b in ch.batteries])
    etad = np.array([b.efficiency_discharge for b in ch.batteries])
    smin = np.array([b.soc_min_mwh for b in ch.batteries])
    smax = np.array([b.soc_max_mwh for b in ch.batteries])
    node = np.array([b.node for b in ch.batteries])
    soc_init = np.array([b.soc_initial_mwh for b in ch.batteries])
    return dict(cap=cap, pchg=pchg, pdis=pdis, etac=etac, etad=etad,
                smin=smin, smax=smax, node=node, soc_init=soc_init)


def _per_battery_lp(T, price_i, pchg_i, pdis_i, etac_i, etad_i, smin_i, smax_i, soc_init_i,
                     include_tx=True):
    """Single-battery continuous LP: maximize revenue [- tx_cost] s.t. exact
    SOC-transition-implied bounds and power bounds. Degradation always
    dropped (see module docstring). Returns (profit, status, schedule)."""
    nvar = 2 * T  # c_0..c_{T-1}, d_0..d_{T-1}
    c_obj = np.zeros(nvar)
    for t in range(T):
        tx = _KAPPA_TX * _DT if include_tx else 0.0
        c_obj[t] = price_i[t] * _DT + tx          # cost coefficient for charge (minimize -profit)
        c_obj[T + t] = tx - price_i[t] * _DT       # cost coefficient for discharge

    bounds = [(0.0, pchg_i)] * T + [(0.0, pdis_i)] * T

    A_ub_rows, b_ub = [], []
    for t in range(T):
        row_hi = np.zeros(nvar)
        row_lo = np.zeros(nvar)
        for s in range(t + 1):
            row_hi[s] = etac_i * _DT
            row_hi[T + s] = -_DT / etad_i
            row_lo[s] = -etac_i * _DT
            row_lo[T + s] = _DT / etad_i
        A_ub_rows.append(row_hi); b_ub.append(smax_i - soc_init_i)
        A_ub_rows.append(row_lo); b_ub.append(soc_init_i - smin_i)

    res = linprog(c_obj, A_ub=np.array(A_ub_rows), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        return None, res.status, res.message, None
    c = res.x[:T]; d = res.x[T:]
    schedule = (d - c).tolist()
    profit = -res.fun
    return profit, res.status, res.message, schedule


def oracle1(ch, rt_traj, ba):
    """Independent per-battery, network-blind, continuous, degradation-dropped."""
    T = ch.num_steps
    total = 0.0
    schedules = []
    for i in range(len(ba["cap"])):
        price_i = rt_traj[:, ba["node"][i]]
        profit, status, msg, sched = _per_battery_lp(
            T, price_i, ba["pchg"][i], ba["pdis"][i], ba["etac"][i], ba["etad"][i],
            ba["smin"][i], ba["smax"][i], ba["soc_init"][i], include_tx=True)
        if profit is None:
            return None, status, msg, None
        total += profit
        schedules.append(sched)
    return total, "optimal", "all batteries solved to optimality", schedules


def oracle3_relaxed(ch, rt_traj, ba):
    """Network-blind, degradation AND transaction cost both dropped -- the
    loosest, structurally simplest bound: pure buy-low/sell-high revenue
    under exact energy/power/SOC physics only."""
    T = ch.num_steps
    total = 0.0
    schedules = []
    for i in range(len(ba["cap"])):
        price_i = rt_traj[:, ba["node"][i]]
        profit, status, msg, sched = _per_battery_lp(
            T, price_i, ba["pchg"][i], ba["pdis"][i], ba["etac"][i], ba["etad"][i],
            ba["smin"][i], ba["smax"][i], ba["soc_init"][i], include_tx=False)
        if profit is None:
            return None, status, msg, None
        total += profit
        schedules.append(sched)
    return total, "optimal", "all batteries solved to optimality", schedules


def oracle2(ch, rt_traj, exo_traj, ba):
    """Joint, network-feasible, continuous, degradation dropped."""
    B = len(ba["cap"])
    T = ch.num_steps
    node = ba["node"]
    ptdf = np.asarray(ch.network.ptdf)
    limits = np.asarray(ch.network.flow_limits)
    slack = ch.network.slack_bus
    L = ptdf.shape[0]

    nvar = 2 * T * B
    def c_idx(t, i): return t * B + i
    def d_idx(t, i): return T * B + t * B + i

    price = rt_traj[:, node]
    c_obj = np.zeros(nvar)
    for t in range(T):
        for i in range(B):
            c_obj[c_idx(t, i)] = price[t, i] * _DT + _KAPPA_TX * _DT
            c_obj[d_idx(t, i)] = _KAPPA_TX * _DT - price[t, i] * _DT

    bounds = [(0.0, ba["pchg"][i]) for t in range(T) for i in range(B)] + \
             [(0.0, ba["pdis"][i]) for t in range(T) for i in range(B)]

    A_ub_rows, b_ub = [], []
    for i in range(B):
        for t in range(T):
            row_hi = np.zeros(nvar)
            row_lo = np.zeros(nvar)
            for s in range(t + 1):
                row_hi[c_idx(s, i)] = ba["etac"][i] * _DT
                row_hi[d_idx(s, i)] = -_DT / ba["etad"][i]
                row_lo[c_idx(s, i)] = -ba["etac"][i] * _DT
                row_lo[d_idx(s, i)] = _DT / ba["etad"][i]
            A_ub_rows.append(row_hi); b_ub.append(ba["smax"][i] - ba["soc_init"][i])
            A_ub_rows.append(row_lo); b_ub.append(ba["soc_init"][i] - ba["smin"][i])

    for t in range(T):
        exo_now = exo_traj[t]
        base_flow = ptdf @ exo_now  # slack column is exactly zero (verified from source), so
        # the recorded (already slack-balanced) exogenous vector can be used directly.
        for l in range(L):
            row_hi = np.zeros(nvar)
            row_lo = np.zeros(nvar)
            for i in range(B):
                row_hi[d_idx(t, i)] = ptdf[l, node[i]]
                row_hi[c_idx(t, i)] = -ptdf[l, node[i]]
                row_lo[d_idx(t, i)] = -ptdf[l, node[i]]
                row_lo[c_idx(t, i)] = ptdf[l, node[i]]
            A_ub_rows.append(row_hi); b_ub.append(limits[l] - base_flow[l])
            A_ub_rows.append(row_lo); b_ub.append(limits[l] + base_flow[l])

    A_ub = np.array(A_ub_rows); b_ub = np.array(b_ub)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None, None, res.status, res.message
    c = res.x[:T * B].reshape(T, B); d = res.x[T * B:].reshape(T, B)
    schedule = (d - c).tolist()
    profit = -res.fun
    return profit, schedule, "optimal", res.message


def replay_through_production(ch, schedule_per_step, rt_traj, bound_tol=1e-6):
    """Replay an oracle schedule through the REAL Challenge.take_step (hence
    Challenge.compute_profit, WITH degradation cost included) using the exact
    recorded RT trajectory via NextRTPricesOverride. schedule_per_step must be
    a list of T action-lists (one B-length list per step, matching what
    Solution.schedule/take_step expect).

    A precomputed whole-horizon schedule is applied against a SEQUENTIALLY
    re-derived state.action_bounds (production tracks SOC step by step with
    its own clip-based update; the LP tracks it via a cumulative-sum
    expression). These are mathematically equivalent but can differ at the
    ~1e-13 floating-point level after 96+ compounding steps -- exactly the
    scale a real policy submission would also clip away by construction
    (every policy in this codebase does np.clip(action, lb, ub) using the
    CURRENT state's own bounds, never its own internally-tracked copy). We
    clip to state.action_bounds at replay time (matching real policy
    behavior) and separately report the magnitude of any clipping needed, so
    a genuine infeasibility is not silently hidden behind a benign one.

    Returns (profit, bound_violation, flow_violation, error_message,
    max_clip_magnitude)."""
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    max_clip = 0.0
    for t in range(ch.num_steps):
        a = [float(x) for x in schedule_per_step[t]]
        for i, (lo, hi) in enumerate(state.action_bounds):
            if a[i] < lo - bound_tol or a[i] > hi + bound_tol:
                return state.total_profit, True, False, f"step {t} battery {i}: {a[i]} not in [{lo},{hi}]", max_clip
            clipped = min(max(a[i], lo), hi)
            max_clip = max(max_clip, abs(clipped - a[i]))
            a[i] = clipped
        next_prices = rt_traj[t + 1].tolist() if t + 1 < ch.num_steps else [0.0] * ch.network.num_nodes
        try:
            state = ch.take_step(state, a, NextRTPricesOverride(next_prices))
        except ValueError as e:
            msg = str(e)
            is_flow = "flow limit" in msg or "Line" in msg
            return state.total_profit, (not is_flow), is_flow, msg, max_clip
    return state.total_profit, False, False, None, max_clip


def independent_schedule_profit_check(ch, ba, schedule_scalar_per_battery, rt_traj, include_tx):
    """For network-blind oracles (1/3): independently recompute profit from
    the schedule using the exact production compute_profit formula (with
    degradation INCLUDED, unlike the LP objective), without going through
    take_step (which would reject it on network grounds by design -- these
    oracles are deliberately network-blind). Returns the degradation-inclusive
    profit for direct comparison against the LP's degradation-EXCLUDED value."""
    total = 0.0
    for t in range(ch.num_steps):
        for i, batt in enumerate(ch.batteries):
            u = schedule_scalar_per_battery[i][t]
            price = rt_traj[t, ba["node"][i]]
            revenue = u * price * _DT
            abs_u = abs(u)
            tx = _KAPPA_TX * abs_u * _DT if include_tx else 0.0
            deg = _KAPPA_DEG * ((abs_u * _DT / batt.capacity_mwh) ** _BETA_DEG)
            total += revenue - tx - deg
    return total


def _reshape_to_per_step(schedule_TxB_or_list, T, B):
    """Normalize either a (T,B)-shaped list-of-lists or a list of B per-battery
    scalar-sequences into a list of T action-lists (length B each)."""
    arr = np.asarray(schedule_TxB_or_list)
    if arr.shape == (T, B):
        return [arr[t].tolist() for t in range(T)]
    elif arr.shape == (B, T):
        return [[arr[i, t] for i in range(B)] for t in range(T)]
    raise ValueError(f"unexpected schedule shape {arr.shape} for T={T}, B={B}")


def run_all(scenario, nonces, label):
    import importlib
    v6 = importlib.import_module("policy_v6")
    rows = []
    for nonce in nonces:
        seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
        ch = Challenge.generate_instance(seed, Track(s=scenario))
        ba = _battery_arrays(ch)
        T, B = ch.num_steps, ch.num_batteries
        rt_traj, exo_traj = _record_rt_trajectory(ch)
        _, baseline_profit = ch.compute_baseline()

        v6._CACHE.clear()
        view = ch.to_policy_view()
        import random
        rng = random.Random(); rng.seed(ch._hidden_seed)
        state = ch._initial_state(rng)
        from competition.energy_arbitrage.python.challenge import NextRTPricesGenerate
        for t in range(T):
            a = v6.policy(view, state)
            next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
            state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
        v6_profit = state.total_profit

        t0 = time.time()
        o1_profit, o1_status, o1_msg, o1_sched = oracle1(ch, rt_traj, ba)
        o1_time = time.time() - t0
        o1_check = independent_schedule_profit_check(ch, ba, o1_sched, rt_traj, include_tx=True) if o1_sched else None

        t0 = time.time()
        o3_profit, o3_status, o3_msg, o3_sched = oracle3_relaxed(ch, rt_traj, ba)
        o3_time = time.time() - t0

        t0 = time.time()
        o2_profit, o2_sched, o2_status, o2_msg = oracle2(ch, rt_traj, exo_traj, ba)
        o2_time = time.time() - t0

        replay_profit, bviol, fviol, err, max_clip = (None, None, None, None, None)
        if o2_sched is not None:
            sched_per_step = _reshape_to_per_step(o2_sched, T, B)
            replay_profit, bviol, fviol, err, max_clip = replay_through_production(ch, sched_per_step, rt_traj)

        def q(p):
            return max(-10.0, min((p - baseline_profit) / (baseline_profit + 1e-6), 10.0))

        row = dict(
            scenario=label, nonce=nonce, baseline_profit=baseline_profit, v6_profit=v6_profit,
            v6_quality=q(v6_profit),
            oracle1_profit=o1_profit, oracle1_status=o1_status, oracle1_time=o1_time,
            oracle1_degcheck=o1_check,
            oracle3_profit=o3_profit, oracle3_status=o3_status, oracle3_time=o3_time,
            oracle2_profit=o2_profit, oracle2_status=o2_status, oracle2_time=o2_time,
            oracle2_replay_profit=replay_profit, oracle2_bound_violation=bviol,
            oracle2_flow_violation=fviol, oracle2_replay_error=err, oracle2_max_clip=max_clip,
            oracle2_quality=(q(o2_profit) if o2_profit is not None else None),
        )
        rows.append(row)
        print(f"{label} nonce={nonce:3d} baseline={baseline_profit:10.1f} v6={v6_profit:10.1f}(q={q(v6_profit):+.3f}) "
              f"O1={o1_profit:10.1f} O3={o3_profit:10.1f} O2={o2_profit:10.1f}(q={q(o2_profit):+.3f}) "
              f"O2_replay={replay_profit:10.1f} bviol={bviol} fviol={fviol} "
              f"O2_gap(LPobj-replay)={o2_profit-replay_profit:8.2f} [{o2_time:.2f}s]")
    return rows


def main():
    baseline_nonces = list(range(0, 100, 5))
    congested_nonces = list(range(1, 100, 5))
    all_rows = []
    all_rows += run_all(Scenario.BASELINE, baseline_nonces, "BASELINE")
    all_rows += run_all(Scenario.CONGESTED, congested_nonces, "CONGESTED")

    import csv
    with open("oracle_certified_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    for scen in ["BASELINE", "CONGESTED"]:
        rows = [r for r in all_rows if r["scenario"] == scen]
        o2q = np.array([r["oracle2_quality"] for r in rows])
        v6q = np.array([r["v6_quality"] for r in rows])
        print(f"\n=== {scen} (n={len(rows)}) ===")
        print(f"Oracle2 (certified upper bound) quality: mean={o2q.mean():+.4f} median={np.median(o2q):+.4f} "
              f"P10={np.percentile(o2q,10):+.4f} min={o2q.min():+.4f} max={o2q.max():+.4f} "
              f"n_saturated(+10)={int(np.sum(o2q>=9.9999))}")
        print(f"policy_v6 quality: mean={v6q.mean():+.4f}")
        print(f"gap (Oracle2 - v6): mean={np.mean(o2q-v6q):+.4f}")
        bviol_any = any(r["oracle2_bound_violation"] for r in rows)
        fviol_any = any(r["oracle2_flow_violation"] for r in rows)
        max_replay_gap = max(abs(r["oracle2_profit"] - r["oracle2_replay_profit"]) for r in rows)
        print(f"replay check: any bound violation={bviol_any} any flow violation={fviol_any} "
              f"max|LP_obj - replay_profit|={max_replay_gap:.4f}")


if __name__ == "__main__":
    main()
