"""Independent red-team audit of the perfect-foresight network-feasible
oracle ceiling (previously reported as 0.7503278 via oracle_certified.py's
oracle2, dense LP).

This is a FRESH, independent re-derivation -- it does not import or call
oracle_certified.py. Every fact used below was re-verified directly from
the production source in this session (challenge.py, battery.py, market.py,
network.py, greedy.py, conservative.py, policy_view.py, constants.py,
scenarios.py), not assumed from prior work:

  - compute_profit: revenue = u*price*dt, tx = kappa_tx*|u|*dt,
    deg = kappa_deg*(|u|*dt/cap)^beta, profit accumulated by simple += per
    step (no path-dependent cost beyond the SOC-linked action bounds).
  - apply_action_to_soc: soc_{t+1} = soc_t + etac*dt*c_t - dt/etad*d_t,
    c_t=max(-u_t,0), d_t=max(u_t,0); clipped to [smin,smax] (a no-op in
    practice since compute_action_bounds guarantees the result lands inside
    the box for any action within its returned bounds).
  - take_step STRICTLY REJECTS (raises ValueError) on any action-bound or
    flow violation -- there is no repair in production; repair logic lives
    only inside policies (best_candidate.py etc.), which must self-repair
    BEFORE calling take_step.
  - PTDF's slack-bus row/column is exactly zero by construction (row_map
    excludes slack_bus when building/inverting the reduced susceptance
    matrix, and the full X matrix is zero-initialized outside row_map x
    row_map), so the slack bus's own assumed injection value never affects
    flow constraints.
  - RT prices are the ONLY per-step randomness that affects anything other
    than State.rt_prices (verified: NextRTPricesGenerate's branch in
    take_step computes rt_prices purely from the seed + state.exogenous_
    injections + market internal state -- no dependency on `action`
    whatsoever), and are drawn fresh every step with a fresh seed (recorded
    once via a zero-action dry run, since price generation doesn't depend
    on the action taken).
  - compute_baseline returns exactly max(greedy_profit, conservative_profit)
    via a real full simulation of each baseline through take_step -- not an
    approximation -- so the reported baseline is exactly what production
    uses, independent of any shadow-baseline precomputation elsewhere in
    this project.

Oracle formulation used here (fresh, NOT copied from oracle_certified.py):
  Variables: c_{t,i} in [0,pchg_i], d_{t,i} in [0,pdis_i] for all steps t and
  batteries i (charge/discharge legs), plus explicit SOC variables soc_{t,i}
  in [smin_i,smax_i] for t=1..T (soc_0 is a known constant, not a variable).
  Equality constraints (T*B of them): soc_{t+1,i} - soc_{t,i}
  - etac_i*dt*c_{t,i} + dt/etad_i*d_{t,i} = 0 (t=0 uses the constant soc_0,i
  in place of a soc_{0,i} variable). SOC bounds are enforced directly via
  variable bounds (no extra inequality rows needed) -- this is a cleaner,
  much sparser formulation than a cumulative-sum-per-timestep approach
  (O(T*B) nonzeros instead of O(T^2*B)), which also makes the exact
  network-feasible LP tractable for CAPSTONE (B=100, T=192) for the first
  time this session -- the prior dense cumulative-sum formulation required
  ~43.9 GiB and failed with a memory error.
  Inequality constraints (flow limits): for every t and line l,
  |sum_i ptdf[l,node_i]*(d_{t,i}-c_{t,i}) + base_flow_l(t)| <= limit_l,
  written as two rows (hi/lo) per (t,l).
  Objective: maximize sum_{t,i} [price_{t,i}*dt*(d_{t,i}-c_{t,i})
  - kappa_tx*dt*(c_{t,i}+d_{t,i})]  (degradation dropped -- CLASS B upper
  bound: dropping a nonnegative cost term can only raise the achievable
  optimum, for any fixed schedule and hence for the max over schedules too).

Simultaneous c_{t,i}>0 and d_{t,i}>0 is never optimal (removing the overlap
strictly reduces tx cost at fixed net action d-c, so the LP's own optimum
never uses it) -- this is re-derived independently in this docstring, not
just cited from prior work: for fixed u=d-c, tx cost is proportional to
c+d >= |d-c| (triangle inequality), with equality iff c=0 or d=0, so any
optimal LP solution has zero overlap and its tx cost exactly matches the
production formula's kappa_tx*|u|*dt for the corresponding net action.

This script produces:
  sandbox/oracle_red_team_results.csv  (all 100 instances, full required columns)
  sandbox/oracle_red_team_report.md
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import (
    Challenge, Track, NextRTPricesOverride, NextRTPricesGenerate,
)
from competition.energy_arbitrage.python.scenarios import Scenario

import random

_DT = 0.25
_KAPPA_TX = 0.25
_KAPPA_DEG = 1.0
_BETA_DEG = 2.0
MASTER_SEED = 987654
SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + b"\x00" * 24


def record_rt_and_exo_trajectory(ch):
    """RT prices provably don't depend on the action taken (re-verified from
    take_step source above), so a zero-action dry run records the true
    trajectory exactly."""
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    rt = np.zeros((ch.num_steps, ch.network.num_nodes))
    exo = np.zeros((ch.num_steps, ch.network.num_nodes))
    for t in range(ch.num_steps):
        rt[t] = state.rt_prices
        exo[t] = state.exogenous_injections
        a = [0.0] * ch.num_batteries
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return rt, exo


def battery_arrays(ch):
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


def solve_oracle_sparse(ch, rt_traj, exo_traj, ba):
    """Fresh sparse LP formulation (see module docstring). Returns dict with
    profit, schedule (T,B), solver status, and residual diagnostics."""
    T = ch.num_steps
    B = len(ba["cap"])
    node = ba["node"]
    ptdf = np.asarray(ch.network.ptdf)
    limits = np.asarray(ch.network.flow_limits)
    slack = ch.network.slack_bus
    L = ptdf.shape[0]

    # Variable layout: c[t,i] -> t*B+i ; d[t,i] -> T*B + t*B+i ;
    # soc[t,i] for t=1..T -> 2*T*B + (t-1)*B + i
    def c_idx(t, i): return t * B + i
    def d_idx(t, i): return T * B + t * B + i
    def s_idx(t, i): return 2 * T * B + (t - 1) * B + i  # t in 1..T

    nvar = 2 * T * B + T * B

    price = rt_traj[:, node]  # (T,B)
    c_obj = np.zeros(nvar)
    for t in range(T):
        c_obj[c_idx(t, np.arange(B))] = price[t] * _DT + _KAPPA_TX * _DT
        c_obj[d_idx(t, np.arange(B))] = _KAPPA_TX * _DT - price[t] * _DT
    # soc has zero objective coefficient (no terminal SOC value in production)

    bounds = [None] * nvar
    for t in range(T):
        for i in range(B):
            bounds[c_idx(t, i)] = (0.0, float(ba["pchg"][i]))
            bounds[d_idx(t, i)] = (0.0, float(ba["pdis"][i]))
    for t in range(1, T + 1):
        for i in range(B):
            bounds[s_idx(t, i)] = (float(ba["smin"][i]), float(ba["smax"][i]))

    # Equality constraints: SOC recursion, T*B rows, sparse (<=4 nonzeros/row)
    eq_rows, eq_cols, eq_data, eq_b = [], [], [], []
    row = 0
    for t in range(T):
        for i in range(B):
            if t == 0:
                # soc[1,i] - etac*dt*c[0,i] + dt/etad*d[0,i] = soc_init_i
                eq_rows += [row, row, row]
                eq_cols += [s_idx(1, i), c_idx(0, i), d_idx(0, i)]
                eq_data += [1.0, -ba["etac"][i] * _DT, _DT / ba["etad"][i]]
                eq_b.append(float(ba["soc_init"][i]))
            else:
                # soc[t+1,i] - soc[t,i] - etac*dt*c[t,i] + dt/etad*d[t,i] = 0
                eq_rows += [row, row, row, row]
                eq_cols += [s_idx(t + 1, i), s_idx(t, i), c_idx(t, i), d_idx(t, i)]
                eq_data += [1.0, -1.0, -ba["etac"][i] * _DT, _DT / ba["etad"][i]]
                eq_b.append(0.0)
            row += 1
    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(row, nvar))
    b_eq = np.array(eq_b)

    # Inequality constraints: flow limits, 2*T*L rows, 2*B nonzeros/row
    ub_rows, ub_cols, ub_data, ub_b = [], [], [], []
    row = 0
    for t in range(T):
        base_flow = ptdf @ exo_traj[t]  # slack column exactly zero, verified from source
        for l in range(L):
            ptdf_l_node = ptdf[l, node]
            # hi: sum_i ptdf[l,node_i]*(d-c) <= limit - base_flow
            for i in range(B):
                ub_rows.append(row); ub_cols.append(d_idx(t, i)); ub_data.append(ptdf_l_node[i])
                ub_rows.append(row); ub_cols.append(c_idx(t, i)); ub_data.append(-ptdf_l_node[i])
            ub_b.append(float(limits[l] - base_flow[l]))
            row += 1
            # lo: -sum_i ptdf[l,node_i]*(d-c) <= limit + base_flow
            for i in range(B):
                ub_rows.append(row); ub_cols.append(d_idx(t, i)); ub_data.append(-ptdf_l_node[i])
                ub_rows.append(row); ub_cols.append(c_idx(t, i)); ub_data.append(ptdf_l_node[i])
            ub_b.append(float(limits[l] + base_flow[l]))
            row += 1
    A_ub = sparse.csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(row, nvar))
    b_ub = np.array(ub_b)

    # NOTE: method="highs" (dual simplex) was tested first and does NOT
    # converge within 90s on CAPSTONE-scale instances (T=192,B=100,L=300;
    # 57600 vars, 134400 rows) -- confirmed via direct experiment, still
    # ~1.9e7 primal infeasibility after 90s. method="highs-ipm" (interior
    # point + crossover) converges to a certified optimal solution in ~156s
    # on the same instance (interior solution objective gap 3.9e-10
    # relative). This is what finally makes CAPSTONE's exact network-
    # feasible oracle tractable -- the original oracle_certified.py's dense
    # formulation failed on MEMORY (43.9 GiB); this sparse formulation fixes
    # that (peak ~200MB), but the naive simplex solve path still separately
    # fails on TIME, which is why highs-ipm specifically is required here.
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                   method="highs-ipm", options={"time_limit": 280})
    if not res.success:
        return dict(success=False, status=res.status, message=res.message)

    x = res.x
    c_arr = np.array([[x[c_idx(t, i)] for i in range(B)] for t in range(T)])
    d_arr = np.array([[x[d_idx(t, i)] for i in range(B)] for t in range(T)])
    soc_arr = np.array([[x[s_idx(t, i)] for i in range(B)] for t in range(1, T + 1)])
    schedule = d_arr - c_arr
    profit = -res.fun

    # Residual diagnostics (independently recomputed, not taken from solver internals alone)
    primal_ub_resid = float(np.max(A_ub @ x - b_ub)) if A_ub.shape[0] else 0.0
    primal_eq_resid = float(np.max(np.abs(A_eq @ x - b_eq))) if A_eq.shape[0] else 0.0
    # SOC residual: recompute soc trajectory manually from (c,d) and compare to solver's soc vars
    soc_manual = np.zeros((T, B))
    prev = ba["soc_init"].copy()
    for t in range(T):
        prev = prev + ba["etac"] * c_arr[t] * _DT - d_arr[t] * _DT / ba["etad"]
        soc_manual[t] = prev
    soc_residual = float(np.max(np.abs(soc_manual - soc_arr)))
    bound_violation = float(max(0.0, np.max(ba["smin"] - soc_arr), np.max(soc_arr - ba["smax"])))

    return dict(success=True, status=res.status, profit=profit, schedule=schedule,
                primal_ub_residual=primal_ub_resid, primal_eq_residual=primal_eq_resid,
                soc_residual=soc_residual, soc_bound_violation=bound_violation,
                n_vars=nvar, n_eq=A_eq.shape[0], n_ub=A_ub.shape[0])


def replay_through_production(ch, schedule, rt_traj, bound_tol=1e-6):
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    max_clip = 0.0
    bviol, fviol = False, False
    err = None
    for t in range(ch.num_steps):
        a = [float(x) for x in schedule[t]]
        for i, (lo, hi) in enumerate(state.action_bounds):
            if a[i] < lo - bound_tol or a[i] > hi + bound_tol:
                bviol = True
                return state.total_profit, bviol, fviol, f"step {t} battery {i}: {a[i]} not in [{lo},{hi}]", max_clip
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
    return state.total_profit, bviol, fviol, err, max_clip


def q(profit, baseline):
    return max(-10.0, min((profit - baseline) / (baseline + 1e-6), 10.0))


def main():
    import csv as _csv
    current_rows = {}
    with open(_ID7_DIR / "best_candidate_benchmark_results.csv") as f:
        for r in _csv.DictReader(f):
            current_rows[int(r["nonce"])] = dict(
                scenario=r["scenario"], miner_profit=float(r["miner_profit"]),
                current_quality=float(r["quality_int"]) / 1e6,
            )

    results = []
    t_start = time.time()
    for nonce in range(100):
        scenario = SCENARIO_ORDER[nonce % 5]
        seed = seed_from_master_nonce(MASTER_SEED, nonce)
        ch = Challenge.generate_instance(seed, Track(s=scenario))
        ba = battery_arrays(ch)
        rt_traj, exo_traj = record_rt_and_exo_trajectory(ch)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            _, baseline_profit = ch.compute_baseline()

        t0 = time.time()
        sol = solve_oracle_sparse(ch, rt_traj, exo_traj, ba)
        solve_time = time.time() - t0

        cur = current_rows[nonce]
        row = dict(scenario=scenario.name, nonce=nonce, baseline_profit=baseline_profit,
                   current_policy_profit=cur["miner_profit"], current_quality=cur["current_quality"],
                   solve_time=solve_time)

        if not sol["success"]:
            row.update(solver_status="FAILED", oracle_profit=None, oracle_quality=None,
                       replay_profit=None, replay_bound_violation=None, replay_flow_violation=None,
                       primal_ub_residual=None, primal_eq_residual=None, soc_residual=None,
                       objective_replay_discrepancy=None)
        else:
            replay_profit, bviol, fviol, err, max_clip = replay_through_production(ch, sol["schedule"], rt_traj)
            oracle_quality = q(sol["profit"], baseline_profit)
            row.update(
                solver_status="optimal", oracle_profit=sol["profit"], oracle_quality=oracle_quality,
                replay_profit=replay_profit, replay_bound_violation=bviol, replay_flow_violation=fviol,
                replay_max_clip=max_clip,
                primal_ub_residual=sol["primal_ub_residual"], primal_eq_residual=sol["primal_eq_residual"],
                soc_residual=sol["soc_residual"], soc_bound_violation=sol["soc_bound_violation"],
                objective_replay_discrepancy=sol["profit"] - replay_profit,
                current_to_oracle_gap=oracle_quality - cur["current_quality"],
                n_vars=sol["n_vars"], n_eq=sol["n_eq"], n_ub=sol["n_ub"],
            )
        results.append(row)
        print(f"nonce={nonce:3d} {scenario.name:10s} status={row['solver_status']:8s} "
              f"cur_q={row['current_quality']:+8.4f} "
              f"oracle_q={row.get('oracle_quality', float('nan')):+8.4f} "
              f"gap={row.get('current_to_oracle_gap', float('nan')):+8.4f} "
              f"primal_resid={row.get('primal_ub_residual', float('nan')):.2e} "
              f"soc_resid={row.get('soc_residual', float('nan')):.2e} "
              f"objdiff={row.get('objective_replay_discrepancy', float('nan')):+.4f} "
              f"[{solve_time:.2f}s] elapsed={time.time()-t_start:.0f}s", flush=True)

        with open(_SANDBOX_DIR / "oracle_red_team_results.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)

    # ---- aggregate ----
    ok = [r for r in results if r["solver_status"] == "optimal"]
    cur_qs = np.array([r["current_quality"] for r in results])
    orc_qs = np.array([r["oracle_quality"] for r in ok])
    current_score = cur_qs.mean() / 10.0
    # for any failed solves (should be none), fall back to current quality (conservative: no headroom claimed)
    orc_full = np.array([r.get("oracle_quality", r["current_quality"]) if r["solver_status"] == "optimal"
                          else r["current_quality"] for r in results])
    oracle_score = orc_full.mean() / 10.0

    any_bviol = any(r.get("replay_bound_violation") for r in ok)
    any_fviol = any(r.get("replay_flow_violation") for r in ok)
    max_obj_replay_gap = max(abs(r["objective_replay_discrepancy"]) for r in ok) if ok else None
    max_primal_resid = max(r["primal_ub_residual"] for r in ok) if ok else None
    max_eq_resid = max(r["primal_eq_residual"] for r in ok) if ok else None
    max_soc_resid = max(r["soc_residual"] for r in ok) if ok else None
    n_failed = sum(1 for r in results if r["solver_status"] != "optimal")

    with open(_SANDBOX_DIR / "oracle_red_team_report.md", "w") as f:
        f.write("# Independent Oracle Red-Team Audit\n\n")
        f.write(f"Instances: {len(results)}/100, solved optimally: {len(ok)}, failed: {n_failed}\n\n")
        f.write(f"Current score (recomputed from CSV, clipped): {current_score:.7f} (reference: 0.7339332)\n\n")
        f.write(f"Independent red-team oracle score: {oracle_score:.7f} "
                f"(previously reported via oracle_certified.py: 0.7503278)\n\n")
        f.write(f"Any replay bound violation: {any_bviol}\n")
        f.write(f"Any replay flow violation: {any_fviol}\n")
        f.write(f"Max |oracle objective - replay profit|: {max_obj_replay_gap}\n")
        f.write(f"Max primal (inequality) residual: {max_primal_resid}\n")
        f.write(f"Max equality (SOC recursion) residual: {max_eq_resid}\n")
        f.write(f"Max SOC-variable-vs-manual-recompute residual: {max_soc_resid}\n\n")
        f.write("## Per-scenario means\n\n")
        f.write("| Scenario | n | current_mean | oracle_mean | gain |\n|---|---|---|---|---|\n")
        for scen in SCENARIO_ORDER:
            sub = [r for r in results if r["scenario"] == scen.name]
            c = np.mean([r["current_quality"] for r in sub])
            o = np.mean([r.get("oracle_quality", r["current_quality"]) if r["solver_status"] == "optimal"
                         else r["current_quality"] for r in sub])
            f.write(f"| {scen.name} | {len(sub)} | {c:.4f} | {o:.4f} | {o-c:+.4f} |\n")
        f.write("\n## Top 20 current-to-oracle gap instances\n\n")
        ranked = sorted(ok, key=lambda r: -(r["current_to_oracle_gap"]))[:20]
        f.write("| nonce | scenario | current_q | oracle_q | gap |\n|---|---|---|---|---|\n")
        for r in ranked:
            f.write(f"| {r['nonce']} | {r['scenario']} | {r['current_quality']:.4f} | "
                    f"{r['oracle_quality']:.4f} | {r['current_to_oracle_gap']:+.4f} |\n")

    print(f"\ncurrent_score={current_score:.7f} oracle_score={oracle_score:.7f} "
          f"(prior claim: 0.7503278, diff={oracle_score-0.7503278:+.7f})")
    print(f"any_bound_violation={any_bviol} any_flow_violation={any_fviol} "
          f"max_obj_replay_gap={max_obj_replay_gap} max_primal_resid={max_primal_resid} "
          f"max_eq_resid={max_eq_resid} max_soc_resid={max_soc_resid} n_failed={n_failed}")
    print("saved oracle_red_team_results.csv and oracle_red_team_report.md")


if __name__ == "__main__":
    main()
