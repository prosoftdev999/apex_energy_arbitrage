"""Causal regret decomposition of policy_v11 (Section 2-4 of the request).

Offline diagnostic ONLY -- may access recorded realized future RT prices,
which a submitted policy never can. Single process throughout.

Oracle ladder (each differs from v11 in exactly one stated component):
  O0  policy_v11 exactly
  O1  v11 allocator, but Vc for the NEXT 1 step uses the REAL realized price
      (rolling K-step lookahead, K=1), reverting to v11's standard
      DA-quadrature Vc as the terminal boundary at t+K
  O4  same, K=4
  OF  full perfect-foresight, network-feasible LP (reuses oracle_certified's
      already-validated oracle2 + replay_through_production)
  OA  v11's own reward+Vc landscape, but forced through the exact LP
      (already what v11 does for B<=60 via the keep-best gate -- this oracle
      quantifies whether the DISCRETE-GRID/repair path ever loses value
      relative to the LP it's compared against)
  OT  v11 normally, except the LAST 16 steps' continuation value is rebuilt
      from REAL realized prices instead of DA-quadrature (isolates terminal
      valuation)
  OV  the ENTIRE continuation value is rebuilt from REAL realized prices for
      all T steps (isolates value-function approximation error), but v11's
      OWN allocator (grid+repair+coordinate-improve+candidate-D+LP) is used
      unchanged (isolates value-function error from allocator error)

All oracles use the SAME real production replay (challenge.take_step) for
final profit -- no internal objective values are compared, only replayed
quality, per instruction.
"""
import sys
from pathlib import Path
import random

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import policy_v11 as v11
import oracle_certified as oc

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE
_TERMINAL_H = 16


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def _record_real_price_trajectory(ch):
    """Record the realized RT-price trajectory using a zero-action rollout --
    valid because price/congestion generation never depends on the chosen
    action (confirmed via source trace of take_step: next_rt_prices depends
    only on the pre-drawn next_seed and state.exogenous_injections, both
    action-independent)."""
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    B = ch.num_batteries
    T = ch.num_steps
    rt = np.zeros((T, ch.network.num_nodes))
    rt[0] = np.asarray(state.rt_prices)
    zero = [0.0] * B
    for t in range(T):
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, zero, NextRTPricesGenerate(next_seed))
        if t + 1 < T:
            rt[t + 1] = np.asarray(state.rt_prices)
    return rt


def _build_real_price_value_function(ba, real_price_node, t_start, t_end, S, V_boundary=None):
    """Backward induction identical in structure to v11._build_da_value_function
    but using REAL (deterministic, already-realized) prices for t in
    [t_start, t_end) instead of the causal DA-quadrature model. V_boundary is
    the value at t_end (defaults to zeros, i.e. true terminal); for splicing
    into an existing V_all (OT), pass V_all[t_end] as the boundary so the
    two pieces are consistent at the join.
    """
    B = ba["B"]
    smin, smax = ba["smin"], ba["smax"]
    etac, etad = ba["etac"], ba["etad"]
    pchg, pdis, cap, node = ba["pchg"], ba["pdis"], ba["cap"], ba["node"]
    soc_grid = smin[:, None] + (smax - smin)[:, None] * np.linspace(0.0, 1.0, S)[None, :]
    smin3, smax3 = smin[:, None, None], smax[:, None, None]
    etac3, etad3 = etac[:, None, None], etad[:, None, None]
    cap3 = cap[:, None, None]
    b_idx = np.arange(B)[:, None, None]

    lb, ub = v11._action_bounds_np(soc_grid, smin[:, None], smax[:, None], etac[:, None], etad[:, None],
                                    pchg[:, None], pdis[:, None], v11._DT)
    u = v11._candidate_grid(lb, ub)
    abs_u = np.abs(u)
    new_soc = v11._apply_action_np(u, soc_grid[:, :, None], smin3, smax3, etac3, etad3, v11._DT)
    span = np.maximum(smax3 - smin3, v11._EPS)
    idx_frac = np.clip((new_soc - smin3) / span * (S - 1), 0.0, S - 1 - 1e-06)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    cost_term = v11._KAPPA_TX * abs_u * v11._DT + v11._KAPPA_DEG * (abs_u * v11._DT / cap3) ** v11._BETA_DEG

    n = t_end - t_start
    V_all_local = np.zeros((n + 1, B, S))
    V_all_local[n] = V_boundary if V_boundary is not None else np.zeros((B, S))
    V_next = V_all_local[n]
    for k in range(n - 1, -1, -1):
        t = t_start + k
        Vc = V_next[b_idx, i0] * (1.0 - w) + V_next[b_idx, i1] * w
        price_t = real_price_node[t][:, None, None]
        reward = u * price_t * v11._DT - cost_term
        total = reward + Vc
        V_here = np.max(total, axis=2)
        V_all_local[k] = V_here
        V_next = V_here
    return V_all_local  # index 0 == t_start, index n == t_end


def _v11_allocate(view, state, ba, V_next, ptdf_np, limits_np, slack, S, B):
    """Exact replica of policy_v11.policy()'s allocation pipeline (grid
    argmax -> repair -> coordinate-improve -> CONGESTED candidate D -> LP),
    taking V_next as an explicit parameter instead of the module cache --
    lets every oracle reuse the IDENTICAL, already-validated allocator while
    only the continuation value differs."""
    soc = np.array(state.socs, dtype=float)
    bounds = np.array(state.action_bounds, dtype=float)
    lb, ub = bounds[:, 0], bounds[:, 1]
    node = ba["node"]
    cap = ba["cap"]
    smin, smax = ba["smin"], ba["smax"]
    etac, etad = ba["etac"], ba["etad"]
    price = np.array(state.rt_prices, dtype=float)[node]

    u = v11._candidate_grid(lb, ub)
    abs_u = np.abs(u)
    reward = u * price[:, None] * v11._DT - v11._KAPPA_TX * abs_u * v11._DT \
        - v11._KAPPA_DEG * (abs_u * v11._DT / cap[:, None]) ** v11._BETA_DEG
    new_soc = v11._apply_action_np(u, soc[:, None], smin[:, None], smax[:, None], etac[:, None], etad[:, None], v11._DT)
    span = np.maximum(smax - smin, v11._EPS)[:, None]
    idx_frac = np.clip((new_soc - smin[:, None]) / span * (S - 1), 0.0, S - 1 - 1e-06)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    rows = np.arange(B)[:, None]
    Vc = V_next[rows, i0] * (1.0 - w) + V_next[rows, i1] * w
    total = reward + Vc
    idx = np.argmax(total, axis=1)
    action = u[np.arange(B), idx]

    exo = np.asarray(state.exogenous_injections, dtype=float)
    flows = v11._flows_np(exo, action, node, slack, ptdf_np)
    if not v11._feasible(flows, limits_np):
        action, idx, ok, moved, curtailed = v11._value_aware_repair(u, total, idx, node, ptdf_np, limits_np, exo, slack)
        if not ok:
            action, _, _ = v11._safe_network_repair(np.array(action, dtype=float), node, ptdf_np, limits_np, exo, slack)
            flows = v11._flows_np(exo, action, node, slack, ptdf_np)
            if not v11._feasible(flows, limits_np):
                action = np.zeros(B)
            return np.clip(action, lb, ub)

    action, idx = v11._coordinate_improve(u, total, idx, node, ptdf_np, limits_np, exo, slack)
    flows = v11._flows_np(exo, action, node, slack, ptdf_np)
    if not v11._feasible(flows, limits_np):
        action, _, _ = v11._safe_network_repair(action, node, ptdf_np, limits_np, exo, slack)
        flows = v11._flows_np(exo, action, node, slack, ptdf_np)
        if not v11._feasible(flows, limits_np):
            action = np.zeros(B)
    action = np.clip(action, lb, ub)

    if v11._CONGESTED_MIN_B <= B <= v11._CONGESTED_MAX_B:
        idx_u = np.argmax(total, axis=1)
        alt_action, _ = v11._constructive_from_grid(u, total, idx_u, node, ptdf_np, limits_np, exo, slack)
        alt_action = np.clip(alt_action, lb, ub)
        alt_flows = v11._flows_np(exo, alt_action, node, slack, ptdf_np)
        if v11._feasible(alt_flows, limits_np):
            val_std = v11._evaluate_total_value(action, price, cap, soc, smin, smax, etac, etad, V_next, S)
            val_alt = v11._evaluate_total_value(alt_action, price, cap, soc, smin, smax, etac, etad, V_next, S)
            if val_alt > val_std:
                action = alt_action

    if B <= v11._LP_MAX_B:
        lp_action = v11._lp_joint_solve(soc, price, cap, smin, smax, etac, etad, lb, ub, V_next, S,
                                         node, ptdf_np, limits_np, exo, slack)
        if lp_action is not None:
            lp_action = np.clip(lp_action, lb, ub)
            lp_flows = v11._flows_np(exo, lp_action, node, slack, ptdf_np)
            if v11._feasible(lp_flows, limits_np):
                val_std = v11._evaluate_total_value(action, price, cap, soc, smin, smax, etac, etad, V_next, S)
                val_lp = v11._evaluate_total_value(lp_action, price, cap, soc, smin, smax, etac, etad, V_next, S)
                if val_lp > val_std:
                    action = lp_action
    return action


def run_oracle(ch, ba, oracle_name, real_price_all, sigma_hat, K=None):
    """Runs a full episode replay for the named oracle, returns final profit."""
    T = ch.num_steps
    B = ba["B"]
    net = ch.network
    ptdf_np = np.asarray(net.ptdf, dtype=float)
    limits_np = np.asarray(net.flow_limits, dtype=float)
    slack = net.slack_bus
    node = ba["node"]
    S = v11._S

    V_all_std = v11._build_da_value_function(ch, ba, sigma_hat=sigma_hat)

    if oracle_name == "OF":
        rt_traj = real_price_all
        exo_traj = np.asarray(ch.exogenous_injections, dtype=float)
        ba_oracle = oc._battery_arrays(ch)  # oracle_certified's own ba dict includes soc_init
        o2_profit, o2_sched, o2_status, _ = oc.oracle2(ch, rt_traj, exo_traj, ba_oracle)
        if o2_sched is None:
            return None
        sched_per_step = oc._reshape_to_per_step(o2_sched, T, B)
        replay_profit, bviol, fviol, err, max_clip = oc.replay_through_production(ch, sched_per_step, rt_traj)
        return replay_profit

    if oracle_name == "OV":
        V_all_use = _build_real_price_value_function(ba, real_price_all[:, node], 0, T, S)
    elif oracle_name == "OT":
        V_all_use = V_all_std.copy()
        boundary = V_all_std[T - _TERMINAL_H] if T - _TERMINAL_H >= 0 else None
        # boundary should be at t_end = T (true terminal, zeros) -- rebuild
        # the last _TERMINAL_H entries with real prices, true zero terminal.
        real_tail = _build_real_price_value_function(ba, real_price_all[:, node], T - _TERMINAL_H, T, S)
        V_all_use[T - _TERMINAL_H:T + 1] = real_tail
    elif oracle_name in ("O1", "O2", "O4", "O8", "OA", "O0"):
        V_all_use = V_all_std
    else:
        raise ValueError(oracle_name)

    K_MAP = {"O1": 1, "O2": 2, "O4": 4, "O8": 8}

    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    for t in range(T):
        if oracle_name in ("OT", "OV", "OA", "O0"):
            V_next = V_all_use[t + 1] if t + 1 < V_all_use.shape[0] else np.zeros((B, S))
        elif oracle_name in K_MAP:
            # CORRECTED semantics: "O_K may know realized prices[t+1 : t+1+K]".
            # The continuation value used for the decision at t represents
            # value FROM t+1 onward, so the informed lookahead must itself
            # START at t+1 (not t -- price[t] is already causal/known to
            # v11 via its own immediate reward, so re-deriving it via a
            # real-price recursion at t_start=t is a no-op by construction,
            # which was exactly the prior bug). Build the K-step real-price
            # value function with t_start=t+1, t_end=t+1+K, boundary at
            # t_end from the STANDARD causal model, and take index 0 of the
            # result (= the value AT t_start=t+1) as V_next for decision t.
            k = K_MAP[oracle_name]
            t1 = t + 1
            if t1 >= T:
                V_next = np.zeros((B, S))
            else:
                t_end = min(t1 + k, T)
                boundary = V_all_std[t_end]
                V_k = _build_real_price_value_function(ba, real_price_all[:, node], t1, t_end, S, V_boundary=boundary)
                V_next = V_k[0]
        else:
            raise ValueError(oracle_name)
        action = _v11_allocate(view, state, ba, V_next, ptdf_np, limits_np, slack, S, B)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, action.tolist(), NextRTPricesGenerate(next_seed))
    return state.total_profit


def main(seeds, scenario, nonces, oracle_names, out_csv):
    import csv
    rows = []
    for seed in seeds:
        for nonce in nonces:
            s = seed_from_master_nonce(seed, nonce)
            ch = Challenge.generate_instance(s, Track(s=scenario))
            _, base_profit = ch.compute_baseline()
            ba = v11._battery_arrays(ch)
            real_price_all = _record_real_price_trajectory(ch)
            da0 = np.asarray(ch.market.day_ahead_prices[0], dtype=float)
            rt0 = real_price_all[0]
            rel_dev0 = (rt0 - da0) / np.maximum(np.abs(da0), 1.0)
            sigma_hat = v11._sigma_from_rel_dev(rel_dev0) if ba["B"] <= v11._QUAD_MAX_B else 0.0

            row = dict(seed=seed, scenario=scenario.name, nonce=nonce, baseline=base_profit)
            for oname in oracle_names:
                v11._CACHE.clear()
                profit = run_oracle(ch, ba, oname, real_price_all, sigma_hat)
                if profit is None:
                    row[oname] = None
                    continue
                q = max(-10.0, min((profit - base_profit) / (base_profit + 1e-6), 10.0))
                row[oname] = q
                print(f"seed={seed} {scenario.name} nonce={nonce} {oname}: profit={profit:.1f} q={q:+.4f}", flush=True)
            rows.append(row)

    with open(out_csv, "w", newline="") as f:
        fieldnames = ["seed", "scenario", "nonce", "baseline"] + oracle_names
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--scenario", type=str, default="BASELINE")
    ap.add_argument("--nonces", type=int, nargs="+", required=True)
    ap.add_argument("--oracles", type=str, nargs="+", default=["O0", "O1", "O2", "O4", "O8", "OF", "OA", "OT", "OV"])
    ap.add_argument("--out", type=str, default="regret_audit_results.csv")
    args = ap.parse_args()
    scen = getattr(Scenario, args.scenario)
    main(args.seeds, scen, args.nonces, args.oracles, args.out)
