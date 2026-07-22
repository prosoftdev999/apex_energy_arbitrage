"""Phase 3: fresh policy architecture --
    V_joint(s) = sum_i V_i(soc_i) + C_theta(phi(s))
implemented via candidate-generation + exact production-value evaluation
(Option C from spec) rather than linearizing C_theta into the LP directly.

This is an independent implementation: it does not import or call
best_candidate.py/policy_v11.py's decision logic. It reuses only the fixed
physics (action bounds, SOC transition, PTDF flows) via fvi_common.py,
which is itself a fresh reimplementation, not a copy.

Causality note (important, and non-obvious): C_theta(phi(next_state)) is
evaluated on the state that WOULD result from a candidate action -- next_soc
is deterministic given the action (legal), and the network/DA-price
features of the next state are public for the full horizon (legal), but the
"current RT-vs-DA residual" feature used during TRAINING (fvi_generate_data
computed phi(t) using the REAL, already-revealed rt_prices at time t) cannot
be evaluated for time t+1 at decision time t, because rt_{t+1} has not been
revealed yet -- using it would be illegal foresight, and per the project's
own rule a state-only constant can't change the argmax anyway if it were
action-independent. Fix: substitute the causal expectation for the unknown
next-step term -- E[RT_{t+1}] = DA_{t+1} exactly (mu_bias=0, verified from
market.py source this session), so rt_da_resid_now is passed as an exact
zero (unbiased plug-in, not a guess); the already-revealed CURRENT price
(rt_t, legitimately known at decision time) is passed as "rt_prev" for the
next state's perspective, which is exactly correct and legal.

Candidate generation (per step):
  A. V_i-only per-battery-independent argmax (same class of mechanism this
     session already validates), safety-repaired if infeasible.
  B. Exact LP joint solve using V_i only (continuous, PTDF-constrained).
  C. "Congestion-relief nudge": starting from A, scale down the actions of
     the batteries most exposed (by |PTDF|) to the currently most-utilized
     line by 10%, re-repaired if needed -- explicitly tests whether the
     learned correction rewards proactive network-aware behavior beyond
     what V_i/LP already choose.
Each feasible candidate's next state is computed exactly, phi(next_state)
extracted, C_theta(phi) evaluated, and the candidate maximizing
[reward + V_i(next_soc) + C_theta(phi(next_state))] is chosen -- this
satisfies "the correction must influence the optimizer through the
candidate next state," not a state-only additive constant.

Safe gate: candidate A (the pure V_i-only baseline) is always among the
evaluated candidates, so the corrected choice can never score worse than
the uncorrected baseline on the SAME objective (reward + V_i + C_theta) --
matching the established safe-gate discipline throughout this project.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

_SANDBOX_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SANDBOX_DIR))
import fvi_common as fc

_DT = fc._DT
_KAPPA_TX = fc._KAPPA_TX
_KAPPA_DEG = fc._KAPPA_DEG
_BETA_DEG = fc._BETA_DEG
_EPS = fc._EPS
_EPS_FLOW = 1e-6

# Embedded fitted correction model (Phase 3 step 3-5, sandbox/fvi_train.py,
# grouped leave-one-seed-out CV R2=+0.8066 on the winning linear_interactions
# class -- see sandbox/fvi_model.json for full CV results across all three
# tested model classes). Embedded directly, no external file dependency at
# evaluation time.
_FVI_MODEL_JSON = r'''
{"theta": [-45029.29418723988, -13864.422868570637, -170.75238102602646, -378.68730004089485, 1835.500390772098, 14079.370224566628, 4238.990660217009, 6863.78322799818, 6030.42374268868, 1787.7257585933064, 37126.025104015294, -69238.84752382753, -976.091605908516, 4887.949413631544, -58.27652797624802, 11421.299248909278, 58.27652798059712, -324.23801955776577, -3901.40245874133, 346.53665023660153, 39.1304858106397, -2671.3034588998985, 9.22296967556584, -675.3035453493146, -555.2275670459774, 1876.39662914622, 18243.386119573894, 7297.019859148889, 12719.497729834773, -3184.0030184946713, 2485.7938719970093, 2278.01333153634, -26693.78547996573, -2007.8329288864836, -10278.872851098806, 2605.8172281677007, -3516.252876425186, -3572.7181996545846, 47586.98676003521, -849.375623572671, 574.2796088696123, -1569.181146687276, -2019.1309384962387, 3222.6651907381515, 94.3306396838336, -5660.154826797912, -5956.430584243394, -6480.419730395819, 3244.085896917543, 3969.989788302048, 5135.109741790027, -7299.639401107726, 12810.470792813641, 12182.28369427836, -71898.90324754376], "mu": [0.50390625, 4.1348760353165925e-17, 1.879283765641541e-19, 0.4836895998815331, 0.4536291490500472, 0.41548972433169656, 0.10118900121341017, 0.29268364483874787, 0.4143045302268912, 0.5394161183919194, 0.30333116319445047, 0.3846159580740096, 0.4781681991022296, 0.2571165611454144, 0.15001562260797333, 0.21421966692815633, 0.8499843773922996, 0.8499843773922995, 0.003346353691804193, -0.02651588044016632, -0.005813123270720643, 2.3727887059220314, 0.05147243845984921, 0.05180760142475512, 0.4111299178135281, 0.3855771676846877], "sd": [0.28866828355106755, 0.7071067821865475, 0.7071067821865475, 0.06690475665107982, 0.10776882320438637, 0.2561838376505732, 0.0454704617661037, 0.25594972412601036, 0.2625803066360285, 0.265110726721736, 0.355989550036313, 0.2602828718828515, 0.4281974430651198, 0.06566770797058248, 0.0010345228591154656, 0.06624209958643983, 0.0010345228591154734, 0.00036546765645416806, 0.34090935544328665, 1.095600603733884, 0.3656547731705973, 1.2406520583037464, 0.17264883180706728, 0.17479235753344527, 0.05687571951088222, 0.09160246770290466], "inter_idx_pairs": [[0, 10], [0, 15], [0, 1], [0, 6], [0, 25], [0, 4], [0, 11], [10, 15], [10, 1], [10, 6], [10, 25], [10, 4], [10, 11], [15, 1], [15, 6], [15, 25], [15, 4], [15, 11], [1, 6], [1, 25], [1, 4], [1, 11], [6, 25], [6, 4], [6, 11], [25, 4], [25, 11], [4, 11]], "lam": 10.0, "kind": "linear_interactions"}
'''
_FVI_MODEL = json.loads(_FVI_MODEL_JSON)
_FVI_THETA = np.array(_FVI_MODEL["theta"])
_FVI_MU = np.array(_FVI_MODEL["mu"])
_FVI_SD = np.array(_FVI_MODEL["sd"])
_FVI_INTER = _FVI_MODEL["inter_idx_pairs"]


def _c_theta(phi):
    xs = (phi - _FVI_MU) / _FVI_SD
    inter = np.array([xs[a] * xs[b] for a, b in _FVI_INTER])
    xfull = np.concatenate([xs, inter, [1.0]])
    return float(xfull @ _FVI_THETA)


_CACHE = {}
_ENABLED_B = {10, 20, 60}  # BASELINE, CONGESTED, DENSE -- same scope as the
# training data (MULTIDAY/CAPSTONE excluded: independent oracle audit shows
# them already within 0.03-0.08 of the absolute +10 ceiling, no material
# headroom for any architecture there).
_LP_BREAKPOINTS = 15


def _episode_signature(challenge):
    b0 = challenge.batteries[0]
    return (challenge.num_batteries, challenge.network.num_nodes, challenge.network.num_lines,
            challenge.num_steps, round(b0.capacity_mwh, 6), b0.node,
            round(challenge.market.day_ahead_prices[0][0], 6))


def _flows_np(exo, action, node, slack, ptdf):
    inj = np.array(exo, dtype=float)
    inj[slack] = 0.0
    np.add.at(inj, node, action)
    inj[slack] = -(inj.sum() - inj[slack])
    return ptdf @ inj


def _feasible(flows, limits):
    return not np.any(np.abs(flows) - limits > _EPS_FLOW * limits)


def _safe_repair(action, node, ptdf, limits, exo, slack, max_iters=64):
    """Fresh reimplementation of the proportional-scale-down repair pattern
    already used (and legally required, since take_step strictly rejects
    infeasible actions -- verified from source) throughout this project's
    baselines and prior policies."""
    action = action.copy()
    for _ in range(max_iters):
        flows = _flows_np(exo, action, node, slack, ptdf)
        viol = np.abs(flows) - limits
        bad = viol > _EPS_FLOW * limits
        if not np.any(bad):
            return action
        l = int(np.argmax(np.where(bad, viol, -np.inf)))
        flow_l, amount = flows[l], viol[l]
        sign = 1.0 if flow_l > 0 else (-1.0 if flow_l < 0 else 0.0)
        if abs(sign) <= 1e-12:
            break
        contrib = ptdf[l, node] * action
        signed = sign * contrib
        worsening = signed > 1e-12
        strength = float(np.sum(signed[worsening]))
        if not np.any(worsening) or strength <= 1e-12:
            break
        keep = max(0.0, min(1.0 - amount / strength, 1.0))
        action[worsening] *= keep
    flows = _flows_np(exo, action, node, slack, ptdf)
    if _feasible(flows, limits):
        return action
    zero = np.zeros_like(action)
    if not _feasible(_flows_np(exo, zero, node, slack, ptdf), limits):
        return zero
    low, high = 0.0, 1.0
    for _ in range(32):
        mid = 0.5 * (low + high)
        if _feasible(_flows_np(exo, mid * action, node, slack, ptdf), limits):
            low = mid
        else:
            high = mid
    return low * action


def _lp_joint_v_only(soc, price, ba, lb, ub, V_next, n_bp=_LP_BREAKPOINTS):
    B = ba["B"]
    node, ptdf, limits, exo, slack = ba["node"], ba["ptdf"], ba["limits"], ba["exo"], ba["slack"]
    cap, smin, smax, etac, etad = ba["cap"], ba["smin"], ba["smax"], ba["etac"], ba["etad"]
    S = V_next.shape[1]
    L = ptdf.shape[0]
    nvar = 2 * B
    bounds = [(float(lb[i]), float(ub[i])) for i in range(B)] + [(None, None)] * B
    c_obj = np.zeros(nvar)
    c_obj[B:] = -1.0
    A_ub_rows, b_ub = [], []
    for i in range(B):
        lo, hi = float(lb[i]), float(ub[i])
        us = np.linspace(lo, hi, n_bp)
        abs_us = np.abs(us)
        reward = us * price[i] * _DT - _KAPPA_TX * abs_us * _DT - _KAPPA_DEG * (abs_us * _DT / cap[i]) ** _BETA_DEG
        c_leg = np.maximum(-us, 0.0)
        d_leg = np.maximum(us, 0.0)
        new_soc = np.clip(soc[i] + etac[i] * c_leg * _DT - d_leg * _DT / etad[i], smin[i], smax[i])
        span_i = max(smax[i] - smin[i], 1e-12)
        frac = np.clip((new_soc - smin[i]) / span_i * (S - 1), 0.0, S - 1 - 1e-6)
        j0 = frac.astype(np.intp)
        j1 = np.minimum(j0 + 1, S - 1)
        jw = frac - j0
        Vc = V_next[i, j0] * (1.0 - jw) + V_next[i, j1] * jw
        fs = reward + Vc
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
    base_flow = _flows_np(exo, np.zeros(B), node, slack, ptdf)
    for l in range(L):
        row_hi = np.zeros(nvar)
        row_lo = np.zeros(nvar)
        for i in range(B):
            row_hi[i] = ptdf[l, node[i]]
            row_lo[i] = -ptdf[l, node[i]]
        A_ub_rows.append(row_hi); b_ub.append(float(limits[l] - base_flow[l]))
        A_ub_rows.append(row_lo); b_ub.append(float(limits[l] + base_flow[l]))
    try:
        res = linprog(c_obj, A_ub=np.array(A_ub_rows), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    return res.x[:B]


def _evaluate_candidate(action, price, soc, ba, V_next, S, t, T, da_all, exo_all, rt_now_actual,
                         use_correction=True):
    """reward + V_i(next_soc) [+ C_theta(phi(next_state)) if use_correction],
    using the causal plug-in for the next state's unknown RT price (see
    module docstring). use_correction=False reproduces the pure V_i-only
    decision rule -- used for out-of-scope scenarios (MULTIDAY/CAPSTONE),
    where the fitted correction was never trained and is out of scope."""
    node = ba["node"]
    abs_a = np.abs(action)
    reward = float(np.sum(action * price * _DT - _KAPPA_TX * abs_a * _DT
                           - _KAPPA_DEG * (abs_a * _DT / ba["cap"]) ** _BETA_DEG))
    next_soc = fc.apply_action_np(action, soc, ba)
    v_i = _v_i_at(V_next, next_soc, ba, S)
    if not use_correction:
        return reward + v_i, reward, v_i, 0.0
    if t + 1 < T:
        da_now_next = float(da_all[t + 1].mean())
        phi_next = fc.extract_features(t + 1, T, next_soc, ba, node, da_all, exo_all,
                                        ba["ptdf"], ba["limits"], ba["slack"],
                                        rt_now=da_now_next, rt_prev=rt_now_actual, da_now=da_now_next)
        correction = _c_theta(phi_next)
    else:
        correction = 0.0
    return reward + v_i + correction, reward, v_i, correction


def _v_i_at(V_next, soc, ba, S):
    smin, smax = ba["smin"], ba["smax"]
    span = np.maximum(smax - smin, _EPS)
    idx_frac = np.clip((soc - smin) / span * (S - 1), 0.0, S - 1 - 1e-6)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    rows = np.arange(len(soc))
    return float(np.sum(V_next[rows, i0] * (1.0 - w) + V_next[rows, i1] * w))


def policy(challenge, state):
    B = challenge.num_batteries
    if B == 0:
        return []

    key = id(challenge)
    sig = _episode_signature(challenge)
    entry = _CACHE.get(key)
    if entry is None or entry.get("sig") != sig:
        if len(_CACHE) > 4:
            _CACHE.clear()
        ba = fc.battery_arrays_from_batteries(challenge.batteries)
        net = challenge.network
        ba["ptdf"] = np.asarray(net.ptdf, dtype=float)
        ba["limits"] = np.asarray(net.flow_limits, dtype=float)
        ba["slack"] = net.slack_bus
        da_all = np.asarray(challenge.market.day_ahead_prices, dtype=float)
        exo_all = np.asarray(challenge.exogenous_injections, dtype=float)
        # V_i (separable value function) is built for EVERY scenario -- it's
        # the established, validated class of mechanism, not what's being
        # tested. Only the FITTED CORRECTION C_theta is scoped to
        # _ENABLED_B (the scenarios it was trained on / where the
        # independent oracle audit found material headroom). Out-of-scope
        # scenarios (MULTIDAY/CAPSTONE) still get a real, reasonable action
        # via the V_i-only candidates -- NOT a zero/no-op action, which
        # would earn zero profit against a positive baseline and produce a
        # meaningless, artificially bad score unrelated to any research
        # question (caught via an initial run that showed quality=-1.0000
        # on both, i.e. a pure own-goal, not a finding).
        correction_active = B in _ENABLED_B
        V_all = fc.build_separable_value_function(da_all, ba["node"], ba, B)
        entry = dict(ba=ba, da_all=da_all, exo_all=exo_all, V_all=V_all, sig=sig, active=correction_active)
        _CACHE[key] = entry

    ba = entry["ba"]
    da_all, exo_all = entry["da_all"], entry["exo_all"]
    node, ptdf, limits, slack = ba["node"], ba["ptdf"], ba["limits"], ba["slack"]
    t = state.time_step
    T = challenge.num_steps
    soc = np.array(state.socs, dtype=float)
    bounds = np.array(state.action_bounds, dtype=float)
    lb, ub = bounds[:, 0], bounds[:, 1]
    price = np.array(state.rt_prices, dtype=float)[node]
    ba["exo"] = exo_all[t]
    exo = exo_all[t]

    V_all = entry["V_all"]
    S = V_all.shape[2]
    V_next = V_all[t + 1] if t + 1 < V_all.shape[0] else np.zeros((B, S))

    # Candidate A: V_i-only per-battery argmax over a small grid
    k = 3
    fracs = np.linspace(1.0 / k, 1.0, k)
    neg = (-lb)[..., None] * fracs[::-1]
    pos = ub[..., None] * fracs
    zero = np.zeros(lb.shape + (1,))
    u = np.concatenate([neg, zero, pos], axis=-1)
    abs_u = np.abs(u)
    reward_grid = u * price[:, None] * _DT - _KAPPA_TX * abs_u * _DT - _KAPPA_DEG * (abs_u * _DT / ba["cap"][:, None]) ** _BETA_DEG
    new_soc = fc.apply_action_np(u, soc[:, None], dict(ba, smin=ba["smin"][:, None], smax=ba["smax"][:, None],
                                                        etac=ba["etac"][:, None], etad=ba["etad"][:, None]))
    span = np.maximum(ba["smax"] - ba["smin"], _EPS)[:, None]
    idx_frac = np.clip((new_soc - ba["smin"][:, None]) / span * (S - 1), 0.0, S - 1 - 1e-6)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    rows = np.arange(B)[:, None]
    Vc = V_next[rows, i0] * (1.0 - w) + V_next[rows, i1] * w
    total = reward_grid + Vc
    idx = np.argmax(total, axis=1)
    candidate_A = u[np.arange(B), idx]
    flows = _flows_np(exo, candidate_A, node, slack, ptdf)
    if not _feasible(flows, limits):
        candidate_A = _safe_repair(candidate_A, node, ptdf, limits, exo, slack)
    candidate_A = np.clip(candidate_A, lb, ub)

    candidates = [candidate_A]

    # Candidate B: exact LP joint solve, V_i only
    lp_action = _lp_joint_v_only(soc, price, ba, lb, ub, V_next)
    if lp_action is not None:
        lp_action = np.clip(lp_action, lb, ub)
        flows_b = _flows_np(exo, lp_action, node, slack, ptdf)
        if not _feasible(flows_b, limits):
            lp_action = _safe_repair(lp_action, node, ptdf, limits, exo, slack)
        candidates.append(np.clip(lp_action, lb, ub))

    # Candidate C: congestion-relief nudge on candidate A -- only generated
    # (and only scored with the fitted correction) for in-scope scenarios;
    # out-of-scope scenarios stick to the plain V_i-only decision between
    # candidates A and B.
    if entry["active"]:
        zero_flows = fc.compute_zero_action_flows(exo, node, ptdf, slack)
        line_util = np.abs(zero_flows) / np.maximum(limits, _EPS)
        worst_line = int(np.argmax(line_util))
        exposure = np.abs(ptdf[worst_line, node])
        if np.max(exposure) > 1e-9:
            exposed = exposure > 0.3 * np.max(exposure)
            candidate_C = candidate_A.copy()
            candidate_C[exposed] *= 0.9
            flows_c = _flows_np(exo, candidate_C, node, slack, ptdf)
            if not _feasible(flows_c, limits):
                candidate_C = _safe_repair(candidate_C, node, ptdf, limits, exo, slack)
            candidates.append(np.clip(candidate_C, lb, ub))

    rt_now_actual = float(np.mean(state.rt_prices))
    best_score, best_action = -np.inf, candidate_A
    for cand in candidates:
        flows = _flows_np(exo, cand, node, slack, ptdf)
        if not _feasible(flows, limits):
            continue
        score, _, _, _ = _evaluate_candidate(cand, price, soc, ba, V_next, S, t, T, da_all, exo_all, rt_now_actual,
                                              use_correction=entry["active"])
        if score > best_score + 1e-6:
            best_score = score
            best_action = cand

    return np.clip(best_action, lb, ub).tolist()
