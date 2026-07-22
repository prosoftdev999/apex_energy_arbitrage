import numpy as np
from scipy.optimize import linprog

_CACHE = {}
_DT = 0.25
_KAPPA_TX = 0.25
_KAPPA_DEG = 1.0
_BETA_DEG = 2.0
_EPS_FLOW = 1e-06
_EPS = 1e-12
_S = 21
_NUM_FRACS = 3
_MAX_REPAIR_ITERS = 300
_COORD_SWEEPS = 2

_BENCH_STATS = {"repair_count": 0, "curtailed_mwh": 0.0, "zero_fallback_count": 0}


def _action_bounds_np(soc, smin, smax, etac, etad, pchg, pdis, dt):
    headroom = np.maximum(smax - soc, 0.0)
    avail = np.maximum(soc - smin, 0.0)
    max_charge_from_soc = np.where(etac > 0, headroom / np.maximum(etac * dt, _EPS), 0.0)
    max_discharge_from_soc = np.where(etad > 0, avail * etad / dt, 0.0)
    max_charge = np.clip(np.minimum(max_charge_from_soc, pchg), 0.0, None)
    max_discharge = np.clip(np.minimum(max_discharge_from_soc, pdis), 0.0, None)
    return (-max_charge, max_discharge)


def _apply_action_np(u, soc, smin, smax, etac, etad, dt):
    c = np.maximum(-u, 0.0)
    d = np.maximum(u, 0.0)
    new_soc = soc + etac * c * dt - d * dt / etad
    return np.clip(new_soc, smin, smax)


def _candidate_grid(lb, ub, k=_NUM_FRACS):
    fracs = np.linspace(1.0 / k, 1.0, k)
    neg = lb[..., None] * fracs[::-1]
    pos = ub[..., None] * fracs
    zero = np.zeros(lb.shape + (1,))
    return np.concatenate([neg, zero, pos], axis=-1)


def _battery_arrays(challenge):
    caps = np.array([b.capacity_mwh for b in challenge.batteries], dtype=float)
    pchg = np.array([b.power_charge_mw for b in challenge.batteries], dtype=float)
    pdis = np.array([b.power_discharge_mw for b in challenge.batteries], dtype=float)
    etac = np.array([b.efficiency_charge for b in challenge.batteries], dtype=float)
    etad = np.array([b.efficiency_discharge for b in challenge.batteries], dtype=float)
    smin = np.array([b.soc_min_mwh for b in challenge.batteries], dtype=float)
    smax = np.array([b.soc_max_mwh for b in challenge.batteries], dtype=float)
    node = np.array([b.node for b in challenge.batteries], dtype=np.intp)
    return dict(cap=caps, pchg=pchg, pdis=pdis, etac=etac, etad=etad, smin=smin, smax=smax, node=node, B=len(caps))


def _build_da_value_function(challenge, ba, S=_S):
    T = challenge.num_steps
    B = ba["B"]
    V_all = np.zeros((T + 1, B, S))
    smin, smax = ba["smin"], ba["smax"]
    etac, etad = ba["etac"], ba["etad"]
    pchg, pdis, cap, node = ba["pchg"], ba["pdis"], ba["cap"], ba["node"]
    soc_grid = smin[:, None] + (smax - smin)[:, None] * np.linspace(0.0, 1.0, S)[None, :]
    da = np.asarray(challenge.market.day_ahead_prices, dtype=float)
    price_forecast = da[:, node]
    smin2, smax2 = smin[:, None], smax[:, None]
    etac2, etad2 = etac[:, None], etad[:, None]
    pchg2, pdis2 = pchg[:, None], pdis[:, None]
    smin3, smax3 = smin[:, None, None], smax[:, None, None]
    etac3, etad3 = etac[:, None, None], etad[:, None, None]
    cap3 = cap[:, None, None]
    b_idx = np.arange(B)[:, None, None]

    lb, ub = _action_bounds_np(soc_grid, smin2, smax2, etac2, etad2, pchg2, pdis2, _DT)
    u = _candidate_grid(lb, ub)
    abs_u = np.abs(u)
    new_soc = _apply_action_np(u, soc_grid[:, :, None], smin3, smax3, etac3, etad3, _DT)
    span = np.maximum(smax3 - smin3, _EPS)
    idx_frac = np.clip((new_soc - smin3) / span * (S - 1), 0.0, S - 1 - 1e-06)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    cost_term = _KAPPA_TX * abs_u * _DT + _KAPPA_DEG * (abs_u * _DT / cap3) ** _BETA_DEG

    V_next = V_all[T]
    for t in range(T - 1, -1, -1):
        Vc = V_next[b_idx, i0] * (1.0 - w) + V_next[b_idx, i1] * w
        price_t = price_forecast[t][:, None, None]
        reward = u * price_t * _DT - cost_term
        total = reward + Vc
        V_here = np.max(total, axis=2)
        V_all[t] = V_here
        V_next = V_here
    return V_all


def _episode_signature(challenge):
    b0 = challenge.batteries[0]
    return (
        challenge.num_batteries, challenge.network.num_nodes, challenge.network.num_lines,
        challenge.num_steps, round(b0.capacity_mwh, 6), b0.node,
        round(challenge.market.day_ahead_prices[0][0], 6),
    )


def _injections_np(exo, action_arr, node_arr, slack):
    inj = np.array(exo, dtype=float)
    inj[slack] = 0.0
    np.add.at(inj, node_arr, action_arr)
    total = inj.sum() - inj[slack]
    inj[slack] = -total
    return inj


def _flows_np(exo, action_arr, node_arr, slack, ptdf_np):
    return ptdf_np @ _injections_np(exo, action_arr, node_arr, slack)


def _feasible(flows, limits):
    viol = np.abs(flows) - limits
    return not np.any(viol > _EPS_FLOW * limits)


def _safe_network_repair(action, node_arr, ptdf_np, limits_np, exo, slack, max_iters=64):
    action = action.copy()
    repaired = False
    curtailed = 0.0
    for _ in range(max_iters):
        flows = _flows_np(exo, action, node_arr, slack, ptdf_np)
        viol = np.abs(flows) - limits_np
        bad = viol > _EPS_FLOW * limits_np
        if not np.any(bad):
            break
        l = int(np.argmax(np.where(bad, viol, -np.inf)))
        flow_l, amount = flows[l], viol[l]
        sign = 1.0 if flow_l > 0 else (-1.0 if flow_l < 0 else 0.0)
        if abs(sign) <= _EPS:
            break
        contrib = ptdf_np[l, node_arr] * action
        signed = sign * contrib
        worsening = signed > _EPS
        strength = float(np.sum(signed[worsening]))
        if not np.any(worsening) or strength <= _EPS:
            break
        keep = max(0.0, min(1.0 - amount / strength, 1.0))
        if abs(1.0 - keep) <= _EPS:
            break
        before = np.abs(action[worsening]).sum()
        action[worsening] *= keep
        curtailed += float(before - np.abs(action[worsening]).sum())
        repaired = True
    flows = _flows_np(exo, action, node_arr, slack, ptdf_np)
    if _feasible(flows, limits_np):
        return action, repaired, curtailed
    zero = np.zeros_like(action)
    if not _feasible(_flows_np(exo, zero, node_arr, slack, ptdf_np), limits_np):
        return zero, True, curtailed
    low, high = 0.0, 1.0
    for _ in range(32):
        mid = 0.5 * (low + high)
        if _feasible(_flows_np(exo, mid * action, node_arr, slack, ptdf_np), limits_np):
            low = mid
        else:
            high = mid
    return low * action, True, curtailed


def _value_aware_repair(u, total, idx, node, ptdf_np, limits_np, exo, slack, max_iters=_MAX_REPAIR_ITERS):
    """Discrete value-aware repair: for the most violated line, rank the batteries
    that worsen it by (value lost) / (congestion relief gained) moving one grid
    step toward zero along their own candidate axis, and apply the cheapest move.
    Uses the FULL local objective (reward + continuation value), not immediate
    reward alone, so a battery with high continuation value is not curtailed
    ahead of one that is only locally attractive."""
    B, A = u.shape
    idx = idx.copy()
    curtailed = 0.0
    any_move = False
    for _ in range(max_iters):
        action = u[np.arange(B), idx]
        flows = _flows_np(exo, action, node, slack, ptdf_np)
        viol = np.abs(flows) - limits_np
        bad = viol > _EPS_FLOW * limits_np
        if not np.any(bad):
            return action, idx, True, any_move, curtailed
        l = int(np.argmax(np.where(bad, viol, -np.inf)))
        flow_l = flows[l]
        sgn = 1.0 if flow_l > 0 else -1.0
        ptdf_l = ptdf_np[l, node]
        contribution = ptdf_l * action
        worsening = (sgn * contribution) > _EPS
        if not np.any(worsening):
            return action, idx, False, any_move, curtailed

        step = np.where(ptdf_l >= 0, -1, 1) * (1 if sgn > 0 else -1)
        new_idx = np.clip(idx + step, 0, A - 1)
        movable = worsening & (new_idx != idx)
        if not np.any(movable):
            return action, idx, False, any_move, curtailed

        wi = np.nonzero(movable)[0]
        new_action_wi = u[wi, new_idx[wi]]
        old_contribution = contribution[wi]
        new_contribution = ptdf_l[wi] * new_action_wi
        relief = sgn * (old_contribution - new_contribution)
        valid = relief > _EPS
        if not np.any(valid):
            return action, idx, False, any_move, curtailed
        wi = wi[valid]
        relief = relief[valid]
        value_loss = total[wi, idx[wi]] - total[wi, new_idx[wi]]
        ratio = value_loss / relief
        pick = wi[np.argmin(ratio)]

        curtailed += float(abs(action[pick] - u[pick, new_idx[pick]]))
        idx[pick] = new_idx[pick]
        any_move = True
    action = u[np.arange(B), idx]
    flows = _flows_np(exo, action, node, slack, ptdf_np)
    return action, idx, _feasible(flows, limits_np), any_move, curtailed


def _coordinate_improve(u, total, idx, node, ptdf_np, limits_np, exo, slack, sweeps=_COORD_SWEEPS):
    B, A = u.shape
    idx = idx.copy()
    for _ in range(sweeps):
        changed = False
        action = u[np.arange(B), idx]
        flows = _flows_np(exo, action, node, slack, ptdf_np)
        for i in range(B):
            cur_val = total[i, idx[i]]
            best_j = idx[i]
            best_gain = 0.0
            ptdf_i = ptdf_np[:, node[i]]
            base_flows = flows - ptdf_i * action[i]
            for j in range(A):
                if j == idx[i]:
                    continue
                trial_flow = base_flows + ptdf_i * u[i, j]
                if not _feasible(trial_flow, limits_np):
                    continue
                gain = total[i, j] - cur_val
                if gain > best_gain + 1e-09:
                    best_gain = gain
                    best_j = j
            if best_j != idx[i]:
                idx[i] = best_j
                action[i] = u[i, best_j]
                flows = base_flows + ptdf_i * action[i]
                changed = True
        if not changed:
            break
    return u[np.arange(B), idx], idx


_CONGESTED_MIN_B = 11
_CONGESTED_MAX_B = 20


def _evaluate_total_value(action, price, cap, soc, smin, smax, etac, etad, V_next, S):
    abs_u = np.abs(action)
    reward = action * price * _DT - _KAPPA_TX * abs_u * _DT - _KAPPA_DEG * (abs_u * _DT / cap) ** _BETA_DEG
    new_soc = _apply_action_np(action, soc, smin, smax, etac, etad, _DT)
    span = np.maximum(smax - smin, _EPS)
    idx_frac = np.clip((new_soc - smin) / span * (S - 1), 0.0, S - 1 - 1e-06)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    rows = np.arange(len(action))
    Vc = V_next[rows, i0] * (1.0 - w) + V_next[rows, i1] * w
    return float(np.sum(reward + Vc))


def _constructive_from_grid(u, total, best_idx, node, ptdf_np, limits_np, exo, slack, max_iters=400):
    """Candidate D: build the joint action UP from zero (always feasible)
    toward each battery's own unconstrained-optimal grid index, greedily
    taking the most valuable feasible single-index step each iteration,
    instead of computing the unconstrained optimum first and curtailing it
    after a violation is detected."""
    B, A = u.shape
    zero_idx = A // 2  # _candidate_grid always places exact zero at the midpoint
    idx = np.full(B, zero_idx, dtype=np.intp)
    action = u[np.arange(B), idx]
    flows = _flows_np(exo, action, node, slack, ptdf_np)

    for _ in range(max_iters):
        best_i, best_gain, best_new_idx, best_flows = -1, 1e-09, None, None
        for i in range(B):
            if idx[i] == best_idx[i]:
                continue
            step = 1 if best_idx[i] > idx[i] else -1
            new_idx_i = idx[i] + step
            trial_flows = flows + ptdf_np[:, node[i]] * (u[i, new_idx_i] - action[i])
            if not _feasible(trial_flows, limits_np):
                continue
            gain = total[i, new_idx_i] - total[i, idx[i]]
            if gain > best_gain:
                best_gain = gain
                best_i, best_new_idx, best_flows = i, new_idx_i, trial_flows
        if best_i < 0:
            break
        idx[best_i] = best_new_idx
        action[best_i] = u[best_i, best_new_idx]
        flows = best_flows

    action, idx = _coordinate_improve(u, total, idx, node, ptdf_np, limits_np, exo, slack)
    return action, idx


_LP_BREAKPOINTS = 21
_LP_MAX_B = 20  # BASELINE (10) and CONGESTED (20) only -- keeps the per-step LP small;
# MULTIDAY/DENSE/CAPSTONE (B=40..100, L=120..300) keep the existing discrete-grid path.


def _lp_joint_solve(soc, price, cap, smin, smax, etac, etad, lb, ub, V_next, S,
                     node, ptdf_np, limits_np, exo, slack, n_bp=_LP_BREAKPOINTS):
    """Exact convex reformulation of the per-step joint decision.

    Each battery's local objective f_i(u) = reward(u) + Vc(soc, u) is provably
    concave (verified via 2nd-difference checks on this exact value table in
    prior work on this project) -- revenue is linear, transaction cost and
    degradation cost are convex, and Vc is a concave piecewise-linear function
    of SOC (hence of u, since SOC evolves affinely in u within each charge/
    discharge branch). A concave piecewise-linear function equals the MIN of
    its own supporting affine segments, so f_i is exactly representable in an
    LP via the standard epigraph trick: introduce v_i and constrain
    v_i <= m_k*u_i + b_k for every breakpoint segment k, then maximize v_i.
    Doing this for all batteries simultaneously, plus the linear PTDF flow
    constraints, turns the ENTIRE joint network-feasible allocation into one
    linear program -- continuous actions, globally optimal for this exact
    convexification, rather than two competing discrete-grid heuristics
    (curtail-down / build-up) operating on a coarse 7-point action grid.
    """
    B = len(soc)
    L = ptdf_np.shape[0]
    nvar = 2 * B  # u_0..u_{B-1}, v_0..v_{B-1}

    bounds = [(float(lb[i]), float(ub[i])) for i in range(B)] + [(None, None)] * B
    c_obj = np.zeros(nvar)
    c_obj[B:] = -1.0  # minimize -sum(v) == maximize sum(v)

    A_ub_rows = []
    b_ub = []
    for i in range(B):
        lo, hi = float(lb[i]), float(ub[i])
        us = np.linspace(lo, hi, n_bp)
        abs_us = np.abs(us)
        reward = us * price[i] * _DT - _KAPPA_TX * abs_us * _DT - _KAPPA_DEG * (abs_us * _DT / cap[i]) ** _BETA_DEG
        c_leg = np.maximum(-us, 0.0)
        d_leg = np.maximum(us, 0.0)
        new_soc = np.clip(soc[i] + etac[i] * c_leg * _DT - d_leg * _DT / etad[i], smin[i], smax[i])
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

    base_flow = _flows_np(exo, np.zeros(B), node, slack, ptdf_np)
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

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    return res.x[:B]


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
        ba = _battery_arrays(challenge)
        V_all = _build_da_value_function(challenge, ba)
        net = challenge.network
        entry = {
            "ba": ba, "V_all": V_all, "sig": sig,
            "ptdf": np.asarray(net.ptdf, dtype=float),
            "limits": np.asarray(net.flow_limits, dtype=float),
        }
        _CACHE[key] = entry
        _BENCH_STATS["repair_count"] = 0
        _BENCH_STATS["curtailed_mwh"] = 0.0
        _BENCH_STATS["zero_fallback_count"] = 0

    ba = entry["ba"]
    V_all = entry["V_all"]
    t = state.time_step
    S = V_all.shape[2]
    V_next = V_all[t + 1] if t + 1 < V_all.shape[0] else np.zeros((B, S))

    soc = np.array(state.socs, dtype=float)
    bounds = np.array(state.action_bounds, dtype=float)
    lb, ub = bounds[:, 0], bounds[:, 1]
    node = ba["node"]
    cap = ba["cap"]
    smin, smax = ba["smin"], ba["smax"]
    etac, etad = ba["etac"], ba["etad"]
    price = np.array(state.rt_prices, dtype=float)[node]

    u = _candidate_grid(lb, ub)
    abs_u = np.abs(u)
    reward = u * price[:, None] * _DT - _KAPPA_TX * abs_u * _DT - _KAPPA_DEG * (abs_u * _DT / cap[:, None]) ** _BETA_DEG
    new_soc = _apply_action_np(u, soc[:, None], smin[:, None], smax[:, None], etac[:, None], etad[:, None], _DT)
    span = np.maximum(smax - smin, _EPS)[:, None]
    idx_frac = np.clip((new_soc - smin[:, None]) / span * (S - 1), 0.0, S - 1 - 1e-06)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    rows = np.arange(B)[:, None]
    Vc = V_next[rows, i0] * (1.0 - w) + V_next[rows, i1] * w
    total = reward + Vc

    idx = np.argmax(total, axis=1)
    action = u[np.arange(B), idx]

    net = challenge.network
    exo = np.asarray(state.exogenous_injections, dtype=float)
    ptdf_np, limits_np, slack = entry["ptdf"], entry["limits"], net.slack_bus
    flows = _flows_np(exo, action, node, slack, ptdf_np)

    if not _feasible(flows, limits_np):
        action, idx, ok, moved, curtailed = _value_aware_repair(u, total, idx, node, ptdf_np, limits_np, exo, slack)
        if moved:
            _BENCH_STATS["repair_count"] += 1
            _BENCH_STATS["curtailed_mwh"] += curtailed * _DT
        if not ok:
            action = np.array(action, dtype=float)
            action, _, _ = _safe_network_repair(action, node, ptdf_np, limits_np, exo, slack)
            flows = _flows_np(exo, action, node, slack, ptdf_np)
            if not _feasible(flows, limits_np):
                action = np.zeros(B)
                _BENCH_STATS["zero_fallback_count"] += 1
            return np.clip(action, lb, ub).tolist()

    action, idx = _coordinate_improve(u, total, idx, node, ptdf_np, limits_np, exo, slack)

    flows = _flows_np(exo, action, node, slack, ptdf_np)
    if not _feasible(flows, limits_np):
        action, _, _ = _safe_network_repair(action, node, ptdf_np, limits_np, exo, slack)
        flows = _flows_np(exo, action, node, slack, ptdf_np)
        if not _feasible(flows, limits_np):
            action = np.zeros(B)
            _BENCH_STATS["zero_fallback_count"] += 1
    action = np.clip(action, lb, ub)

    # CONGESTED only: also try the constructive build-up-from-zero allocator
    # (Candidate D) and keep it only if it scores strictly better on the same
    # causal local objective (reward + validated continuation value) -- a
    # safe fallback discipline (Candidate C), never worse than the standard
    # policy_v2 result above. BASELINE (B<=10) and the legacy scenarios keep
    # the unmodified, already-validated policy_v2 behavior: Phase 2's oracle
    # analysis showed BASELINE is already close to its information-theoretic
    # ceiling, and Phase 1/3/4 showed an alternative (rolling-horizon) control
    # law for it actively regresses, so no new mechanism is applied there.
    if _CONGESTED_MIN_B <= B <= _CONGESTED_MAX_B:
        idx_unconstrained = np.argmax(total, axis=1)
        alt_action, _ = _constructive_from_grid(u, total, idx_unconstrained, node, ptdf_np, limits_np, exo, slack)
        alt_action = np.clip(alt_action, lb, ub)
        alt_flows = _flows_np(exo, alt_action, node, slack, ptdf_np)
        if _feasible(alt_flows, limits_np):
            val_std = _evaluate_total_value(action, price, cap, soc, smin, smax, etac, etad, V_next, S)
            val_alt = _evaluate_total_value(alt_action, price, cap, soc, smin, smax, etac, etad, V_next, S)
            if val_alt > val_std:
                action = alt_action

    # BASELINE + CONGESTED: also try the exact convex (LP epigraph) joint
    # solve -- continuous actions, globally optimal for the exact
    # convexification of this step's local objective, subject to the true
    # PTDF flow constraints as hard LP constraints (not a post-hoc repair).
    # Kept only if it scores strictly better than the discrete-grid result
    # above on the identical local objective -- never worse by construction.
    if B <= _LP_MAX_B:
        lp_action = _lp_joint_solve(soc, price, cap, smin, smax, etac, etad, lb, ub, V_next, S,
                                     node, ptdf_np, limits_np, exo, slack)
        if lp_action is not None:
            lp_action = np.clip(lp_action, lb, ub)
            lp_flows = _flows_np(exo, lp_action, node, slack, ptdf_np)
            if _feasible(lp_flows, limits_np):
                val_std = _evaluate_total_value(action, price, cap, soc, smin, smax, etac, etad, V_next, S)
                val_lp = _evaluate_total_value(lp_action, price, cap, soc, smin, smax, etac, etad, V_next, S)
                if val_lp > val_std:
                    action = lp_action

    return action.tolist()
