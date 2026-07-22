"""
Energy Arbitrage miner policy (v3, follow-up to submission 138231 / policy.py).

Strategy
--------
1. On the first call of an episode, run a vectorized backward dynamic
   program (per battery, ignoring the network) over the remaining horizon
   using day-ahead prices as an unbiased forecast of real-time prices.
   This produces a continuation-value table V[t][soc] per battery that
   captures the *opportunity cost* of using energy now instead of saving
   it for a better future price (classic storage "water value").

2. At every step, combine the true, currently-known real-time price with
   the cached continuation value to pick a per-battery action via a small
   1-D search (myopic reward + expected future value == receding horizon
   / MPC-style control).

3. Batteries are only coupled through the DC power-flow network
   constraints. The per-battery ideal actions are projected onto the
   feasible flow polytope with a merit-order curtailment (cut the
   cheapest-to-curtail MW first), independently double-checked and, if
   anything is off, safely repaired with a proven proportional
   scale-down + bisection fallback so a flow violation can never be
   submitted.

4. Grid resolution (SOC buckets / action buckets, both offline for the DP
   and online for the per-step search) scales inversely with the number of
   batteries in the instance, so small scenarios get a much finer search.

5. NEW: the day-ahead-only value function assumes real-time prices equal
   day-ahead prices in expectation, which is unbiased but throws away the
   fact that real-time deviations are locally persistent (weather / demand
   shocks / congestion do not vanish one step after they're observed). Each
   step, an EMA of the (real-time - day-ahead) gap per node is updated from
   the true observed price, and a short local backward re-plan (a few hours
   ahead, decaying back to the day-ahead assumption) is spliced onto the
   long-horizon cached table before the 1-D action search. This is a pure
   addition to the value estimate used for planning; the safety/projection
   pipeline is unchanged, and any failure in the patch silently falls back
   to the original day-ahead-only continuation value.

Only the standard library and numpy are used.
"""

import numpy as np

_CACHE = {}

_DT = 0.25
_KAPPA_TX = 0.25
_KAPPA_DEG = 1.00
_BETA_DEG = 2.0
_EPS_FLOW = 1e-6
_EPS = 1e-12

# Bias-correction (short-horizon replan) parameters. The observed
# (rt - da) gap at a single step is already the relevant state for a
# Markovian deviation process (no EMA lag needed), but a lone observation
# is noisy and may just be a one-off spike, so it is shrunk toward zero and
# clipped using the node's own day-ahead price volatility (self-calibrating,
# no scenario-specific tuning needed) before being extrapolated forward.
_BIAS_SHRINK = 0.5      # fraction of the observed gap trusted as signal
_BIAS_CLIP_MULT = 1.5   # clip |gap| to this many day-ahead price std-devs
_BIAS_PHI = 0.7         # per-step decay applied to the gap going forward
_BIAS_K = 6             # how many steps ahead the local replan covers
_BIAS_A = 27            # action-grid size for the (cheap, short) local replan


# ---------------------------------------------------------------------
# Battery physics (vectorized re-implementation of Battery.* for the
# offline planning grid; at runtime we always use the environment's own
# state.action_bounds, never a recomputed value).
# ---------------------------------------------------------------------


def _action_bounds_np(soc, smin, smax, etac, etad, pchg, pdis, dt):
    headroom = np.maximum(smax - soc, 0.0)
    avail = np.maximum(soc - smin, 0.0)
    max_charge_from_soc = np.where(etac > 0, headroom / np.maximum(etac * dt, _EPS), 0.0)
    max_discharge_from_soc = np.where(etad > 0, avail * etad / dt, 0.0)
    max_charge = np.clip(np.minimum(max_charge_from_soc, pchg), 0.0, None)
    max_discharge = np.clip(np.minimum(max_discharge_from_soc, pdis), 0.0, None)
    return -max_charge, max_discharge


def _apply_action_np(u, soc, smin, smax, etac, etad, dt):
    c = np.maximum(-u, 0.0)
    d = np.maximum(u, 0.0)
    new_soc = soc + etac * c * dt - d * dt / etad
    return np.clip(new_soc, smin, smax)


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


# ---------------------------------------------------------------------
# Adaptive grid resolution.
# ---------------------------------------------------------------------


def _resolution_for(B):
    """SOC-grid size, offline-DP action-grid size, and online action-grid
    size, tiered by battery count. Chosen so that horizon*B*S*A for the
    offline DP build stays in the same ballpark across tiers (small B gets
    a much finer grid instead of leaving compute headroom unused); the
    online per-step search cost (B*A2) is trivial at every tier."""
    if B <= 10:
        return 121, 81, 241
    if B <= 20:
        return 91, 61, 181
    if B <= 40:
        return 61, 45, 121
    if B <= 60:
        return 45, 35, 91
    return 31, 25, 61


# ---------------------------------------------------------------------
# Offline backward DP: per-battery continuation-value table.
# ---------------------------------------------------------------------


def _build_value_function(challenge, t_start, ba, S, A):
    T = challenge.num_steps
    B = ba["B"]
    horizon = T - t_start
    V_all = np.zeros((horizon + 1, B, S))
    if horizon <= 0:
        return V_all

    smin, smax = ba["smin"], ba["smax"]
    etac, etad = ba["etac"], ba["etad"]
    pchg, pdis, cap, node = ba["pchg"], ba["pdis"], ba["cap"], ba["node"]

    soc_grid = smin[:, None] + (smax - smin)[:, None] * np.linspace(0.0, 1.0, S)[None, :]  # (B,S)

    da = np.asarray(challenge.market.day_ahead_prices, dtype=float)  # (T,N)
    price_forecast = da[:, node]  # (T,B)

    smin2, smax2 = smin[:, None], smax[:, None]
    etac2, etad2 = etac[:, None], etad[:, None]
    pchg2, pdis2 = pchg[:, None], pdis[:, None]

    smin3, smax3 = smin[:, None, None], smax[:, None, None]
    etac3, etad3 = etac[:, None, None], etad[:, None, None]
    cap3 = cap[:, None, None]

    frac = np.linspace(0.0, 1.0, A)[None, None, :]
    b_idx = np.arange(B)[:, None, None]

    V_next = V_all[horizon]
    for idx in range(horizon - 1, -1, -1):
        t = t_start + idx
        lb, ub = _action_bounds_np(soc_grid, smin2, smax2, etac2, etad2, pchg2, pdis2, _DT)  # (B,S)
        u = lb[:, :, None] + (ub - lb)[:, :, None] * frac  # (B,S,A)

        price_t = price_forecast[t][:, None, None]
        abs_u = np.abs(u)
        reward = u * price_t * _DT - _KAPPA_TX * abs_u * _DT - _KAPPA_DEG * (abs_u * _DT / cap3) ** _BETA_DEG

        new_soc = _apply_action_np(u, soc_grid[:, :, None], smin3, smax3, etac3, etad3, _DT)  # (B,S,A)
        span = np.maximum(smax3 - smin3, _EPS)
        idx_frac = np.clip((new_soc - smin3) / span * (S - 1), 0.0, S - 1 - 1e-6)
        i0 = idx_frac.astype(np.intp)
        i1 = np.minimum(i0 + 1, S - 1)
        w = idx_frac - i0

        Vc = V_next[b_idx, i0] * (1.0 - w) + V_next[b_idx, i1] * w
        total = reward + Vc
        V_here = np.max(total, axis=2)  # (B,S)
        V_all[idx] = V_here
        V_next = V_here

    return V_all


# ---------------------------------------------------------------------
# NEW: short-horizon, bias-corrected local replan.
#
# The cached V_all table above assumes RT price == day-ahead price in
# expectation for every future step. That's an unbiased planning
# assumption but ignores that a currently-observed RT/DA gap tends to
# persist for a while (weather, demand shocks, congestion don't vanish
# instantly). This rebuilds just the next _BIAS_K steps of the value
# function with price = day_ahead + ema_gap * phi^k, stitched onto the
# existing long-horizon table as the terminal boundary. It reuses the same
# soc_grid resolution as the cached table so the two are interpolation-
# compatible.
# ---------------------------------------------------------------------


def _local_patch_value(challenge, ba, t, S, V_all, idx_next, bias_node):
    B = ba["B"]
    T = challenge.num_steps
    k_end = min(_BIAS_K, T - (t + 1))
    if k_end <= 0:
        if 0 <= idx_next < V_all.shape[0]:
            return V_all[idx_next]
        return np.zeros((B, S))

    smin, smax = ba["smin"], ba["smax"]
    etac, etad = ba["etac"], ba["etad"]
    pchg, pdis, cap, node = ba["pchg"], ba["pdis"], ba["cap"], ba["node"]

    soc_grid = smin[:, None] + (smax - smin)[:, None] * np.linspace(0.0, 1.0, S)[None, :]
    da = np.asarray(challenge.market.day_ahead_prices, dtype=float)

    idx_boundary = idx_next + k_end
    if 0 <= idx_boundary < V_all.shape[0]:
        V_next = V_all[idx_boundary]
    else:
        V_next = np.zeros((B, S))

    smin2, smax2 = smin[:, None], smax[:, None]
    etac2, etad2 = etac[:, None], etad[:, None]
    pchg2, pdis2 = pchg[:, None], pdis[:, None]
    smin3, smax3 = smin[:, None, None], smax[:, None, None]
    etac3, etad3 = etac[:, None, None], etad[:, None, None]
    cap3 = cap[:, None, None]

    A = _BIAS_A
    frac = np.linspace(0.0, 1.0, A)[None, None, :]
    b_idx = np.arange(B)[:, None, None]

    bias_b = bias_node[node]  # (B,)

    lb, ub = _action_bounds_np(soc_grid, smin2, smax2, etac2, etad2, pchg2, pdis2, _DT)
    u = lb[:, :, None] + (ub - lb)[:, :, None] * frac
    abs_u = np.abs(u)
    new_soc = _apply_action_np(u, soc_grid[:, :, None], smin3, smax3, etac3, etad3, _DT)
    span = np.maximum(smax3 - smin3, _EPS)
    idx_frac = np.clip((new_soc - smin3) / span * (S - 1), 0.0, S - 1 - 1e-6)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    deg_term = _KAPPA_DEG * (abs_u * _DT / cap3) ** _BETA_DEG
    tx_term = _KAPPA_TX * abs_u * _DT

    for k in range(k_end - 1, -1, -1):
        tt = t + 1 + k
        price_t = (da[tt][node] + bias_b * (_BIAS_PHI ** k))[:, None, None]
        reward = u * price_t * _DT - tx_term - deg_term

        Vc = V_next[b_idx, i0] * (1.0 - w) + V_next[b_idx, i1] * w
        total = reward + Vc
        V_next = np.max(total, axis=2)

    return V_next


# ---------------------------------------------------------------------
# Network helpers (fast numpy re-implementation of PolicyView / Network
# public methods, used only for speed inside the curtailment loop; the
# final action is always sanity-checked).
# ---------------------------------------------------------------------


def _injections_np(exo, action_arr, node_arr, num_nodes, slack):
    inj = np.array(exo, dtype=float)
    inj[slack] = 0.0
    np.add.at(inj, node_arr, action_arr)
    total = inj.sum() - inj[slack]
    inj[slack] = -total
    return inj


def _flows_np(exo, action_arr, node_arr, num_nodes, slack, ptdf_np):
    return ptdf_np @ _injections_np(exo, action_arr, node_arr, num_nodes, slack)


def _feasible(flows, limits):
    viol = np.abs(flows) - limits
    return not np.any(viol > _EPS_FLOW * limits)


def _battery_value(u, price, cap):
    abs_u = np.abs(u)
    return u * price * _DT - _KAPPA_TX * abs_u * _DT - _KAPPA_DEG * (abs_u * _DT / cap) ** _BETA_DEG


def _merit_order_curtail(action, price_arr, cap_arr, node_arr, ptdf_np, limits_np, exo, num_nodes, slack, max_iters=600):
    action = action.copy()
    for _ in range(max_iters):
        flows = _flows_np(exo, action, node_arr, num_nodes, slack, ptdf_np)
        viol = np.abs(flows) - limits_np
        bad = viol > _EPS_FLOW * limits_np
        if not np.any(bad):
            return action, True

        masked = np.where(bad, viol, -np.inf)
        l = int(np.argmax(masked))
        flow_l = flows[l]
        sign_flow = 1.0 if flow_l > 0 else -1.0

        contrib = ptdf_np[l, node_arr] * action
        worsening = (contrib * sign_flow) > _EPS
        idxs = np.nonzero(worsening)[0]
        if idxs.size == 0:
            return action, False  # cannot fix this line by curtailing batteries

        u_cur = action[idxs]
        step = np.maximum(np.abs(u_cur) * 0.02, 1e-4)
        u_red = u_cur - np.sign(u_cur) * step
        v_cur = _battery_value(u_cur, price_arr[idxs], cap_arr[idxs])
        v_red = _battery_value(u_red, price_arr[idxs], cap_arr[idxs])
        marginal_loss = (v_cur - v_red) / step
        order = idxs[np.argsort(marginal_loss)]

        target = limits_np[l] * (1.0 - 1e-9) * sign_flow
        need = flow_l - target

        for b in order:
            ptdf_bl = ptdf_np[l, node_arr[b]]
            if abs(ptdf_bl) < _EPS:
                continue
            delta_u = need / ptdf_bl
            new_u = action[b] - delta_u
            if action[b] >= 0:
                new_u = max(0.0, min(new_u, action[b]))
            else:
                new_u = min(0.0, max(new_u, action[b]))
            actual_delta_contrib = (action[b] - new_u) * ptdf_bl
            action[b] = new_u
            need -= actual_delta_contrib
            if abs(need) < 1e-9:
                break

    flows = _flows_np(exo, action, node_arr, num_nodes, slack, ptdf_np)
    return action, _feasible(flows, limits_np)


def _proven_safe_projection(challenge, state, action):
    """Verbatim port of the reference baselines' proportional-softening
    + global-bisection fallback (greedy.py / conservative.py). Guaranteed
    to return a flow-feasible action, assuming zero is feasible."""
    net = challenge.network
    action = list(action)

    def flows_of(a):
        inj = challenge.compute_total_injections(state, a)
        return net.compute_flows(inj)

    def most_violated(flows):
        best = None
        for l, flow in enumerate(flows):
            limit = net.flow_limits[l]
            v = abs(flow) - limit
            if v > _EPS_FLOW * limit:
                if best is None or v > best[2]:
                    best = (l, flow, v)
        return best

    def is_feasible(a):
        return most_violated(flows_of(a)) is None

    for _ in range(64):
        flows = flows_of(action)
        v = most_violated(flows)
        if v is None:
            return action
        line, flow, amount = v
        sign = 1.0 if flow > 0 else (-1.0 if flow < 0 else 0.0)
        if abs(sign) <= _EPS:
            break
        worsening_idx = []
        worsening_strength = 0.0
        for i, battery in enumerate(challenge.batteries):
            contribution = net.ptdf[line][battery.node] * action[i]
            signed = sign * contribution
            if signed > _EPS:
                worsening_strength += signed
                worsening_idx.append(i)
        if not worsening_idx or worsening_strength <= _EPS:
            break
        keep = max(0.0, min(1.0 - amount / worsening_strength, 1.0))
        if abs(1.0 - keep) <= _EPS:
            break
        for i in worsening_idx:
            action[i] *= keep

    if is_feasible(action):
        return action

    zero = [0.0] * len(action)
    if not is_feasible(zero):
        return zero  # last resort; should not happen by construction

    base = action
    low, high = 0.0, 1.0
    for _ in range(32):
        mid = 0.5 * (low + high)
        scaled = [mid * u for u in base]
        if is_feasible(scaled):
            low = mid
        else:
            high = mid
    return [low * u for u in base]


# ---------------------------------------------------------------------
# Main policy entry point.
# ---------------------------------------------------------------------


def policy(challenge, state):
    B = challenge.num_batteries
    if B == 0:
        return []

    key = id(challenge)
    b0 = challenge.batteries[0]
    sig = (
        challenge.num_batteries,
        challenge.network.num_nodes,
        challenge.network.num_lines,
        challenge.num_steps,
        round(b0.capacity_mwh, 6),
        round(b0.node, 6),
        round(challenge.market.day_ahead_prices[0][0], 6),
    )
    entry = _CACHE.get(key)
    t = state.time_step

    if entry is None or entry.get("sig") != sig or t < entry["t_start"]:
        if len(_CACHE) > 4:
            _CACHE.clear()  # avoid unbounded growth if keys keep colliding/rotating
        ba = _battery_arrays(challenge)
        S, A, A2 = _resolution_for(B)
        try:
            V_all = _build_value_function(challenge, t, ba, S, A)
        except Exception:
            V_all = np.zeros((challenge.num_steps - t + 1, B, S))
        entry = {"ba": ba, "V_all": V_all, "t_start": t, "sig": sig, "S": S, "A2": A2, "bias": None}
        _CACHE[key] = entry

    ba = entry["ba"]
    V_all = entry["V_all"]
    idx_next = t - entry["t_start"] + 1
    S = V_all.shape[2]
    if 0 <= idx_next < V_all.shape[0]:
        V_next = V_all[idx_next]
    else:
        V_next = np.zeros((B, S))

    # --- bias tracking + short-horizon local replan --------------------
    try:
        sigma_da = entry.get("sigma_da")
        if sigma_da is None:
            da_all = np.asarray(challenge.market.day_ahead_prices, dtype=float)
            sigma_da = np.std(da_all, axis=0)
            sigma_da = np.maximum(sigma_da, 1.0)  # floor so a flat price series doesn't zero out the clip
            entry["sigma_da"] = sigma_da

        da_now = np.asarray(challenge.market.day_ahead_prices[t], dtype=float)
        rt_now = np.asarray(state.rt_prices, dtype=float)
        gap_obs = rt_now - da_now
        clip = _BIAS_CLIP_MULT * sigma_da
        bias_node = _BIAS_SHRINK * np.clip(gap_obs, -clip, clip)
        V_next = _local_patch_value(challenge, ba, t, S, V_all, idx_next, bias_node)
    except Exception:
        pass  # fall back to the unpatched day-ahead-only V_next

    soc = np.array(state.socs, dtype=float)
    bounds = np.array(state.action_bounds, dtype=float)  # (B,2), authoritative
    lb, ub = bounds[:, 0], bounds[:, 1]

    node = ba["node"]
    cap = ba["cap"]
    smin, smax = ba["smin"], ba["smax"]
    etac, etad = ba["etac"], ba["etad"]

    price = np.array(state.rt_prices, dtype=float)[node]  # true current RT price per battery

    A2 = entry["A2"]
    frac = np.linspace(0.0, 1.0, A2)[None, :]
    u = lb[:, None] + (ub - lb)[:, None] * frac  # (B,A2)

    abs_u = np.abs(u)
    reward = u * price[:, None] * _DT - _KAPPA_TX * abs_u * _DT - _KAPPA_DEG * (abs_u * _DT / cap[:, None]) ** _BETA_DEG

    new_soc = _apply_action_np(u, soc[:, None], smin[:, None], smax[:, None], etac[:, None], etad[:, None], _DT)
    span = np.maximum(smax - smin, _EPS)[:, None]
    idx_frac = np.clip((new_soc - smin[:, None]) / span * (S - 1), 0.0, S - 1 - 1e-6)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    rows = np.arange(B)[:, None]
    Vc = V_next[rows, i0] * (1.0 - w) + V_next[rows, i1] * w

    total = reward + Vc
    best = np.argmax(total, axis=1)
    u_star = u[np.arange(B), best]
    u_star = np.clip(u_star, lb, ub)

    # --- network-feasibility projection -------------------------------
    net = challenge.network
    num_nodes = net.num_nodes
    slack = net.slack_bus
    exo = np.asarray(state.exogenous_injections, dtype=float)

    pcache = entry.get("net")
    if pcache is None:
        pcache = {
            "ptdf": np.asarray(net.ptdf, dtype=float),
            "limits": np.asarray(net.flow_limits, dtype=float),
        }
        entry["net"] = pcache
    ptdf_np, limits_np = pcache["ptdf"], pcache["limits"]

    final_action = u_star
    try:
        curtailed, ok = _merit_order_curtail(u_star, price, cap, node, ptdf_np, limits_np, exo, num_nodes, slack)
        if ok:
            flows_check = _flows_np(exo, curtailed, node, num_nodes, slack, ptdf_np)
            if _feasible(flows_check, limits_np):
                final_action = curtailed
            else:
                ok = False
        if not ok:
            final_action = np.array(_proven_safe_projection(challenge, state, list(u_star)), dtype=float)
    except Exception:
        try:
            final_action = np.array(_proven_safe_projection(challenge, state, list(u_star)), dtype=float)
        except Exception:
            final_action = np.zeros(B)

    final_action = np.clip(final_action, lb, ub)

    # Final hard safety check; degrade to zero only if truly necessary.
    flows_final = _flows_np(exo, final_action, node, num_nodes, slack, ptdf_np)
    if not _feasible(flows_final, limits_np):
        try:
            final_action = np.array(_proven_safe_projection(challenge, state, list(final_action)), dtype=float)
        except Exception:
            final_action = np.zeros(B)
        final_action = np.clip(final_action, lb, ub)
        flows_final = _flows_np(exo, final_action, node, num_nodes, slack, ptdf_np)
        if not _feasible(flows_final, limits_np):
            final_action = np.zeros(B)

    return final_action.tolist()
