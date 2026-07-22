"""Shared, fresh (not copied from best_candidate.py/policy_v11.py) building
blocks for the Phase 3 fitted-value-iteration architecture:

  V_joint(s) = sum_i V_i(soc_i) + C_theta(phi(s))

V_i is a fresh, simplified per-battery Gaussian-quadrature Bellman
recursion (the core mechanism already validated this session -- NOT what's
being tested here). C_theta(phi(s)) is the new, learned joint/network-
coupled correction this phase adds.

All feature-extraction inputs are PUBLIC and CAUSAL: SOC (own state),
day-ahead prices (given for the full horizon at t=0, verified from
PolicyView/Market source), exogenous injections (given for the full horizon
at t=0, verified from PolicyView source), network PTDF/flow limits (fixed,
public), and the CURRENT step's own realized RT price (causally revealed at
decision time, same channel every real policy has). No future RT price, no
seed/nonce, no hidden state.
"""
import numpy as np

_DT = 0.25
_KAPPA_TX = 0.25
_KAPPA_DEG = 1.0
_BETA_DEG = 2.0
_EPS = 1e-12
_S = 21  # SOC grid points for the separable value function

_MARKET_PARAMS_BY_B = {
    10: dict(sigma=0.10), 20: dict(sigma=0.15), 40: dict(sigma=0.20),
    60: dict(sigma=0.25), 100: dict(sigma=0.30),
}

_GH_ORDER = 5
_GH_X, _GH_W_RAW = np.polynomial.hermite.hermgauss(_GH_ORDER)
_GH_Z = np.sqrt(2.0) * _GH_X
_GH_W = _GH_W_RAW / np.sqrt(np.pi)

N_FEATURES = 26
FEATURE_NAMES = [
    "time_remaining_frac", "cyc_sin", "cyc_cos",
    "agg_charge_headroom_norm", "agg_discharge_headroom_norm",
    "soc_frac_mean", "soc_frac_std", "soc_frac_p10", "soc_frac_p50", "soc_frac_p90",
    "frac_near_bounds", "charge_headroom_cv", "discharge_headroom_cv",
    "top3_line_margin_mean", "top3_line_margin_min",
    "line_util_mean", "line_util_max",
    "future_congestion_risk",
    "da_level_norm", "da_slope", "da_curvature", "da_std_future",
    "rt_da_resid_now", "rt_da_resid_prev",
    "headroom_x_util_charge", "headroom_x_util_discharge",
]


def battery_arrays_from_batteries(batteries):
    cap = np.array([b.capacity_mwh for b in batteries])
    pchg = np.array([b.power_charge_mw for b in batteries])
    pdis = np.array([b.power_discharge_mw for b in batteries])
    etac = np.array([b.efficiency_charge for b in batteries])
    etad = np.array([b.efficiency_discharge for b in batteries])
    smin = np.array([b.soc_min_mwh for b in batteries])
    smax = np.array([b.soc_max_mwh for b in batteries])
    node = np.array([b.node for b in batteries], dtype=np.intp)
    return dict(cap=cap, pchg=pchg, pdis=pdis, etac=etac, etad=etad,
                smin=smin, smax=smax, node=node, B=len(cap))


def action_bounds_np(soc, ba):
    smin, smax, etac, etad, pchg, pdis = ba["smin"], ba["smax"], ba["etac"], ba["etad"], ba["pchg"], ba["pdis"]
    headroom = np.maximum(smax - soc, 0.0)
    avail = np.maximum(soc - smin, 0.0)
    max_charge = np.clip(np.minimum(headroom / np.maximum(etac * _DT, _EPS), pchg), 0.0, None)
    max_discharge = np.clip(np.minimum(avail * etad / _DT, pdis), 0.0, None)
    return max_charge, max_discharge


def apply_action_np(u, soc, ba):
    c = np.maximum(-u, 0.0)
    d = np.maximum(u, 0.0)
    new_soc = soc + ba["etac"] * c * _DT - d * _DT / ba["etad"]
    return np.clip(new_soc, ba["smin"], ba["smax"])


def build_separable_value_function(day_ahead_prices, node, ba, B_key):
    """Fresh per-battery Gaussian-quadrature Bellman recursion (independent
    reimplementation of the core, already-validated mechanism from this
    session -- simplified: Gaussian sigma term only, no jump/congestion
    mixture, since the correction term C_theta is what's being tested here,
    not a re-tuning of the base value function). Returns V_all[t, i, s]."""
    da = np.asarray(day_ahead_prices, dtype=float)
    T = da.shape[0]
    B = ba["B"]
    sigma = _MARKET_PARAMS_BY_B.get(B_key, dict(sigma=0.15))["sigma"]
    smin, smax = ba["smin"], ba["smax"]
    cap = ba["cap"]
    soc_grid = smin[:, None] + (smax - smin)[:, None] * np.linspace(0.0, 1.0, _S)[None, :]
    price_forecast = da[:, node]

    smin2, smax2 = smin[:, None], smax[:, None]
    ba2 = dict(ba, smin=smin2, smax=smax2, etac=ba["etac"][:, None], etad=ba["etad"][:, None],
               pchg=ba["pchg"][:, None], pdis=ba["pdis"][:, None])
    lb, ub = action_bounds_np(soc_grid, ba2)
    k = 3
    fracs = np.linspace(1.0 / k, 1.0, k)
    u = np.concatenate([(-lb)[..., None] * fracs[::-1], np.zeros(lb.shape + (1,)), ub[..., None] * fracs], axis=-1)
    abs_u = np.abs(u)
    smin3, smax3 = smin[:, None, None], smax[:, None, None]
    ba3 = dict(ba, smin=smin3, smax=smax3, etac=ba["etac"][:, None, None], etad=ba["etad"][:, None, None])
    new_soc = apply_action_np(u, soc_grid[:, :, None], ba3)
    span = np.maximum(smax3 - smin3, _EPS)
    idx_frac = np.clip((new_soc - smin3) / span * (_S - 1), 0.0, _S - 1 - 1e-6)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, _S - 1)
    w = idx_frac - i0
    cost_term = _KAPPA_TX * abs_u * _DT + _KAPPA_DEG * (abs_u * _DT / cap[:, None, None]) ** _BETA_DEG

    V_all = np.zeros((T + 1, B, _S))
    b_idx = np.arange(B)[:, None, None]
    V_next = V_all[T]
    for t in range(T - 1, -1, -1):
        Vc = V_next[b_idx, i0] * (1.0 - w) + V_next[b_idx, i1] * w
        base_price = price_forecast[t][:, None, None]
        V_here = np.zeros((B, _S))
        for z, wq in zip(_GH_Z, _GH_W):
            price_k = base_price * (1.0 + sigma * z)
            total_k = u * price_k * _DT - cost_term + Vc
            V_here += wq * np.max(total_k, axis=2)
        V_all[t] = V_here
        V_next = V_here
    return V_all


def separable_value_at(V_all, t, soc, ba):
    """Interpolated sum_i V_i(soc_i) at step t."""
    S = V_all.shape[2]
    smin, smax = ba["smin"], ba["smax"]
    span = np.maximum(smax - smin, _EPS)
    idx_frac = np.clip((soc - smin) / span * (S - 1), 0.0, S - 1 - 1e-6)
    i0 = idx_frac.astype(np.intp)
    i1 = np.minimum(i0 + 1, S - 1)
    w = idx_frac - i0
    rows = np.arange(len(soc))
    Vt = V_all[t]
    return float(np.sum(Vt[rows, i0] * (1.0 - w) + Vt[rows, i1] * w))


def compute_zero_action_flows(exo, node, ptdf, slack):
    inj = np.array(exo, dtype=float)
    inj[slack] = 0.0
    inj[slack] = -(inj.sum() - inj[slack])
    return ptdf @ inj


def extract_features(t, T, soc, ba, node, da_all, exo_all, ptdf, limits, slack,
                      rt_now, rt_prev, da_now):
    """phi(s): fixed-size (N_FEATURES,) causal feature vector. All inputs
    are public/causal (see module docstring)."""
    B = ba["B"]
    smin, smax = ba["smin"], ba["smax"]
    cap = ba["cap"]
    max_charge, max_discharge = action_bounds_np(soc, ba)
    total_power = np.sum(ba["pchg"]) + np.sum(ba["pdis"]) + _EPS

    soc_frac = (soc - smin) / np.maximum(smax - smin, _EPS)
    near_bounds = np.mean((soc_frac < 0.15) | (soc_frac > 0.85))

    charge_headroom_cv = float(np.std(max_charge) / (np.mean(max_charge) + _EPS))
    discharge_headroom_cv = float(np.std(max_discharge) / (np.mean(max_discharge) + _EPS))

    zero_flows = compute_zero_action_flows(exo_all[t], node, ptdf, slack)
    line_util = np.abs(zero_flows) / np.maximum(limits, _EPS)
    top3_idx = np.argsort(-line_util)[:3]
    top3_margin = 1.0 - line_util[top3_idx]

    H = 8
    end = min(t + 1 + H, T)
    future_utils = []
    for tf in range(t + 1, end):
        f = compute_zero_action_flows(exo_all[tf], node, ptdf, slack)
        future_utils.append(np.max(np.abs(f) / np.maximum(limits, _EPS)))
    future_congestion_risk = float(np.mean(future_utils)) if future_utils else float(np.max(line_util))

    da_node0 = da_all[:, 0]
    da_mean_all = float(np.mean(da_node0))
    da_level_norm = float((da_now - da_mean_all) / (da_mean_all + _EPS))
    end2 = min(t + 1 + H, T)
    fut_da = da_node0[t + 1:end2]
    da_slope = float((fut_da[-1] - da_now) / max(len(fut_da), 1)) if len(fut_da) else 0.0
    da_curv = float(np.mean(np.diff(fut_da, 2))) if len(fut_da) > 2 else 0.0
    da_std_future = float(np.std(fut_da)) if len(fut_da) else 0.0

    rt_da_resid_now = float((rt_now - da_now) / (abs(da_now) + _EPS))
    rt_da_resid_prev = float((rt_prev - da_now) / (abs(da_now) + _EPS)) if rt_prev is not None else 0.0

    feats = np.array([
        (T - t) / T, np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T),
        float(np.sum(max_charge)) / total_power, float(np.sum(max_discharge)) / total_power,
        float(np.mean(soc_frac)), float(np.std(soc_frac)),
        float(np.percentile(soc_frac, 10)), float(np.percentile(soc_frac, 50)), float(np.percentile(soc_frac, 90)),
        float(near_bounds), charge_headroom_cv, discharge_headroom_cv,
        float(np.mean(top3_margin)), float(np.min(top3_margin)),
        float(np.mean(line_util)), float(np.max(line_util)),
        future_congestion_risk,
        da_level_norm, da_slope, da_curv, da_std_future,
        rt_da_resid_now, rt_da_resid_prev,
        float(np.sum(max_charge)) / total_power * float(np.max(line_util)),
        float(np.sum(max_discharge)) / total_power * float(np.max(line_util)),
    ], dtype=float)
    assert feats.shape[0] == N_FEATURES
    return feats
