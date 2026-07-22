"""Architecture B test: policy ensemble of complete candidate schedules,
rolled forward through the internal (causal, DA-based) model, executing the
first action of the highest-valued rollout. Not a submission policy -- a
diagnostic to test whether this architecture can beat policy_v9.
"""
import sys
from pathlib import Path

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch
import policy_v9 as v9

_DT = 0.25
_KAPPA_TX = 0.25
_KAPPA_DEG = 1.0
_BETA_DEG = 2.0
H = 16  # rolling horizon


def _battery_arrays(ch):
    return v9._battery_arrays(ch)


def _rollout_value(ba, soc0, price_path, style, node):
    """Simulate a full H-step schedule of the given 'style' using the DA
    forecast (causal, no future RT), return (first_action, total_value)."""
    B = len(soc0)
    smin, smax, etac, etad, cap = ba["smin"], ba["smax"], ba["etac"], ba["etad"], ba["cap"]
    pchg, pdis = ba["pchg"], ba["pdis"]
    soc = soc0.copy()
    total_value = np.zeros(B)
    first_action = np.zeros(B)
    Hc = price_path.shape[0]

    for t in range(Hc):
        headroom = np.maximum(smax - soc, 0.0)
        avail = np.maximum(soc - smin, 0.0)
        max_c = np.minimum(headroom / np.maximum(etac * _DT, 1e-12), pchg)
        max_d = np.minimum(avail * etad / _DT, pdis)
        price_t = price_path[t]
        future_mean = price_path[t:].mean(axis=0) if t < Hc else price_t

        if style == "conservative":
            action = np.zeros(B)
        elif style == "aggressive":
            action = np.where(price_t > future_mean, max_d, -max_c)
        elif style == "reserve":
            target_frac = 0.5
            desired = smin + target_frac * (smax - smin)
            delta = desired - soc
            action = np.where(delta > 0, -np.minimum(delta / np.maximum(etac * _DT, 1e-12), max_c),
                               np.minimum(-delta * etad / _DT, max_d))
        elif style == "congestion_aware":
            action = 0.5 * np.where(price_t > future_mean, max_d, -max_c)
        else:  # "threshold" -- simple price-vs-future-mean like the greedy baseline family
            action = np.where(price_t > future_mean * 1.05, max_d,
                               np.where(price_t < future_mean * 0.95, -max_c, 0.0))

        abs_a = np.abs(action)
        reward = action * price_t * _DT - _KAPPA_TX * abs_a * _DT - _KAPPA_DEG * (abs_a * _DT / cap) ** _BETA_DEG
        total_value += reward
        c = np.maximum(-action, 0.0); d = np.maximum(action, 0.0)
        soc = np.clip(soc + etac * c * _DT - d * _DT / etad, smin, smax)
        if t == 0:
            first_action = action.copy()

    return first_action, total_value


def policy_arch_b(challenge, state):
    B = challenge.num_batteries
    if B == 0:
        return []
    ba = _battery_arrays(challenge)
    node = ba["node"]
    t = state.time_step
    T = challenge.num_steps
    da_all = np.asarray(challenge.market.day_ahead_prices)
    Hc = min(H, T - t)
    price_path = da_all[t:t + Hc][:, node].copy()
    price_path[0] = np.asarray(state.rt_prices)[node]  # step 0 uses the real observed price

    soc0 = np.array(state.socs)
    styles = ["conservative", "aggressive", "reserve", "congestion_aware", "threshold"]
    best_val = None
    best_action = np.zeros(B)
    for style in styles:
        a0, val = _rollout_value(ba, soc0, price_path, style, node)
        total = val.sum()
        if best_val is None or total > best_val:
            best_val = total
            best_action = a0

    bounds = np.array(state.action_bounds)
    lb, ub = bounds[:, 0], bounds[:, 1]
    action = np.clip(best_action, lb, ub)

    net = challenge.network
    exo = np.asarray(state.exogenous_injections)
    ptdf = np.asarray(net.ptdf)
    limits = np.asarray(net.flow_limits)
    slack = net.slack_bus
    flows = v9._flows_np(exo, action, node, slack, ptdf)
    if not v9._feasible(flows, limits):
        action, _, _ = v9._safe_network_repair(action, node, ptdf, limits, exo, slack)
    return np.clip(action, lb, ub).tolist()


def run(policy_fn, ch, is_module=False):
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    for t in range(ch.num_steps):
        a = policy_fn(view, state)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return state.total_profit


if __name__ == "__main__":
    for scen_name, scen, nonces in [("BASELINE", Scenario.BASELINE, list(range(0, 50, 5))),
                                      ("CONGESTED", Scenario.CONGESTED, list(range(1, 51, 5)))]:
        qbs, q9s = [], []
        for nonce in nonces:
            seed = bch.seed_from_master_nonce(123, nonce)
            ch = Challenge.generate_instance(seed, Track(s=scen))
            _, base_p = ch.compute_baseline()
            v9._CACHE.clear()
            pb = run(policy_arch_b, ch)
            p9 = run(v9.policy, ch)
            qb = max(-10.0, min((pb - base_p) / (base_p + 1e-6), 10.0))
            q9 = max(-10.0, min((p9 - base_p) / (base_p + 1e-6), 10.0))
            qbs.append(qb); q9s.append(q9)
        print(f"{scen_name}: mean_archB={np.mean(qbs):+.4f}  mean_v9={np.mean(q9s):+.4f}  diff={np.mean(qbs)-np.mean(q9s):+.4f}")
