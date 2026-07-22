"""Phase 3 gate: is RT-DA residual causally predictable? Uses only past/
current information (no lookahead). Pooled across all 20 instances per
scenario at master_seed=123 for statistical power.
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

MASTER_SEED = 123


def record_residuals(ch):
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    T, N = ch.num_steps, ch.network.num_nodes
    da_all = np.asarray(ch.market.day_ahead_prices)
    rt = np.zeros((T, N))
    exo = np.zeros((T, N))
    for t in range(T):
        rt[t] = state.rt_prices
        exo[t] = state.exogenous_injections
        a = [0.0] * ch.num_batteries
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    da = da_all[:T]
    denom = np.maximum(np.abs(da), 1.0)
    residual = (rt - da) / denom
    return residual, exo, ch.network


def analyze(scenario, nonces, label):
    common_all, local_all = [], []
    congestion_flags_all, local_flags_all = [], []
    for nonce in nonces:
        seed = bch.seed_from_master_nonce(MASTER_SEED, nonce)
        ch = Challenge.generate_instance(seed, Track(s=scenario))
        residual, exo, net = record_residuals(ch)
        common = np.median(residual, axis=1)  # (T,)
        local = residual - common[:, None]      # (T,N)
        common_all.append(common)
        local_all.append(local)

        # PTDF congestion exposure: exogenous-only flow utilization per line,
        # then map to a per-node "max incident-line utilization" as a causal
        # (uses only exogenous data, known in advance) congestion-exposure proxy.
        ptdf = np.asarray(net.ptdf)
        limits = np.asarray(net.flow_limits)
        flows = exo @ ptdf.T  # (T, L)
        util = np.abs(flows) / np.maximum(limits, 1e-9)  # (T, L)
        node_util = np.zeros((residual.shape[0], net.num_nodes))
        for node in range(net.num_nodes):
            lines = net.node_incident_lines[node]
            if lines:
                node_util[:, node] = util[:, lines].max(axis=1)
        congestion_flags_all.append(node_util)

    common_pooled = np.concatenate(common_all)
    local_pooled = np.concatenate([l.flatten() for l in local_all])
    node_util_pooled = np.concatenate([c.flatten() for c in congestion_flags_all])
    local_flat_for_util = np.concatenate([l.flatten() for l in local_all])

    def autocorr(x, lag):
        if len(x) <= lag:
            return float("nan")
        x0 = x[:-lag] - x.mean()
        x1 = x[lag:] - x.mean()
        denom = np.sqrt(np.sum(x0 ** 2) * np.sum(x1 ** 2))
        return float(np.sum(x0 * x1) / denom) if denom > 0 else float("nan")

    print(f"\n=== {label}: RT-DA residual predictability (pooled, n_instances={len(nonces)}) ===")
    print(f"common[t] (median-across-nodes) factor:")
    for lag in (1, 2, 4):
        print(f"  lag-{lag} autocorrelation: {autocorr(common_pooled, lag):+.4f}")
    print(f"local[t,node] (node-specific residual after removing common):")
    # local autocorrelation computed within each instance/node series separately then averaged
    local_lag1 = []
    for l in local_all:
        for node in range(l.shape[1]):
            local_lag1.append(autocorr(l[:, node], 1))
    print(f"  mean lag-1 autocorrelation across all (instance,node) series: {np.nanmean(local_lag1):+.4f}")

    # mean reversion: regress common[t] on common[t-1]
    x = common_pooled[:-1] - common_pooled.mean()
    y = common_pooled[1:] - common_pooled.mean()
    slope = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")
    print(f"  mean-reversion slope (common[t] ~ common[t-1]): {slope:+.4f} (1.0=random walk, 0=no memory)")

    # sign persistence
    sign_seq = np.sign(common_pooled)
    sign_persist = float(np.mean(sign_seq[1:] == sign_seq[:-1]))
    print(f"  sign persistence P(sign[t]==sign[t-1]): {sign_persist:.4f} (0.5 = no persistence)")

    # spike clustering: correlation of |common[t]| with |common[t-1]|
    abs_common = np.abs(common_pooled)
    spike_corr = autocorr(abs_common, 1)
    print(f"  |common[t]| vs |common[t-1]| correlation (spike clustering): {spike_corr:+.4f}")

    # conditional expectation after large residual
    thresh = np.percentile(np.abs(common_pooled), 90)
    after_pos = common_pooled[1:][common_pooled[:-1] > thresh]
    after_neg = common_pooled[1:][common_pooled[:-1] < -thresh]
    print(f"  E[common[t+1] | common[t] > P90={thresh:.3f}]: {after_pos.mean() if len(after_pos) else float('nan'):+.4f} (n={len(after_pos)})")
    print(f"  E[common[t+1] | common[t] < -P90]: {after_neg.mean() if len(after_neg) else float('nan'):+.4f} (n={len(after_neg)})")
    print(f"  unconditional E[common[t+1]]: {common_pooled.mean():+.4f}")

    # relationship with congestion exposure (causal, from exogenous data)
    valid = np.isfinite(local_flat_for_util) & np.isfinite(node_util_pooled)
    if valid.sum() > 10:
        corr = np.corrcoef(local_flat_for_util[valid], node_util_pooled[valid])[0, 1]
        print(f"  corr(local residual magnitude proxy, PTDF congestion exposure): {corr:+.4f}")
        corr_abs = np.corrcoef(np.abs(local_flat_for_util[valid]), node_util_pooled[valid])[0, 1]
        print(f"  corr(|local residual|, PTDF congestion exposure): {corr_abs:+.4f}")


if __name__ == "__main__":
    baseline_nonces = list(range(0, 100, 5))
    congested_nonces = list(range(1, 100, 5))
    analyze(Scenario.BASELINE, baseline_nonces, "BASELINE")
    analyze(Scenario.CONGESTED, congested_nonces, "CONGESTED")
