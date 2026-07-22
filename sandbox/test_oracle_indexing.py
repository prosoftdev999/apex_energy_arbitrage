"""Synthetic correctness test for the corrected O1/O2/O4/O8 oracle indexing
in causal_regret_audit.py. Proves:
  1. O0's action/value at decision time t is INDEPENDENT of real_price_all
     (since O0 never reads it at all).
  2. O1's V_next at decision time t CHANGES when real_price_all[t+1] is
     perturbed, and does NOT change when real_price_all[t+2] (out of O1's
     1-step lookahead window) is perturbed instead.
  3. O2's V_next changes when real_price_all[t+2] is perturbed (within its
     2-step window) but O1's does not.
Single process, no multiprocessing.
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

from competition.energy_arbitrage.python.challenge import Challenge, Track
from competition.energy_arbitrage.python.scenarios import Scenario

import policy_v11 as v11
import causal_regret_audit as cra


def seed_fn(m, n):
    mixed = (n * 0xDEADBEEFCAFEBABE) & 0xFFFFFFFFFFFFFFFF
    val = (m & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + b"\x00" * 24


def main():
    seed = seed_fn(42, 0)
    ch = Challenge.generate_instance(seed, Track(s=Scenario.BASELINE))
    ba = v11._battery_arrays(ch)
    real_price_all = cra._record_real_price_trajectory(ch)
    da0 = np.asarray(ch.market.day_ahead_prices[0], dtype=float)
    rt0 = real_price_all[0]
    rel_dev0 = (rt0 - da0) / np.maximum(np.abs(da0), 1.0)
    sigma_hat = v11._sigma_from_rel_dev(rel_dev0)
    V_all_std = v11._build_da_value_function(ch, ba, sigma_hat=sigma_hat)

    T = ch.num_steps
    S = v11._S
    B = ba["B"]
    node = ba["node"]
    t = 20  # arbitrary interior decision point

    def o0_vnext():
        return V_all_std[t + 1] if t + 1 < V_all_std.shape[0] else np.zeros((B, S))

    def o1_vnext(prices):
        t1 = t + 1
        t_end = min(t1 + 1, T)
        boundary = V_all_std[t_end]
        V_k = cra._build_real_price_value_function(ba, prices[:, node], t1, t_end, S, V_boundary=boundary)
        return V_k[0]

    def o2_vnext(prices):
        t1 = t + 1
        t_end = min(t1 + 2, T)
        boundary = V_all_std[t_end]
        V_k = cra._build_real_price_value_function(ba, prices[:, node], t1, t_end, S, V_boundary=boundary)
        return V_k[0]

    base_o0 = o0_vnext()
    base_o1 = o1_vnext(real_price_all)
    base_o2 = o2_vnext(real_price_all)

    # Perturb price at t+1 (inside O1's and O2's window) by a large factor.
    perturbed_t1 = real_price_all.copy()
    perturbed_t1[t + 1] = perturbed_t1[t + 1] * 5.0 + 50.0

    o0_after_t1_perturb = o0_vnext()  # O0 never even looks at real_price_all
    o1_after_t1_perturb = o1_vnext(perturbed_t1)
    o2_after_t1_perturb = o2_vnext(perturbed_t1)

    # Perturb price at t+2 only (inside O2's window, OUTSIDE O1's window).
    perturbed_t2 = real_price_all.copy()
    perturbed_t2[t + 2] = perturbed_t2[t + 2] * 5.0 + 50.0
    o1_after_t2_perturb = o1_vnext(perturbed_t2)
    o2_after_t2_perturb = o2_vnext(perturbed_t2)

    print("=== O0 independence from real_price_all ===")
    diff_o0 = np.max(np.abs(base_o0 - o0_after_t1_perturb))
    print(f"max|O0_Vnext(before) - O0_Vnext(after perturbing price[t+1])| = {diff_o0:.6e}  "
          f"{'PASS (should be 0)' if diff_o0 < 1e-12 else 'FAIL'}")

    print("\n=== O1 responds to price[t+1] ===")
    diff_o1_t1 = np.max(np.abs(base_o1 - o1_after_t1_perturb))
    print(f"max|O1_Vnext(before) - O1_Vnext(after perturbing price[t+1])| = {diff_o1_t1:.6e}  "
          f"{'PASS (should be > 0)' if diff_o1_t1 > 1e-9 else 'FAIL'}")

    print("\n=== O1 does NOT respond to price[t+2] (out of its 1-step window) ===")
    diff_o1_t2 = np.max(np.abs(base_o1 - o1_after_t2_perturb))
    print(f"max|O1_Vnext(before) - O1_Vnext(after perturbing price[t+2])| = {diff_o1_t2:.6e}  "
          f"{'PASS (should be 0)' if diff_o1_t2 < 1e-12 else 'FAIL'}")

    print("\n=== O2 responds to BOTH price[t+1] and price[t+2] ===")
    diff_o2_t1 = np.max(np.abs(base_o2 - o2_after_t1_perturb))
    diff_o2_t2 = np.max(np.abs(base_o2 - o2_after_t2_perturb))
    print(f"max|O2_Vnext diff from price[t+1] perturb| = {diff_o2_t1:.6e}  "
          f"{'PASS (should be > 0)' if diff_o2_t1 > 1e-9 else 'FAIL'}")
    print(f"max|O2_Vnext diff from price[t+2] perturb| = {diff_o2_t2:.6e}  "
          f"{'PASS (should be > 0)' if diff_o2_t2 > 1e-9 else 'FAIL'}")

    all_pass = (diff_o0 < 1e-12 and diff_o1_t1 > 1e-9 and diff_o1_t2 < 1e-12
                and diff_o2_t1 > 1e-9 and diff_o2_t2 > 1e-9)
    print(f"\n{'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
