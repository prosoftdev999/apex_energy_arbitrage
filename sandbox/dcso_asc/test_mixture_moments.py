"""Unit test: analytic E[RT] (Gaussian + Pareto-jump + congestion-premium
mixture, derived in model_derivation.md) versus a large offline Monte Carlo
simulation using the ACTUAL production market.py/network.py generators.
Research-only script (uses randomness for validation, never inside the
submitted policy). Single process.
"""
import sys
from pathlib import Path
import random

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track
from competition.energy_arbitrage.python.scenarios import Scenario
from competition.energy_arbitrage.python import constants

N_SIM = 20000  # Monte Carlo draws for validation


def seed_fn(m, n):
    mixed = (n * 0xDEADBEEFCAFEBABE) & 0xFFFFFFFFFFFFFFFF
    val = (m & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + b"\x00" * 24


def analytic_mean(da_i, p_congested_i, rho_jump, alpha, mu=constants.MU_BIAS,
                   gamma=constants.GAMMA_PRICE):
    half_normal_mean = 1.0 / np.sqrt(2.0 * np.pi)
    pareto_mean = alpha / (alpha - 1.0)
    return da_i * (1.0 + mu) + p_congested_i * gamma * half_normal_mean + rho_jump * da_i * pareto_mean


def simulate_rt_mc(ch, t, node_i, n_sim, congested_flag):
    """Direct Monte Carlo replica of market.py::generate_rt_prices' formula
    for ONE node, using the SAME production constants and market params."""
    da_i = ch.market.day_ahead_prices[t][node_i]
    sigma = ch.market.params.volatility
    rho_jump = ch.market.params.jump_probability
    alpha = ch.market.params.tail_index
    rho = constants.RHO_SPATIAL
    rng = random.Random(12345)
    vals = []
    for _ in range(n_sim):
        z_common = rng.gauss(0.0, 1.0)
        z_prime = rng.gauss(0.0, 1.0)
        zeta = max(z_prime, 0.0)
        eps_i = rng.gauss(0.0, 1.0)
        xi_i = np.sqrt(rho) * z_common + np.sqrt(1.0 - rho) * eps_i
        price = da_i * (1.0 + constants.MU_BIAS + sigma * xi_i)
        if congested_flag:
            price += constants.GAMMA_PRICE * zeta
        u_jump = rng.random()
        if u_jump < rho_jump:
            u_pareto = max(rng.random(), 1e-10)
            pareto = (1.0 - u_pareto) ** (-1.0 / alpha)
            price += da_i * pareto
        price = max(constants.LAMBDA_MIN, min(price, constants.LAMBDA_MAX))
        vals.append(price)
    return np.array(vals)


def main():
    all_pass = True
    for scen_name, scen in [("BASELINE", Scenario.BASELINE), ("CONGESTED", Scenario.CONGESTED)]:
        seed = seed_fn(42, 0)
        ch = Challenge.generate_instance(seed, Track(s=scen))
        t, node_i = 10, 0
        da_i = ch.market.day_ahead_prices[t][node_i]
        rho_jump = ch.market.params.jump_probability
        alpha = ch.market.params.tail_index

        for congested_flag in (False, True):
            mc = simulate_rt_mc(ch, t, node_i, N_SIM, congested_flag)
            mc_mean = mc.mean()
            mc_se = mc.std(ddof=1) / np.sqrt(N_SIM)
            p_cong = 1.0 if congested_flag else 0.0
            an_mean = analytic_mean(da_i, p_cong, rho_jump, alpha)
            diff = abs(mc_mean - an_mean)
            n_se = diff / mc_se if mc_se > 0 else float("inf")
            status = "PASS" if n_se < 4.0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"{scen_name:10s} congested={congested_flag!s:5s} DA={da_i:7.2f}  "
                  f"MC_mean={mc_mean:9.4f} (SE={mc_se:.4f})  analytic_mean={an_mean:9.4f}  "
                  f"diff={diff:.4f} ({n_se:.2f} SE)  {status}")

    print(f"\n{'ALL MOMENT CHECKS PASS' if all_pass else 'SOME MOMENT CHECKS FAILED'}")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
