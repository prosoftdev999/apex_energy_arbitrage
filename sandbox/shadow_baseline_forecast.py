"""Section 2/15: forward-projected (episode-start) shadow-baseline estimate.

Unlike shadow_baseline_validate.py (which proved EXACT retrospective shadow
tracking using already-realized RT prices), this estimates the baseline
BEFORE the episode starts, using DA prices as a stand-in for the still-
unknown future RT prices (E[RT] ~= DA under the no-congestion/no-jump
component; the jump/congestion mean biases are small and were already
found immaterial to counter-productive when used as a policy decision
input in this session's earlier work). Since greedy/conservative's ACTION
CHOICE never depends on RT prices at all (only compute_profit, for cost
accounting, does), the entire action schedule is deterministic and known
upfront -- only the PROFIT estimate is approximate.

Reports the actual estimation error against the true (hindsight) baseline.
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

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate, State
from competition.energy_arbitrage.python.scenarios import Scenario
from competition.energy_arbitrage.python.greedy import policy as greedy_policy
from competition.energy_arbitrage.python.conservative import policy as conservative_policy

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def forecast_baseline_at_episode_start(ch):
    """Simulate greedy/conservative forward using DA price as the E[RT]
    stand-in -- no realized RT prices used anywhere. Returns
    (estimated_greedy_profit, estimated_conservative_profit)."""
    view = ch.to_policy_view()
    da_all = ch.market.day_ahead_prices
    soc0 = [b.soc_initial_mwh for b in ch.batteries]
    bounds0 = [b.compute_action_bounds(s) for b, s in zip(ch.batteries, soc0)]

    def run_shadow(policy_fn):
        state = State(time_step=0, socs=list(soc0), rt_prices=list(da_all[0]),
                      exogenous_injections=list(ch.exogenous_injections[0]),
                      action_bounds=list(bounds0), total_profit=0.0)
        for t in range(ch.num_steps):
            a = policy_fn(view, state)
            profit = ch.compute_profit(state, a)  # uses state.rt_prices = DA[t] stand-in
            next_socs = [ch.batteries[i].apply_action_to_soc(a[i], state.socs[i]) for i in range(len(a))]
            next_bounds = [ch.batteries[i].compute_action_bounds(next_socs[i]) for i in range(len(next_socs))]
            nt = t + 1
            if nt < ch.num_steps:
                state = State(time_step=nt, socs=next_socs, rt_prices=list(da_all[nt]),
                               exogenous_injections=list(ch.exogenous_injections[nt]),
                               action_bounds=next_bounds, total_profit=state.total_profit + profit)
            else:
                state.total_profit += profit
        return state.total_profit

    return run_shadow(greedy_policy), run_shadow(conservative_policy)


def main():
    scenarios = [("BASELINE", Scenario.BASELINE, [0, 5, 10, 15]),
                 ("CONGESTED", Scenario.CONGESTED, [1, 6, 11, 16]),
                 ("MULTIDAY", Scenario.MULTIDAY, [2, 12]),
                 ("DENSE", Scenario.DENSE, [3, 13]),
                 ("CAPSTONE", Scenario.CAPSTONE, [4, 14])]
    errs = []
    for scen_name, scen, nonces in scenarios:
        for nonce in nonces:
            seed = seed_from_master_nonce(987654, nonce)
            ch = Challenge.generate_instance(seed, Track(s=scen))
            _, real_baseline = ch.compute_baseline()
            est_g, est_c = forecast_baseline_at_episode_start(ch)
            est_baseline = max(est_g, est_c)
            err = est_baseline - real_baseline
            rel_err = err / max(abs(real_baseline), 1.0)
            errs.append(rel_err)
            print(f"{scen_name:10s} nonce={nonce:3d} real={real_baseline:12.1f} est={est_baseline:12.1f} "
                  f"err={err:10.1f} rel_err={rel_err:+.4f}")
    errs = np.array(errs)
    print(f"\nrel_err stats: mean={errs.mean():+.4f} std={errs.std():.4f} "
          f"min={errs.min():+.4f} max={errs.max():+.4f}")


if __name__ == "__main__":
    main()
