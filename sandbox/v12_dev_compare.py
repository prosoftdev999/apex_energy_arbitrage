"""Compare policy_v11 vs policy_v12 on real replayed profit, BASELINE +
CONGESTED, held-out dev seeds (42, 2025) -- never used for tuning prior to
this session's quadrature work either. Reports quality per instance and the
mean gap, plus how many instances actually differed in action at all."""
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
import policy_v11 as v11
import policy_v12 as v12


def run(mod, ch):
    mod._CACHE.clear()
    view = ch.to_policy_view()
    import random
    rng = random.Random(); rng.seed(ch._hidden_seed)
    state = mod_state = ch._initial_state(rng)
    n_actions_differ = 0
    for t in range(ch.num_steps):
        a = mod.policy(view, state)
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return state.total_profit


def main():
    for dev_seed in (42, 2025):
        for scen_name, scen, nonces in [("BASELINE", Scenario.BASELINE, list(range(0, 20, 2))),
                                          ("CONGESTED", Scenario.CONGESTED, list(range(1, 21, 2)))]:
            q11s, q12s = [], []
            for nonce in nonces:
                seed = bch.seed_from_master_nonce(dev_seed, nonce)
                ch = Challenge.generate_instance(seed, Track(s=scen))
                _, base_p = ch.compute_baseline()
                p11 = run(v11, ch)
                p12 = run(v12, ch)
                q11 = max(-10.0, min((p11 - base_p) / (base_p + 1e-6), 10.0))
                q12 = max(-10.0, min((p12 - base_p) / (base_p + 1e-6), 10.0))
                q11s.append(q11); q12s.append(q12)
            print(f"dev_seed={dev_seed} {scen_name}: mean_v11={np.mean(q11s):+.4f} "
                  f"mean_v12={np.mean(q12s):+.4f} diff={np.mean(q12s)-np.mean(q11s):+.5f} "
                  f"n_worse={sum(1 for a,b in zip(q11s,q12s) if b<a-1e-9)}")


if __name__ == "__main__":
    main()
