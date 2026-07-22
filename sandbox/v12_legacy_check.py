"""Confirm policy_v12 produces byte-identical actions to policy_v11 on
MULTIDAY/DENSE/CAPSTONE (B>20, out of _LEX_MAX_B scope) -- required before
any scored test, per project rule of never touching validated scenarios."""
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

for scen_name, scen in [("MULTIDAY", Scenario.MULTIDAY), ("DENSE", Scenario.DENSE), ("CAPSTONE", Scenario.CAPSTONE)]:
    seed = bch.seed_from_master_nonce(42, 0)
    ch = Challenge.generate_instance(seed, Track(s=scen))
    for mod in (v11, v12):
        mod._CACHE.clear()
    import random
    view = ch.to_policy_view()
    rng11 = random.Random(); rng11.seed(ch._hidden_seed)
    state11 = ch._initial_state(rng11)
    rng12 = random.Random(); rng12.seed(ch._hidden_seed)
    state12 = ch._initial_state(rng12)
    max_diff = 0.0
    for t in range(min(8, ch.num_steps)):
        a11 = np.array(v11.policy(view, state11))
        a12 = np.array(v12.policy(view, state12))
        max_diff = max(max_diff, float(np.max(np.abs(a11 - a12))) if len(a11) else 0.0)
        next_seed = bytes([rng11.randint(0, 255) for _ in range(32)])
        state11 = ch.take_step(state11, a11.tolist(), NextRTPricesGenerate(next_seed))
        state12 = ch.take_step(state12, a12.tolist(), NextRTPricesGenerate(next_seed))
    print(f"{scen_name}: B={ch.num_batteries} max|a11-a12| over 8 steps = {max_diff:.3e}")
