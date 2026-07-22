"""Section 4 evidence gate: does the stochastic (quadrature) Bellman value
differ MATERIALLY from a deterministic mean-price Bellman value, changing
selected actions on at least 5% of unsaturated BASELINE/CONGESTED
battery-steps? Single process.

A: deterministic mean-price Bellman (sigma_hat=0 path in policy_v11)
B: stochastic quadrature Bellman (sigma_hat>0 path -- what policy_v11
   ALREADY does for B<=20; this audit quantifies its real effect rather
   than assuming it)
C: perfect-foresight oracle (already-validated oracle2)
"""
import sys
from pathlib import Path
import random

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent.parent
_ID7_DIR = _SANDBOX_DIR.parent
sys.path.insert(0, str(_SANDBOX_DIR / "_pkgroot"))
sys.path.insert(0, str(_ID7_DIR))
sys.path.insert(0, str(_SANDBOX_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import benchmark as bch
import policy_v11 as v11
import oracle_certified as oc

_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def run_with_sigma(ch, sigma_override):
    """Run policy_v11's exact pipeline, but force sigma_hat to a fixed value
    (0.0 for deterministic A, or the real per-scenario sigma for B) instead
    of the causal MAD estimate, isolating JUST the stochastic-vs-deterministic
    Vc effect (not conflated with sigma estimation noise)."""
    v11._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    actions = []
    n_decisions_differ = [0]
    n_total = [0]

    orig_build = v11._build_da_value_function

    def patched_build(challenge, ba, sigma_hat=0.0, S=v11._S):
        return orig_build(challenge, ba, sigma_hat=sigma_override, S=S)

    v11._build_da_value_function = patched_build
    try:
        for t in range(ch.num_steps):
            a = v11.policy(view, state)
            actions.append(list(a))
            next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
            state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    finally:
        v11._build_da_value_function = orig_build
    return actions, state.total_profit


def main():
    results = []
    for scen_name, scen, nonces in [("BASELINE", Scenario.BASELINE, [0, 5, 10, 15]),
                                      ("CONGESTED", Scenario.CONGESTED, [1, 6, 11, 16])]:
        sigma_real = {"BASELINE": 0.10, "CONGESTED": 0.15}[scen_name]
        for dev_seed in [42, 2025]:
            for nonce in nonces:
                seed = bch.seed_from_master_nonce(dev_seed, nonce)
                ch = Challenge.generate_instance(seed, Track(s=scen))
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    _, base = ch.compute_baseline()

                actions_a, profit_a = run_with_sigma(ch, 0.0)          # A: deterministic
                actions_b, profit_b = run_with_sigma(ch, sigma_real)   # B: stochastic quadrature

                n_steps = len(actions_a)
                n_differ = sum(1 for aa, ab in zip(actions_a, actions_b)
                                if np.max(np.abs(np.array(aa) - np.array(ab))) > 1e-6)
                pct_differ = n_differ / n_steps

                q_a = max(-10.0, min((profit_a - base) / (base + 1e-6), 10.0))
                q_b = max(-10.0, min((profit_b - base) / (base + 1e-6), 10.0))

                results.append(dict(scenario=scen_name, dev_seed=dev_seed, nonce=nonce,
                                     baseline=base, q_a=q_a, q_b=q_b, stochastic_gain=q_b - q_a,
                                     pct_steps_differ=pct_differ, saturated=(q_b >= 9.9999)))
                print(f"{scen_name:10s} seed={dev_seed} nonce={nonce:3d} q_A={q_a:+.4f} q_B={q_b:+.4f} "
                      f"gain={q_b-q_a:+.4f} pct_steps_differ={pct_differ:.1%}")

    print("\n=== Evidence gate summary ===")
    unsaturated = [r for r in results if not r["saturated"]]
    pct_differs = [r["pct_steps_differ"] for r in unsaturated]
    gains = [r["stochastic_gain"] for r in unsaturated]
    frac_meeting_5pct = np.mean([p >= 0.05 for p in pct_differs])
    print(f"n unsaturated instances: {len(unsaturated)}")
    print(f"mean pct steps where action differs (A vs B): {np.mean(pct_differs):.1%}")
    print(f"fraction of instances with >=5% steps differing: {frac_meeting_5pct:.1%}")
    print(f"mean stochastic_gain (B-A) on unsaturated instances: {np.mean(gains):+.4f}")
    gate_pass = frac_meeting_5pct >= 0.5  # majority of instances should clear the 5% bar
    print(f"\nEvidence gate (Section 4): {'PASS' if gate_pass else 'FAIL'} "
          f"(requirement: >=5% of battery-steps show action changes)")
    return gate_pass, results


if __name__ == "__main__":
    ok, _ = main()
    sys.exit(0 if ok else 1)
