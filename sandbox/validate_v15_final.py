"""Phase 5 final validation checklist for policy_v15.py. Single process."""
import sys
from pathlib import Path
import random

import numpy as np

_SANDBOX_DIR = Path(__file__).resolve().parent
_ID7_DIR = _SANDBOX_DIR.parent
_PKGROOT = _SANDBOX_DIR / "_pkgroot"
sys.path.insert(0, str(_PKGROOT))
sys.path.insert(0, str(_ID7_DIR))

from competition.energy_arbitrage.python.challenge import Challenge, Track, NextRTPricesGenerate
from competition.energy_arbitrage.python.scenarios import Scenario

import policy_v15 as v15

SCENARIO_ORDER = [Scenario.BASELINE, Scenario.CONGESTED, Scenario.MULTIDAY, Scenario.DENSE, Scenario.CAPSTONE]
_SEED_NONCE_MUL = 0xDEADBEEFCAFEBABE


def seed_from_master_nonce(master_seed, nonce):
    mixed = (nonce * _SEED_NONCE_MUL) & 0xFFFFFFFFFFFFFFFF
    val = (master_seed & 0xFFFFFFFFFFFFFFFF) ^ mixed
    return val.to_bytes(8, "little") + (b"\x00" * 24)


def run_instance(seed_master, nonce, check_bounds=True):
    scen = SCENARIO_ORDER[nonce % 5]
    seed = seed_from_master_nonce(seed_master, nonce)
    ch = Challenge.generate_instance(seed, Track(s=scen))
    net = ch.network
    ptdf = np.asarray(net.ptdf)
    limits = np.asarray(net.flow_limits)
    slack = net.slack_bus
    node = np.array([b.node for b in ch.batteries])
    exo_full = np.asarray(ch.exogenous_injections)

    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    actions = []
    for t in range(ch.num_steps):
        a = v15.policy(view, state)
        arr = np.asarray(a, dtype=float)
        if check_bounds:
            assert np.all(np.isfinite(arr)), f"NaN/Inf action at t={t}: {arr}"
            bounds = np.asarray(state.action_bounds)
            lb, ub = bounds[:, 0], bounds[:, 1]
            assert np.all(arr >= lb - 1e-6) and np.all(arr <= ub + 1e-6), \
                f"action bound violation at t={t}: lb={lb} ub={ub} a={arr}"
            exo_t = np.array(exo_full[t], dtype=float)
            inj = exo_t.copy()
            inj[slack] = 0.0
            np.add.at(inj, node, arr)
            inj[slack] = -(inj.sum() - inj[slack])
            flows = ptdf @ inj
            assert np.all(np.abs(flows) <= limits * 1.000001 + 1e-6), f"flow violation at t={t}"
        actions.append(list(a))
        next_seed = bytes([rng.randint(0, 255) for _ in range(32)])
        state = ch.take_step(state, a, NextRTPricesGenerate(next_seed))
    return actions, state.total_profit


def main():
    results = {}

    print("=== 1. Syntax/import check ===")
    print("PASS (module already imported cleanly)")

    print("\n=== 2. Fresh-process style + bounds/NaN/flow checks across all 5 scenarios ===")
    for nonce in range(5):
        actions, profit = run_instance(987654, nonce)
        scen = SCENARIO_ORDER[nonce % 5]
        print(f"  {scen.name:10s}: profit={profit:.1f}  PASS (no NaN/Inf, no bound/flow violations)")

    print("\n=== 3. Forward-order / reverse-order / cache-clear determinism (target nonce=137) ===")
    TARGET = 137
    MASTER = 987654
    actions_fresh, profit_fresh = run_instance(MASTER, TARGET, check_bounds=False)
    for n in range(0, 30):
        run_instance(MASTER, n, check_bounds=False)
    actions_fwd, profit_fwd = run_instance(MASTER, TARGET, check_bounds=False)
    v15._CACHE.clear()
    for n in reversed(range(0, 30)):
        run_instance(MASTER, n, check_bounds=False)
    actions_rev, profit_rev = run_instance(MASTER, TARGET, check_bounds=False)
    v15._CACHE.clear()
    actions_clr, profit_clr = run_instance(MASTER, TARGET, check_bounds=False)

    def cmp(name, a, p):
        md = max((np.max(np.abs(np.array(x) - np.array(y))) for x, y in zip(actions_fresh, a)), default=0.0)
        status = "PASS" if md < 1e-9 and abs(p - profit_fresh) < 1e-6 else "FAIL"
        print(f"  {name}: max|diff|={md:.3e} profit_diff={abs(p-profit_fresh):.3e}  {status}")
        return status == "PASS"

    ok1 = cmp("after 30 forward", actions_fwd, profit_fwd)
    ok2 = cmp("after 30 reversed", actions_rev, profit_rev)
    ok3 = cmp("cache cleared", actions_clr, profit_clr)

    print("\n=== 4. Repeated-call determinism (same state, called twice) ===")
    seed = seed_from_master_nonce(987654, 1)
    ch = Challenge.generate_instance(seed, Track(s=Scenario.CONGESTED))
    v15._CACHE.clear()
    view = ch.to_policy_view()
    rng = random.Random()
    rng.seed(ch._hidden_seed)
    state = ch._initial_state(rng)
    a1 = v15.policy(view, state)
    a2 = v15.policy(view, state)
    det_ok = np.allclose(a1, a2, atol=0.0)
    print(f"  repeated call on identical state: max|diff|={np.max(np.abs(np.array(a1)-np.array(a2))):.3e}  "
          f"{'PASS' if det_ok else 'FAIL'}")

    print("\n=== 5. File size check ===")
    fsize = len(open(_ID7_DIR / "policy_v15.py").read())
    print(f"  policy_v15.py: {fsize} characters  {'PASS' if fsize < 50000 else 'FAIL'}")

    all_pass = ok1 and ok2 and ok3 and det_ok and fsize < 50000
    print(f"\n{'=== ALL CHECKS PASS ===' if all_pass else '=== SOME CHECKS FAILED ==='}")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
