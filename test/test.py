"""
Apex Energy Arbitrage local tester (compatible with policy_138231 style).

Usage:
  python test.py policy_138231.py --target-data apex_round66_eval_metadata.json --cases 10

This is a local mock tester, not the official Apex hidden evaluator.
It creates Challenge/State objects with the attribute names used by Apex docs:
  challenge.batteries[*].capacity_mwh, power_charge_mw, ...
  challenge.market.day_ahead_prices
  challenge.network.ptdf, flow_limits, compute_flows(...)
  state.time_step, socs, rt_prices, action_bounds, exogenous_injections
"""
from __future__ import annotations

# Standard-library imports only. This keeps the tester easy to run on Windows/Linux.

import argparse, importlib.util, json, math, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

# One Apex time step is 15 minutes = 0.25 hours.
DT = 0.25
TRANSACTION_COST = 0.25
DEGRADATION_COST = 1.00

# Mock versions of the 5 official scenario sizes shown in the Apex docs.
# Tuple format: (scenario_name, number_of_nodes, number_of_lines, number_of_batteries, time_steps).
SCENARIOS = [
    ("BASELINE", 20, 30, 10, 96),
    ("CONGESTED", 40, 60, 20, 96),
    ("MULTIDAY", 80, 120, 40, 192),
    ("DENSE", 100, 200, 60, 192),
    ("CAPSTONE", 150, 300, 100, 192),
]

@dataclass
class Battery:
    """Small mock battery object with the same attribute names Apex policies expect."""
    node: int
    capacity_mwh: float
    power_charge_mw: float
    power_discharge_mw: float
    efficiency_charge: float = 0.95
    efficiency_discharge: float = 0.95
    soc_min_mwh: float = 0.0
    soc_max_mwh: float = 0.0

    # extra aliases, for other submissions
    @property
    def capacity(self): return self.capacity_mwh
    @property
    def power_limit(self): return min(self.power_charge_mw, self.power_discharge_mw)
    @property
    def soc_min(self): return self.soc_min_mwh
    @property
    def soc_max(self): return self.soc_max_mwh
    @property
    def charge_efficiency(self): return self.efficiency_charge
    @property
    def discharge_efficiency(self): return self.efficiency_discharge

class Network:
    """Mock DC power-flow network.

    ptdf maps node injections to line flows. If any line flow is above its
    flow_limit, the action is treated as invalid and replaced with zeros.
    """
    def __init__(self, num_nodes, num_lines, ptdf, flow_limits, slack_bus=0):
        self.num_nodes = num_nodes
        self.num_lines = num_lines
        self.ptdf = ptdf
        self.flow_limits = flow_limits
        self.slack_bus = slack_bus

    def compute_flows(self, injections):
        flows = []
        for row in self.ptdf:
            flows.append(sum(row[i] * injections[i] for i in range(self.num_nodes)))
        return flows

class Challenge(SimpleNamespace):
    """Mock Apex challenge object passed into policy(challenge, state)."""
    def compute_total_injections(self, state, actions):
        inj = list(state.exogenous_injections)
        slack = self.network.slack_bus
        inj[slack] = 0.0
        for a, b in zip(actions, self.batteries):
            inj[b.node] += float(a)
        total = sum(inj) - inj[slack]
        inj[slack] = -total
        return inj

class State(SimpleNamespace):
    """Mock Apex state object passed into policy(challenge, state)."""
    pass

def load_submission(path):
    """Import the submitted .py file and return the module.

    The submitted file must define a function named:
        policy(challenge, state)
    """
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("submission_under_test", p)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_under_test"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "policy"):
        raise RuntimeError("Submission must define policy(challenge, state)")
    return mod

def make_prices(T, nodes, rng):
    """Create mock day-ahead and real-time prices.

    Day-ahead prices follow a daily pattern. Real-time prices are day-ahead
    prices plus noise, with occasional spikes/crashes. This is only a local
    approximation, not the hidden Apex generator.
    """
    da, rt = [], []
    for t in range(T):
        hour = (t % 96) / 4.0
        low_high_cycle = 42 + 20 * math.sin((hour - 14) / 24 * 2 * math.pi)
        evening = 40 * math.exp(-((hour - 18.0) ** 2) / 4.5)
        morning = 14 * math.exp(-((hour - 8.0) ** 2) / 5.0)
        base = low_high_cycle + evening + morning
        row_da, row_rt = [], []
        for n in range(nodes):
            loc = rng.uniform(-5, 5)
            p_da = max(0.0, base + loc + rng.gauss(0, 2.5))
            p_rt = max(0.0, p_da + rng.gauss(0, 8.0))
            # occasional real-time spike/crash
            if rng.random() < 0.015:
                p_rt += rng.uniform(25, 80)
            if rng.random() < 0.010:
                p_rt = max(0.0, p_rt - rng.uniform(20, 55))
            row_da.append(p_da)
            row_rt.append(p_rt)
        da.append(row_da); rt.append(row_rt)
    return da, rt

def make_case(name, nodes, lines_n, B, T, seed):
    """Build one complete mock Apex instance for a scenario and seed."""
    rng = random.Random(seed)
    batteries = []
    for _ in range(B):
        cap = rng.uniform(60, 220)
        pc = rng.uniform(8, 38)
        pd = rng.uniform(8, 38)
        batteries.append(Battery(
            node=rng.randrange(nodes), capacity_mwh=cap,
            power_charge_mw=pc, power_discharge_mw=pd,
            soc_min_mwh=0.10*cap, soc_max_mwh=0.90*cap,
        ))
    da, rt = make_prices(T, nodes, rng)
    exo = [[rng.gauss(0, 10) for _ in range(nodes)] for __ in range(T)]
    # Random PTDF, with safer limits. This creates some congestion but zero remains feasible.
    ptdf = [[rng.uniform(-0.30, 0.30) for _ in range(nodes)] for __ in range(lines_n)]
    limits = [rng.uniform(80, 260) for _ in range(lines_n)]
    network = Network(nodes, lines_n, ptdf, limits, slack_bus=0)
    market = SimpleNamespace(day_ahead_prices=da)
    ch = Challenge(
        scenario=name, num_nodes=nodes, num_batteries=B, num_steps=T,
        batteries=batteries, market=market, network=network,
        day_ahead_prices=da, exogenous_injections=exo,
    )
    return ch, rt

def action_bounds(socs, batteries):
    """Calculate legal charge/discharge bounds for every battery.

    Negative action means charging. Positive action means discharging.
    Bounds depend on current SOC and battery power limits.
    """
    out = []
    for s, b in zip(socs, batteries):
        max_charge_soc = max(0.0, (b.soc_max_mwh - s) / (b.efficiency_charge * DT))
        max_dis_soc = max(0.0, (s - b.soc_min_mwh) * b.efficiency_discharge / DT)
        out.append((-min(b.power_charge_mw, max_charge_soc), min(b.power_discharge_mw, max_dis_soc)))
    return out

def update_soc(socs, actions, batteries):
    """Update battery state-of-charge after applying actions for one step."""
    new = []
    for s, a, b in zip(socs, actions, batteries):
        if a < 0:
            s2 = s + (-a) * b.efficiency_charge * DT
        else:
            s2 = s - a * DT / b.efficiency_discharge
        new.append(min(max(s2, b.soc_min_mwh), b.soc_max_mwh))
    return new

def profit_step(actions, rt_node_prices, batteries):
    """Calculate one-step profit using the Apex formula.

    profit = action * real_time_price * DT
             - transaction_cost
             - degradation_cost
    """
    total = 0.0
    for a, b in zip(actions, batteries):
        p = rt_node_prices[b.node]
        total += a * p * DT
        total -= TRANSACTION_COST * abs(a) * DT
        total -= DEGRADATION_COST * ((abs(a) * DT / b.capacity_mwh) ** 2)
    return total

def greedy_action(ch, state):
    """Simple baseline policy used for local comparison.

    It discharges when current RT price is higher than future DA average,
    charges when current RT price is lower, otherwise does nothing.
    """
    acts = []
    future = ch.market.day_ahead_prices[state.time_step:min(ch.num_steps, state.time_step + 96)]
    for i, b in enumerate(ch.batteries):
        vals = [row[b.node] for row in future] or [state.rt_prices[b.node]]
        avg = sum(vals) / len(vals)
        lo, hi = state.action_bounds[i]
        p = state.rt_prices[b.node]
        if p > avg + 4:
            acts.append(hi)
        elif p < avg - 4:
            acts.append(lo)
        else:
            acts.append(0.0)
    return acts

def feasible(ch, state, actions):
    """Return True when actions do not violate network line-flow limits."""
    flows = ch.network.compute_flows(ch.compute_total_injections(state, actions))
    for f, lim in zip(flows, ch.network.flow_limits):
        if abs(f) > lim + 1e-6 * max(1.0, lim):
            return False
    return True

def score_quality(profit, baseline):
    """Calculate quality using the Apex docs formula.

    quality = (miner_profit - baseline_profit) / (baseline_profit + 1e-6)
    Then clamp quality to [-10, 10].
    """
    baseline = max(float(baseline), 1e-6)
    
    q = (profit - baseline) / (baseline + 1e-6)
    
    return max(-10.0, min(10.0, q))

def quality_int_from_quality(q):
    # Apex docs: quality_int = round(clamp(quality, -10, +10) * 1,000,000)
    return int(round(max(-10.0, min(10.0, float(q))) * 1_000_000))

def final_score_from_quality_ints(qints):
    # Apex final score shown = average raw quality_int / 10,000,000
    if not qints:
        return 0.0, 0.0
    raw = sum(qints) / len(qints)
    final = raw / 10_000_000
    return raw, final

def run_one(policy_func, scenario, seed, verbose=False):
    """Run one full scenario from t=0 to final time step.

    For each time step:
      1. Build State.
      2. Call user policy.
      3. Clean invalid outputs.
      4. Check network feasibility.
      5. Add profit and update SOC.
      6. Also run local baseline for comparison.
    """
    name, nodes, lines_n, B, T = scenario
    ch, rt_all = make_case(name, nodes, lines_n, B, T, seed)
    socs = [0.50 * b.capacity_mwh for b in ch.batteries]
    profit = baseline_profit = 0.0
    invalid = errors = 0
    total_time = 0.0
    for t in range(T):
        # Current legal action range for every battery.
        bounds = action_bounds(socs, ch.batteries)

        # Build a state object with several alias names because different
        # submissions may use slightly different attribute names.
        state = State(
            time_step=t, t=t, time=t,
            socs=list(socs), soc=list(socs), state_of_charge=list(socs),
            rt_prices=rt_all[t], real_time_prices=rt_all[t], prices=rt_all[t],
            action_bounds=bounds,
            action_low=[x[0] for x in bounds], action_high=[x[1] for x in bounds],
            feasible_action_bounds=bounds,
            exogenous_injections=ch.exogenous_injections[t],
            accumulated_profit=profit, profit=profit,
        )
        # Time the policy call and catch runtime errors so the local test
        # can continue instead of crashing.
        st = time.time()
        try:
            actions = policy_func(ch, state)
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  error t={t}: {type(e).__name__}: {e}")
            actions = [0.0] * B
        total_time += time.time() - st
        # Convert the policy output into a list of floats and repair common
        # invalid outputs: wrong type, wrong length, NaN/inf, or out-of-bounds.
        try:
            actions = list(actions)
        except Exception:
            invalid += 1; actions = [0.0]*B
        if len(actions) != B:
            invalid += 1; actions = (actions + [0.0]*B)[:B]
        clean = []
        for i, a in enumerate(actions):
            lo, hi = bounds[i]
            try: a = float(a)
            except Exception: invalid += 1; a = 0.0
            if not math.isfinite(a): invalid += 1; a = 0.0
            if a < lo - 1e-7 or a > hi + 1e-7:
                invalid += 1; a = max(lo, min(hi, a))
            clean.append(a)
        # If the action causes a line-flow violation, count it invalid and
        # replace it with zero actions. Zero is normally network-feasible.
        if not feasible(ch, state, clean):
            invalid += 1
            clean = [0.0] * B
        # Apply cleaned user action to profit and battery SOC.
        profit += profit_step(clean, rt_all[t], ch.batteries)
        socs = update_soc(socs, clean, ch.batteries)

        # Calculate local baseline profit for the same state.
        g = greedy_action(ch, state)
        if not feasible(ch, state, g):
            g = [0.0] * B
        baseline_profit += profit_step(g, rt_all[t], ch.batteries)
    quality = score_quality(profit, max(baseline_profit, 1e-6))
    print( "quality:", quality)
    return {
        "scenario": name, "seed": seed, "profit": profit,
        "baseline": max(baseline_profit, 1e-6),
        "quality": quality,
        "quality_int": quality_int_from_quality(quality),
        "final_part": quality_int_from_quality(quality) / 10_000_000,
        "invalid": invalid, "errors": errors,
        "avg_call_ms": 1000*total_time/max(T,1),
    }

def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--target-data")
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=590682929)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    mod = load_submission(args.submission)
    """print(f"Testing: {Path(args.submission).resolve()}")"""
    """print("Local mock only; official Apex score can be different.\n")"""
    if args.target_data:
        data = json.load(open(args.target_data))
        ev = data.get("eval", [])
        if ev:
            qints = [int(round(float(x.get("quality", 0)))) for x in ev]
            raw, final = final_score_from_quality_ints(qints)
            """print(f"Target file: {len(ev)} Apex instances")
            print(f"Official Raw Score from metadata:   {raw:.1f}")
            print(f"Official Final Score from metadata: {final:.7f}\n")"""
    results = []
    for k in range(args.cases):
        r = run_one(mod.policy, SCENARIOS[k % len(SCENARIOS)], args.seed + k, args.verbose)
        results.append(r)
        """print(f"{k+1:02d} {r['scenario']:<9} seed={r['seed']} profit={r['profit']:11.2f} baseline={r['baseline']:11.2f} quality={r['quality']:7.4f} q_int={r['quality_int']:9d} invalid={r['invalid']:3d} errors={r['errors']:3d} avg_call={r['avg_call_ms']:8.3f}ms")"""
    # Average quality_int values to mimic the Apex score display style.
    qints = [r["quality_int"] for r in results]
    raw_score, final_score = final_score_from_quality_ints(qints)
    avgq = sum(r['quality'] for r in results)/len(results)
    print("\nSubmission Details Style Summary")
    print(f"Raw Score:       {raw_score:.1f}")
    print(f"Final Score:     {final_score:.7f}")
    print(f"Average quality: {avgq:.6f}")
    print(f"Total invalid actions: {sum(r['invalid'] for r in results)}")
    print(f"Total runtime errors:  {sum(r['errors'] for r in results)}")
    print(f"Worst avg call time:   {max(r['avg_call_ms'] for r in results):.3f} ms")

if __name__ == "__main__":
    main()
