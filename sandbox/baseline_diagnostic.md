# BASELINE root-cause diagnostic (policy_v2, dev seed=42, 20 instances)

## 4. Network congestion / repair impact (BASELINE only)

- repair codepath engaged on 51.3% of battery-steps (episode-level infeasibility detected)
- this battery's OWN action was actually changed on 19.5% of battery-steps
- total |desired action| removed by repair: 70127.0 MW-steps out of 229517.0 MW-steps desired (30.55%)

## 5. Opportunity loss (local best vs chosen feasible action)

- mean opportunity loss per battery-step: 5.1297
- mean opportunity loss on repaired steps only: 26.3343
- mean opportunity loss on NON-repaired steps: -0.000000  (should be ~0 if repair is the only source of loss)
- total opportunity loss across all 19200 battery-steps: 98490.3

## 3. Battery utilization / ending SOC

- ending mean SOC%: mean=10.16  min=10.00  max=11.63  (SOC floor = 10%)
- instances ending above 15% mean SOC: 0 / 20

## 6. DP marginal value of stored energy vs realized price

- mean marginal_value(dVc/dSOC) over full horizon: 47.252  vs mean rt_price: 49.962
- mean marginal_value in FINAL 8 steps only: 24.764  vs mean rt_price in final 8 steps: 31.625
- battery-steps holding (action~0) while SOC>15%: 7127 (37.12% of all steps)
  -> in these steps, mean marginal_value=52.426 vs mean rt_price=51.916 (marginal_value > price would rationally justify holding)

## 7. Candidate-action usage histogram (unconstrained/desired action)

- zero (hold): 47.6%
- nonzero actions: 52.4% (magnitudes vary continuously with SOC-dependent bounds; grid is 7-point: full/2-3/1-3 charge, zero, 1-3/2-3/full discharge)

## 2. Comparison against greedy baseline

- action DIRECTION agrees with greedy on 50.8% of battery-steps
- we charge / greedy doesn't: 2310 steps
- we discharge / greedy doesn't: 3193 steps
- greedy charges / we don't: 1605 steps
- greedy discharges / we don't: 2756 steps
- network-repair-removed-action steps: 3740 (19.48%)

## Per-instance summary

| nonce | quality | my_profit | greedy_profit | baseline_profit | ending_mean_soc% | repairs |
|---|---|---|---|---|---|---|
| 0 | +0.438 | 43468.9 | 30227.6 | 30227.6 | 10.0 | 44 |
| 5 | +1.180 | 35614.9 | 16334.2 | 16334.2 | 10.0 | 49 |
| 10 | +0.466 | 58427.9 | 39843.8 | 39843.8 | 10.0 | 40 |
| 15 | +2.418 | 43222.0 | 12645.3 | 12645.3 | 10.5 | 74 |
| 20 | +2.201 | 40998.3 | 12807.4 | 12807.4 | 10.0 | 44 |
| 25 | +1.299 | 35711.7 | 15532.4 | 15532.4 | 10.0 | 57 |
| 30 | +0.730 | 41349.6 | 23907.7 | 23907.7 | 10.0 | 37 |
| 35 | +0.613 | 46185.4 | 28626.6 | 28626.6 | 10.0 | 35 |
| 40 | +10.000 | 31150.9 | -213.6 | 0.0 | 10.0 | 48 |
| 45 | +5.245 | 42769.0 | 6848.6 | 6848.6 | 10.3 | 66 |
| 50 | +6.754 | 46186.2 | 5956.7 | 5956.7 | 10.5 | 39 |
| 55 | +0.441 | 48806.2 | 33865.4 | 33865.4 | 10.0 | 34 |
| 60 | +0.806 | 41411.9 | 22926.7 | 22926.7 | 10.0 | 57 |
| 65 | +0.608 | 44878.0 | 27913.2 | 27913.2 | 10.0 | 47 |
| 70 | +1.363 | 46774.5 | 19792.6 | 19792.6 | 10.4 | 49 |
| 75 | +0.991 | 43237.4 | 21713.3 | 21713.3 | 10.0 | 46 |
| 80 | +0.618 | 57355.0 | 35447.0 | 35447.0 | 10.0 | 48 |
| 85 | +0.872 | 52725.9 | 28172.7 | 28172.7 | 10.0 | 54 |
| 90 | +5.029 | 41900.9 | 6950.4 | 6950.4 | 10.0 | 60 |
| 95 | +0.331 | 41183.4 | 30953.2 | 30953.2 | 11.6 | 57 |