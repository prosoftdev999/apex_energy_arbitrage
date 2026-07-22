# Independent Oracle Red-Team Audit

Instances: 100/100, solved optimally: 100, failed: 0

Current score (recomputed from CSV, clipped): 0.7339332 (reference: 0.7339332)

Independent red-team oracle score: 0.7503278 (previously reported via oracle_certified.py: 0.7503278)

Any replay bound violation: False
Any replay flow violation: False
Max |oracle objective - replay profit|: 42.4600744030904
Max primal (inequality) residual: 8.908784820960136e-10
Max equality (SOC recursion) residual: 5.573866701524821e-09
Max SOC-variable-vs-manual-recompute residual: 5.577469153195125e-09

## Per-scenario means

| Scenario | n | current_mean | oracle_mean | gain |
|---|---|---|---|---|
| BASELINE | 20 | 1.8216 | 1.9810 | +0.1594 |
| CONGESTED | 20 | 5.3405 | 5.6979 | +0.3574 |
| MULTIDAY | 20 | 9.9239 | 10.0000 | +0.0761 |
| DENSE | 20 | 9.6378 | 9.8375 | +0.1997 |
| CAPSTONE | 20 | 9.9728 | 10.0000 | +0.0272 |

## Top 20 current-to-oracle gap instances

| nonce | scenario | current_q | oracle_q | gap |
|---|---|---|---|---|
| 3 | DENSE | 8.2523 | 10.0000 | +1.7477 |
| 83 | DENSE | 7.6791 | 9.0359 | +1.3569 |
| 56 | CONGESTED | 7.8308 | 9.1695 | +1.3387 |
| 6 | CONGESTED | 7.0918 | 8.0771 | +0.9854 |
| 88 | DENSE | 6.8247 | 7.7141 | +0.8894 |
| 66 | CONGESTED | 3.7465 | 4.4116 | +0.6651 |
| 61 | CONGESTED | 9.3418 | 10.0000 | +0.6582 |
| 96 | CONGESTED | 6.2398 | 6.8542 | +0.6144 |
| 49 | CAPSTONE | 9.4565 | 10.0000 | +0.5435 |
| 5 | BASELINE | 5.0654 | 5.5971 | +0.5317 |
| 47 | MULTIDAY | 9.4936 | 10.0000 | +0.5064 |
| 37 | MULTIDAY | 9.5845 | 10.0000 | +0.4155 |
| 76 | CONGESTED | 3.1469 | 3.5581 | +0.4113 |
| 70 | BASELINE | 4.0403 | 4.4271 | +0.3868 |
| 41 | CONGESTED | 1.7089 | 2.0949 | +0.3860 |
| 31 | CONGESTED | 3.6252 | 3.9863 | +0.3611 |
| 51 | CONGESTED | 2.8801 | 3.2249 | +0.3448 |
| 17 | MULTIDAY | 9.6839 | 10.0000 | +0.3161 |
| 60 | BASELINE | 3.1978 | 3.4964 | +0.2986 |
| 81 | CONGESTED | 1.8478 | 2.1415 | +0.2937 |
