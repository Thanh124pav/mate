# Coverage-rate gain correlation tests

Coverage uses `custom_metrics/coverage_rate_mean`. At each FOCUSS metric timestep, both FOCUSS and backbone coverage are aligned by last observation carried forward; values outside a run's observed range are clamped to its first/last observation. Gain is FOCUSS minus backbone.

Circular-shift tests use 9,999 permutations and preserve within-run temporal autocorrelation. BH q-values correct the Pearson circular tests across all reported algorithm/metric combinations. Detrended r removes a separate linear timestep trend within each algorithm.

## Run pairs

| Algorithm | Backbone run | FOCUSS run | Seed | Coverage metric | n | Min timestep | Max timestep |
|---|---|---|---|---|---|---|---|
| facmac_focuss | f04b7_00000 | f052d_00000 | 0 | custom_metrics/coverage_rate_mean | 304 | 5777 | 1780269 |
| resq_focuss | cdbcf_00000 | resq_focuss-1494505-8463bc34 | 0 | custom_metrics/coverage_rate_mean | 143 | 878818 | 1712494 |
| riskq_focuss | riskq-1494503-09472b1c | riskq_focuss-1630178-550d9a76 | 0 | custom_metrics/coverage_rate_mean | 106 | 5703 | 619629 |
| qfree_focuss | qfree-1630177-fd3edb95 | qfree_focuss-1732233-bd21616e | 0 | custom_metrics/coverage_rate_mean | 143 | 5706 | 835089 |

## Results

| Scope | Metric | n | Pearson r | Circular p | BH q | Spearman rho | Spearman circular p | Detrended r | Detrended circular p | Detrended BH q |
|---|---|---|---|---|---|---|---|---|---|---|
| facmac_focuss | focuss_credit_loss | 304 | -0.2925 | 0.0647 | 0.08823 | -0.2399 | 0.0035 | -0.1168 | 0.1658 | 0.2495 |
| facmac_focuss | focuss_action_loss | 304 | -0.552 | 0.1181 | 0.1363 | -0.4898 | 0.0756 | -0.1544 | 0.7154 | 0.7154 |
| facmac_focuss | credit_alignment_kl | 304 | -0.8639 | 0.0034 | 0.006375 | -0.5142 | 0.0074 | -0.7866 | 0.0303 | 0.0909 |
| qfree_focuss | focuss_credit_loss | 143 | -0.5556 | 0.0001 | 0.0002143 | -0.5264 | 0.0057 | -0.1379 | 0.2596 | 0.3245 |
| qfree_focuss | focuss_action_loss | 143 | -0.6462 | 0.0001 | 0.0002143 | -0.5702 | 0.0001 | -0.2614 | 0.1375 | 0.2495 |
| qfree_focuss | credit_alignment_kl | 143 | -0.6505 | 0.0001 | 0.0002143 | -0.6259 | 0.0001 | -0.2399 | 0.1629 | 0.2495 |
| resq_focuss | focuss_credit_loss | 143 | 0.06875 | 0.5815 | 0.5815 | 0.06256 | 0.6674 | 0.1156 | 0.2464 | 0.3245 |
| resq_focuss | focuss_action_loss | 143 | -0.1346 | 0.2602 | 0.2788 | 0.1762 | 0.1166 | -0.21 | 0.0853 | 0.2133 |
| resq_focuss | credit_alignment_kl | 143 | -0.3073 | 0.0001 | 0.0002143 | 0.1647 | 0.1365 | -0.3844 | 0.0001 | 0.00075 |
| riskq_focuss | focuss_credit_loss | 106 | -0.3695 | 0.0001 | 0.0002143 | -0.3446 | 0.0285 | -0.3114 | 0.0001 | 0.00075 |
| riskq_focuss | focuss_action_loss | 106 | -0.2381 | 0.0269 | 0.04483 | -0.239 | 0.25 | -0.1282 | 0.1663 | 0.2495 |
| riskq_focuss | credit_alignment_kl | 106 | -0.2116 | 0.0994 | 0.1243 | -0.2889 | 0.1271 | -0.05954 | 0.4474 | 0.5162 |
| pooled_within_algorithm | focuss_credit_loss | 696 | -0.1888 | 0.0001 | 0.0002143 | -0.2015 | 0.0006 | -0.07676 | 0.0099 | 0.03713 |
| pooled_within_algorithm | focuss_action_loss | 696 | -0.4949 | 0.0612 | 0.08823 | -0.5375 | 0.0003 | -0.1529 | 0.5739 | 0.6149 |
| pooled_within_algorithm | credit_alignment_kl | 696 | -0.7156 | 0.0001 | 0.0002143 | -0.5762 | 0.0002 | -0.6077 | 0.0003 | 0.0015 |

IID p-values are retained in the CSV but should not be the primary inference because adjacent timesteps are autocorrelated.
