# Coverage-rate gain correlation summary

Pooled within-algorithm results after centering each variable within its algorithm (`n = 696`). Circular-shift tests use 9,999 permutations to preserve temporal autocorrelation. BH q-values control the false-discovery rate across the reported algorithm/metric tests.

| Metric | n | Pearson r | Circular p | BH q | Spearman rho | Circular p | BH q | Detrended r | Detrended circular p | Detrended BH q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `focuss_credit_loss` | 696 | -0.1888 | 0.0001 | 0.000214 | -0.2015 | 0.0006 | 0.001800 | -0.0768 | 0.0099 | 0.037125 |
| `focuss_action_loss` | 696 | -0.4949 | 0.0612 | 0.088227 | -0.5375 | 0.0003 | 0.001125 | -0.1529 | 0.5739 | 0.614893 |
| `credit_alignment_kl` | 696 | **-0.7156** | **0.0001** | **0.000214** | **-0.5762** | **0.0002** | **0.001000** | **-0.6077** | **0.0003** | **0.001500** |

The primary inference should use circular-shift p-values rather than IID p-values because adjacent training timesteps are autocorrelated. Credit alignment KL shows the strongest and most robust negative association with coverage-rate gain after detrending.
