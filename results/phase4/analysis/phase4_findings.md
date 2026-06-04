# Phase 4 Findings (Auto-Generated)

- Runs: **21**
- Clean runs: **4**
- Missing baselines: **0**

## Perturbations Ranked by Macro Δ vs Clean (lower = worse)

| Rank | Perturbation | Mean Macro Δ |
|---:|---|---:|
| 1 | `shuffle_snippets` | -0.0385 |
| 2 | `lexical_noise_medium` | -0.0295 |
| 3 | `irrelevant_noise_heavy` | -0.0204 |
| 4 | `lexical_noise` | -0.0081 |
| 5 | `lexical_noise_heavy` | -0.0048 |
| 6 | `irrelevant_noise` | -0.0028 |
| 7 | `contradiction` | 0.0000 |

## Mean Δ vs Clean by Metric (per perturbation)

| Perturbation | macro_avg | yesno_acc | factoid_f1 | list_f1 | summary_rougeL | n |
|---|---:|---:|---:|---:|---:|---:|
| `contradiction` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4 |
| `irrelevant_noise` | -0.0028 | 0.0000 | -0.0303 | 0.0087 | 0.0103 | 2 |
| `irrelevant_noise_heavy` | -0.0204 | 0.0000 | 0.0028 | -0.0897 | 0.0053 | 3 |
| `lexical_noise` | -0.0081 | 0.0000 | 0.0000 | -0.0339 | 0.0013 | 2 |
| `lexical_noise_heavy` | -0.0048 | 0.0000 | 0.0061 | 0.0012 | -0.0263 | 3 |
| `lexical_noise_medium` | -0.0295 | 0.0000 | 0.0067 | -0.0524 | -0.0722 | 1 |
| `shuffle_snippets` | -0.0385 | 0.0000 | -0.0123 | -0.0125 | -0.1290 | 2 |
