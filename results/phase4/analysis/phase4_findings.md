# Phase 4 Findings (Auto-Generated)

- Runs: **16**
- Clean runs: **3**
- Missing baselines: **0**

## Perturbations Ranked by Macro Δ vs Clean (lower = worse)

| Rank | Perturbation | Mean Macro Δ |
|---:|---|---:|
| 1 | `shuffle_snippets` | -0.0096 |
| 2 | `irrelevant_noise_heavy` | -0.0079 |
| 3 | `contradiction` | 0.0000 |
| 4 | `irrelevant_noise` | 0.0027 |
| 5 | `lexical_noise_heavy` | 0.0077 |
| 6 | `lexical_noise_medium` | 0.0080 |
| 7 | `lexical_noise` | 0.0085 |

## Mean Δ vs Clean by Metric (per perturbation)

| Perturbation | macro_avg | yesno_acc | factoid_f1 | list_f1 | summary_rougeL | n |
|---|---:|---:|---:|---:|---:|---:|
| `contradiction` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |
| `irrelevant_noise` | 0.0027 | 0.0000 | 0.0000 | -0.0774 | 0.0880 | 1 |
| `irrelevant_noise_heavy` | -0.0079 | 0.0000 | 0.0085 | -0.0694 | 0.0294 | 3 |
| `lexical_noise` | 0.0085 | 0.0000 | 0.0256 | 0.0083 | 0.0000 | 1 |
| `lexical_noise_heavy` | 0.0077 | 0.0000 | 0.0118 | 0.0214 | -0.0023 | 3 |
| `lexical_noise_medium` | 0.0080 | 0.0000 | 0.0238 | 0.0083 | 0.0000 | 1 |
| `shuffle_snippets` | -0.0096 | 0.0000 | -0.0031 | 0.0000 | -0.0353 | 1 |
