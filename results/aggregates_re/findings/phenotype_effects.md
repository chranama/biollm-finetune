# Phenotype Effects Summary

- Rows (input): 39
- Rows (used): 26
- Buckets: 14

## Worst perturbations by phenotype

### long_context
- dataset: `bioasq_TINY` | runtime: `mps_fp32` | model: `tinyllama-1.1b-chat`

**delta__macro_avg**
- shuffle_snippets: mean=-0.0142 std=0.0000 n=1 seeds=[42]
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_medium: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0060 std=0.0103 n=3 seeds=[13, 42, 97]

**delta__yesno_acc**
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]

**delta__factoid_f1**
- shuffle_snippets: mean=-0.0037 std=0.0000 n=1 seeds=[42]
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]

**delta__factoid_em**
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]

**delta__list_f1**
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_medium: mean=0.0000 std=0.0000 n=1 seeds=[42]
- shuffle_snippets: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0238 std=0.0412 n=3 seeds=[13, 42, 97]

**delta__summary_rougeL**
- shuffle_snippets: mean=-0.0530 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=-0.0234 std=0.0236 n=3 seeds=[13, 42, 97]
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]

### multi_answer_list
- dataset: `bioasq_TINY` | runtime: `mps_fp32` | model: `tinyllama-1.1b-chat`

**delta__macro_avg**
- irrelevant_noise: mean=-0.0774 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=-0.0694 std=0.0599 n=3 seeds=[13, 42, 97]
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- shuffle_snippets: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise: mean=0.0083 std=0.0000 n=1 seeds=[42]

**delta__yesno_acc**
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]

**delta__factoid_f1**
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]

**delta__factoid_em**
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]

**delta__list_f1**
- irrelevant_noise: mean=-0.0774 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=-0.0694 std=0.0599 n=3 seeds=[13, 42, 97]
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- shuffle_snippets: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise: mean=0.0083 std=0.0000 n=1 seeds=[42]

**delta__summary_rougeL**
- contradiction: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- irrelevant_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- irrelevant_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
- lexical_noise: mean=0.0000 std=0.0000 n=1 seeds=[42]
- lexical_noise_heavy: mean=0.0000 std=0.0000 n=3 seeds=[13, 42, 97]
