# Perturbation Severity Ranking

## phenotype=ALL | metric=factoid_em

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=ALL | metric=factoid_f1

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | shuffle_snippets | -0.0031 |  | -0.003050582499735 | -0.003050582499735 | 1 |
| 2.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 2.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | irrelevant_noise_heavy | 0.0085 | 0.0148 | 0.0 | 0.0256410256410256 | 3 |
| 4.0 | lexical_noise_heavy | 0.0118 | 0.0162 | 0.0 | 0.0303030303030302 | 3 |
| 5.0 | lexical_noise_medium | 0.0238 |  | 0.0238095238095237 | 0.0238095238095237 | 1 |
| 6.0 | lexical_noise | 0.0256 |  | 0.0256410256410256 | 0.0256410256410256 | 1 |


## phenotype=ALL | metric=list_f1

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | irrelevant_noise | -0.0774 |  | -0.0773809523809523 | -0.0773809523809523 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0694 | 0.0599 | -0.125 | -0.0059523809523809 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | lexical_noise | 0.0083 |  | 0.0083333333333333 | 0.0083333333333333 | 1 |
| 4.0 | lexical_noise_medium | 0.0083 |  | 0.0083333333333333 | 0.0083333333333333 | 1 |
| 5.0 | lexical_noise_heavy | 0.0214 | 0.0302 | 0.0 | 0.0559523809523809 | 3 |


## phenotype=ALL | metric=list_precision

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | irrelevant_noise | -0.0694 |  | -0.0694444444444444 | -0.0694444444444444 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0556 | 0.0599 | -0.1111111111111111 | 0.0079365079365079 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | lexical_noise | 0.0139 |  | 0.0138888888888888 | 0.0138888888888888 | 1 |
| 4.0 | lexical_noise_medium | 0.0139 |  | 0.0138888888888888 | 0.0138888888888888 | 1 |
| 5.0 | lexical_noise_heavy | 0.0231 | 0.0289 | 0.0 | 0.0555555555555555 | 3 |


## phenotype=ALL | metric=list_recall

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | irrelevant_noise | -0.0873 |  | -0.0873015873015873 | -0.0873015873015873 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0873 | 0.0599 | -0.1428571428571428 | -0.0238095238095237 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | lexical_noise_heavy | 0.0185 | 0.0321 | 0.0 | 0.0555555555555555 | 3 |


## phenotype=ALL | metric=macro_avg

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | shuffle_snippets | -0.0096 |  | -0.0095987290154585 | -0.0095987290154585 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0079 | 0.0239 | -0.0333302849484738 | 0.0140471186117253 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 4.0 | irrelevant_noise | 0.0027 |  | 0.0026585072231139 | 0.0026585072231139 | 1 |
| 5.0 | lexical_noise_heavy | 0.0077 | 0.0203 | -0.0105698353155733 | 0.0295795028941096 | 3 |
| 6.0 | lexical_noise_medium | 0.0080 |  | 0.0080357142857142 | 0.0080357142857142 | 1 |
| 7.0 | lexical_noise | 0.0085 |  | 0.0084935897435897 | 0.0084935897435897 | 1 |


## phenotype=ALL | metric=summary_rougeL

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | shuffle_snippets | -0.0353 |  | -0.0353443335620992 | -0.0353443335620992 | 1 |
| 2.0 | lexical_noise_heavy | -0.0023 | 0.0786 | -0.0556631796461319 | 0.0880149812734082 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | irrelevant_noise_heavy | 0.0294 | 0.0347 | -0.0083211397938955 | 0.0600688533272803 | 3 |
| 5.0 | irrelevant_noise | 0.0880 |  | 0.0880149812734082 | 0.0880149812734082 | 1 |


## phenotype=ALL | metric=yesno_acc

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=long_context | metric=factoid_em

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=long_context | metric=factoid_f1

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | shuffle_snippets | -0.0037 |  | -0.003660698999682 | -0.003660698999682 | 1 |
| 2.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 2.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 2.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 2.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 2.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 2.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=long_context | metric=list_f1

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 2.0 | lexical_noise_heavy | 0.0238 | 0.0412 | 0.0 | 0.0714285714285714 | 3 |
| 3.0 | irrelevant_noise_heavy | 0.0556 | 0.0599 | 0.0 | 0.119047619047619 | 3 |
| 4.0 | irrelevant_noise | 0.0571 |  | 0.0571428571428571 | 0.0571428571428571 | 1 |


## phenotype=long_context | metric=list_precision

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 2.0 | lexical_noise_heavy | 0.0208 | 0.0361 | 0.0 | 0.0625 | 3 |
| 3.0 | irrelevant_noise | 0.0500 |  | 0.05 | 0.05 | 1 |
| 4.0 | irrelevant_noise_heavy | 0.0556 | 0.0599 | 0.0 | 0.119047619047619 | 3 |


## phenotype=long_context | metric=list_recall

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 2.0 | lexical_noise_heavy | 0.0278 | 0.0481 | 0.0 | 0.0833333333333333 | 3 |
| 3.0 | irrelevant_noise_heavy | 0.0556 | 0.0599 | 0.0 | 0.119047619047619 | 3 |
| 4.0 | irrelevant_noise | 0.0667 |  | 0.0666666666666666 | 0.0666666666666666 | 1 |


## phenotype=long_context | metric=macro_avg

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | shuffle_snippets | -0.0142 |  | -0.0141692998357077 | -0.0141692998357077 | 1 |
| 2.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 2.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 2.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | lexical_noise_heavy | 0.0060 | 0.0103 | 0.0 | 0.0178571428571428 | 3 |
| 4.0 | irrelevant_noise_heavy | 0.0080 | 0.0091 | -0.0004181075872252 | 0.0176434604743428 | 3 |
| 5.0 | irrelevant_noise | 0.0143 |  | 0.0142857142857142 | 0.0142857142857142 | 1 |


## phenotype=long_context | metric=summary_rougeL

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | shuffle_snippets | -0.0530 |  | -0.0530165003431489 | -0.0530165003431489 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0234 | 0.0236 | -0.0484737771502477 | -0.0016724303489009 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=long_context | metric=yesno_acc

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=multi_answer_list | metric=factoid_em

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=multi_answer_list | metric=factoid_f1

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=multi_answer_list | metric=list_f1

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | irrelevant_noise | -0.0774 |  | -0.0773809523809523 | -0.0773809523809523 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0694 | 0.0599 | -0.125 | -0.0059523809523809 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | lexical_noise | 0.0083 |  | 0.0083333333333333 | 0.0083333333333333 | 1 |
| 4.0 | lexical_noise_medium | 0.0083 |  | 0.0083333333333333 | 0.0083333333333333 | 1 |
| 5.0 | lexical_noise_heavy | 0.0214 | 0.0302 | 0.0 | 0.0559523809523809 | 3 |


## phenotype=multi_answer_list | metric=list_precision

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | irrelevant_noise | -0.0694 |  | -0.0694444444444444 | -0.0694444444444444 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0556 | 0.0599 | -0.1111111111111111 | 0.0079365079365079 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | lexical_noise | 0.0139 |  | 0.0138888888888888 | 0.0138888888888888 | 1 |
| 4.0 | lexical_noise_medium | 0.0139 |  | 0.0138888888888888 | 0.0138888888888888 | 1 |
| 5.0 | lexical_noise_heavy | 0.0231 | 0.0289 | 0.0 | 0.0555555555555555 | 3 |


## phenotype=multi_answer_list | metric=list_recall

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | irrelevant_noise | -0.0873 |  | -0.0873015873015873 | -0.0873015873015873 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0873 | 0.0599 | -0.1428571428571428 | -0.0238095238095237 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 3.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | lexical_noise_heavy | 0.0185 | 0.0321 | 0.0 | 0.0555555555555555 | 3 |


## phenotype=multi_answer_list | metric=macro_avg

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | irrelevant_noise | -0.0774 |  | -0.0773809523809523 | -0.0773809523809523 | 1 |
| 2.0 | irrelevant_noise_heavy | -0.0694 | 0.0599 | -0.125 | -0.0059523809523809 | 3 |
| 3.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 3.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |
| 4.0 | lexical_noise | 0.0083 |  | 0.0083333333333333 | 0.0083333333333333 | 1 |
| 4.0 | lexical_noise_medium | 0.0083 |  | 0.0083333333333333 | 0.0083333333333333 | 1 |
| 5.0 | lexical_noise_heavy | 0.0214 | 0.0302 | 0.0 | 0.0559523809523809 | 3 |


## phenotype=multi_answer_list | metric=summary_rougeL

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |


## phenotype=multi_answer_list | metric=yesno_acc

| rank_worst | perturbation | delta_mean | delta_std | delta_min | delta_max | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | contradiction | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | irrelevant_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | irrelevant_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | lexical_noise_heavy | 0.0000 | 0.0000 | 0.0 | 0.0 | 3 |
| 1.0 | lexical_noise_medium | 0.0000 |  | 0.0 | 0.0 | 1 |
| 1.0 | shuffle_snippets | 0.0000 |  | 0.0 | 0.0 | 1 |

