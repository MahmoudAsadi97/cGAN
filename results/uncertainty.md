| quantity                            |   estimate |   SE households |   SD seeds |   SE total | 95% CI       |   one-sided 95% lower bound |
|:------------------------------------|-----------:|----------------:|-----------:|-----------:|:-------------|----------------------------:|
| accuracy, final (mean of 5 seeds)   |      87.87 |            0.73 |       0.21 |       0.74 | 86.4 to 89.3 |                       86.66 |
| final - GitHub cGAN [last]          |      39.99 |            0.73 |     nan    |     nan    | 38.6 to 41.4 |                       38.78 |
| final - GitHub cGAN [best]          |      33.03 |            0.64 |     nan    |     nan    | 31.8 to 34.3 |                       31.98 |
| final - real: validation households |       1.81 |            0.64 |     nan    |     nan    | 0.6 to 3.1   |                        0.75 |
| final - real: training households   |      -0.08 |            0.39 |     nan    |     nan    | -0.8 to 0.7  |                       -0.72 |

Stratified half-sampling of the test households, 100 replicates; differences are paired (same household splits for all sets).  Accuracies in percent, differences in percentage points.  The final-model SE for differences uses the household term only (seed SD is shown separately).
