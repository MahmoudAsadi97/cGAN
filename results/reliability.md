| model                       |   seeds | c2st_strong            | accuracy               | kp_ks                  | coverage               | dcr_ratio              |
|:----------------------------|--------:|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| final [best]                |       5 | 0.697 (0.685 to 0.709) | 0.879 (0.864 to 0.894) | 0.061 (0.056 to 0.067) | 0.835 (0.811 to 0.859) | 1.007 (0.972 to 1.042) |
| faithful [last]             |       3 | 1.000 (1.000 to 1.000) | 0.484 (0.463 to 0.505) | 0.404 (0.374 to 0.434) | 0.053 (0.033 to 0.073) | 1.958 (1.888 to 2.028) |
| real: validation households |       1 | 0.505 (0.482 to 0.528) | 0.861 (0.840 to 0.881) | 0.068 (0.061 to 0.075) | 0.822 (0.797 to 0.847) | 1.024 (0.991 to 1.057) |
| real: training households   |       1 | 0.483 (0.471 to 0.496) | 0.879 (0.863 to 0.896) | 0.053 (0.049 to 0.058) | 0.851 (0.827 to 0.875) | 0.265 (0.252 to 0.278) |

Test households, label-averaged; in brackets the 95 % interval over households (stratified half-sampling, 10 replicates) including the seed spread.  Ideal: c2st_strong 0.5, accuracy higher, kp_ks lower, coverage higher, dcr_ratio about 1.  CPU sampling.

Paired differences to the real validation-household benchmark (same household splits):

| comparison                                    | c2st_strong               | accuracy                  | kp_ks                     | coverage                  | dcr_ratio                 |
|:----------------------------------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| final [best] - real: validation households    | +0.192 (+0.172 to +0.211) | +0.019 (+0.003 to +0.034) | -0.007 (-0.016 to +0.002) | +0.013 (+0.007 to +0.019) | -0.018 (-0.024 to -0.011) |
| faithful [last] - real: validation households | +0.495 (+0.472 to +0.518) | -0.377 (-0.390 to -0.364) | +0.336 (+0.331 to +0.341) | -0.769 (-0.790 to -0.748) | +0.934 (+0.900 to +0.968) |
