| metric                   | better        | final             |   GitHub [last] |   GitHub [best] | final better             |
|:-------------------------|:--------------|:------------------|----------------:|----------------:|:-------------------------|
| Accuracy                 | higher        | 0.8787 ± 0.0021   |        0.4835   |        0.5514   | yes, complete separation |
| Mean profile             | higher        | 0.8673 ± 0.0021   |        0.1699   |        0.2959   | yes, complete separation |
| Std profile              | higher        | 0.7964 ± 0.0038   |        0.5974   |        0.6653   | yes, complete separation |
| KP mean                  | higher        | 0.9279 ± 0.002    |        0.5121   |        0.5777   | yes, complete separation |
| KP std                   | higher        | 0.9008 ± 0.0037   |        0.6858   |        0.6606   | yes, complete separation |
| Hourly 1-KS              | higher        | 0.9347 ± 0.00046  |        0.5389   |        0.6106   | yes, complete separation |
| KP 1-KS                  | higher        | 0.9384 ± 0.00091  |        0.5963   |        0.6398   | yes, complete separation |
| Worst label              | higher        | 0.7464 ± 0.0057   |        0.2744   |        0.3977   | yes, complete separation |
| Labels >= 85%            | higher        | 0.825 ± 0.03      |        0        |        0        | yes, complete separation |
| FID (5 KP)               | lower         | 0.8534 ± 0.034    |       36.46     |       37.71     | yes, complete separation |
| KL near-peak             | lower         | 0.05611 ± 0.0017  |        1.045    |        0.8416   | yes, complete separation |
| KL near-base             | lower         | 0.0797 ± 0.0021   |        2.274    |        2.007    | yes, complete separation |
| rFD                      | lower         | 0.117 ± 0.0048    |        1.397    |        0.9126   | yes, complete separation |
| Corr. MAE                | lower         | 0.07063 ± 0.00098 |        0.3829   |        0.364    | yes, complete separation |
| C2ST AUC                 | closer to 0.5 | 0.6086 ± 0.0056   |        0.9998   |        0.9994   | yes, complete separation |
| Precision                | higher        | 0.8227 ± 0.0023   |        0.2196   |        0.3357   | yes, complete separation |
| Recall                   | higher        | 0.878 ± 0.0015    |        0.005065 |        0.004238 | yes, complete separation |
| Coverage                 | higher        | 0.8083 ± 0.0021   |        0.03878  |        0.05629  | yes, complete separation |
| DCR ratio                | closer to 1.0 | 1.007 ± 0.0031    |        2.004    |        1.512    | yes, complete separation |
| Neg. share (no PV)       | lower         | 0 ± 0             |        0.2194   |        0.1815   | yes, complete separation |
| paper.mean_diff_lt1      | higher        | 0.9925 ± 0.0036   |        0.5396   |        0.6521   | yes, complete separation |
| paper.std_diff_lt1       | higher        | 0.9994 ± 0.0014   |        0.7812   |        0.7375   | yes, complete separation |
| paper.kl_lt0.3           | higher        | 1 ± 0             |        0.07292  |        0.1062   | yes, complete separation |
| paper.fid_lt5            | higher        | 1 ± 0             |        0.7135   |        0.7344   | yes, complete separation |
| paper.mean_kl            | lower         | 0.03721 ± 0.001   |        1.41     |        1.299    | yes, complete separation |
| paper.mean_fid           | lower         | 0.1433 ± 0.007    |        7.012    |        7.218    | yes, complete separation |
| paper_code.mean_diff_lt1 | higher        | 0.9925 ± 0.0028   |        0.6104   |        0.5573   | yes, complete separation |
| paper_code.std_diff_lt1  | higher        | 1 ± 0             |        0.7792   |        0.8219   | yes, complete separation |
| paper_code.fid_lt5       | higher        | 1 ± 0             |        0.7115   |        0.699    | yes, complete separation |
| paper_code.mean_fid      | lower         | 0.136 ± 0.007     |        6.021    |        6.639    | yes, complete separation |
| paper_code.kl_lt0.3      | descriptive   | 0.9769 ± 0.0036   |        0.9083   |        0.9167   | descriptive only         |
| paper_code.mean_kl       | descriptive   | 0.1282 ± 0.005    |        0.1102   |        0.102    | descriptive only         |
