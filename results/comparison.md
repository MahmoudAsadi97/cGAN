| metric                   | better        | final             |   GitHub [last] |   GitHub [best] | final better             |
|:-------------------------|:--------------|:------------------|----------------:|----------------:|:-------------------------|
| Accuracy                 | higher        | 0.8787 ± 0.0021   |        0.4788   |        0.5483   | yes, complete separation |
| Mean profile             | higher        | 0.8673 ± 0.0021   |        0.1622   |        0.2536   | yes, complete separation |
| Std profile              | higher        | 0.7964 ± 0.0038   |        0.6327   |        0.6973   | yes, complete separation |
| KP mean                  | higher        | 0.9279 ± 0.002    |        0.4802   |        0.6044   | yes, complete separation |
| KP std                   | higher        | 0.9008 ± 0.0037   |        0.6746   |        0.6262   | yes, complete separation |
| Hourly 1-KS              | higher        | 0.9347 ± 0.00046  |        0.5203   |        0.597    | yes, complete separation |
| KP 1-KS                  | higher        | 0.9384 ± 0.00091  |        0.5652   |        0.6572   | yes, complete separation |
| Worst label              | higher        | 0.7464 ± 0.0057   |        0.2914   |        0.4329   | yes, complete separation |
| Labels >= 85%            | higher        | 0.825 ± 0.03      |        0        |        0        | yes, complete separation |
| FID (5 KP)               | lower         | 0.8534 ± 0.034    |       39.96     |       27.66     | yes, complete separation |
| KL near-peak             | lower         | 0.05611 ± 0.0017  |        1.131    |        1.084    | yes, complete separation |
| KL near-base             | lower         | 0.0797 ± 0.0021   |        2.707    |        1.009    | yes, complete separation |
| rFD                      | lower         | 0.117 ± 0.0048    |        1.21     |        0.9432   | yes, complete separation |
| Corr. MAE                | lower         | 0.07063 ± 0.00098 |        0.3442   |        0.3831   | yes, complete separation |
| C2ST AUC                 | closer to 0.5 | 0.6086 ± 0.0056   |        0.9998   |        0.9998   | yes, complete separation |
| Precision                | higher        | 0.8227 ± 0.0023   |        0.1533   |        0.3415   | yes, complete separation |
| Recall                   | higher        | 0.878 ± 0.0015    |        0.003033 |        0.001323 | yes, complete separation |
| Coverage                 | higher        | 0.8083 ± 0.0021   |        0.02454  |        0.05146  | yes, complete separation |
| DCR ratio                | closer to 1.0 | 1.007 ± 0.0031    |        2.01     |        1.497    | yes, complete separation |
| Neg. share (no PV)       | lower         | 0 ± 0             |        0.2544   |        0.1346   | yes, complete separation |
| paper.mean_diff_lt1      | higher        | 0.9925 ± 0.0036   |        0.4562   |        0.6969   | yes, complete separation |
| paper.std_diff_lt1       | higher        | 0.9994 ± 0.0014   |        0.7406   |        0.7188   | yes, complete separation |
| paper.kl_lt0.3           | higher        | 1 ± 0             |        0.0625   |        0.075    | yes, complete separation |
| paper.fid_lt5            | higher        | 1 ± 0             |        0.6344   |        0.7438   | yes, complete separation |
| paper.mean_kl            | lower         | 0.03721 ± 0.001   |        1.595    |        1.164    | yes, complete separation |
| paper.mean_fid           | lower         | 0.1433 ± 0.007    |        7.74     |        5.198    | yes, complete separation |
| paper_code.mean_diff_lt1 | higher        | 0.9925 ± 0.0028   |        0.6406   |        0.5312   | yes, complete separation |
| paper_code.std_diff_lt1  | higher        | 1 ± 0             |        0.8031   |        0.8031   | yes, complete separation |
| paper_code.fid_lt5       | higher        | 1 ± 0             |        0.7188   |        0.6375   | yes, complete separation |
| paper_code.mean_fid      | lower         | 0.136 ± 0.007     |        5.552    |        5.708    | yes, complete separation |
| paper_code.kl_lt0.3      | descriptive   | 0.9769 ± 0.0036   |        0.8844   |        0.9531   | descriptive only         |
| paper_code.mean_kl       | descriptive   | 0.1282 ± 0.005    |        0.1247   |        0.09365  | descriptive only         |
