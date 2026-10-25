| config | count0 | fmtcache | ln | streams | depth | workers | batch | chunks/s | vs baseline | identical full output | identical kept values |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| baseline | False | False | False | 1 | 0 | 0 | 16 | 50.91 | 1.000x | True | 290/290 |
| count0 | True | False | False | 1 | 0 | 0 | 16 | 53.40 | 1.049x | True | 290/290 |
| count0_streams2 | True | False | False | 2 | 0 | 0 | 16 | 56.20 | 1.104x | True | 290/290 |
| ln | False | False | True | 1 | 0 | 0 | 16 | 51.86 | 1.019x | True | 290/290 |
| count0_streams2_ln | True | False | True | 2 | 0 | 0 | 16 | 58.05 | 1.140x | True | 290/290 |

