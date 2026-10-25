| config | count0 | fmtcache | streams | depth | workers | batch | chunks/s | vs baseline | identical full output | identical kept values |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| baseline | False | False | 1 | 0 | 0 | 16 | 47.79 | 1.000x | True | 290/290 |
| count0 | True | False | 1 | 0 | 0 | 16 | 49.64 | 1.039x | True | 290/290 |
| fmtcache | False | True | 1 | 0 | 0 | 16 | 47.31 | 0.990x | True | 290/290 |
| exact | True | True | 1 | 0 | 0 | 16 | 49.52 | 1.036x | True | 290/290 |
| exact_prefetch1 | True | True | 1 | 1 | 0 | 16 | 49.62 | 1.038x | True | 290/290 |
| streams2 | False | False | 2 | 0 | 0 | 16 | 49.74 | 1.041x | True | 290/290 |
| exact_streams2 | True | True | 2 | 0 | 0 | 16 | 50.32 | 1.053x | True | 290/290 |

