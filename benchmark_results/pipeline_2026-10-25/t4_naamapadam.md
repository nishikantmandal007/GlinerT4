### fastino/gliner2-multi-v1

| dataset | sentences | stock_fp16 F1 | glinert4 F1 | stock_fp16 sent/s | glinert4 sent/s | glinert4 speedup vs fp16 / fp32 | identical to fp16 |
|---|---|---|---|---|---|---|---|
| naamapadam_as | 51 | 0.3881 | 0.3881 | 28.6 | 391.2 | 13.69x | 100.0% |
| naamapadam_bn | 607 | 0.6378 | 0.6378 | 114.3 | 205.7 | 1.80x | 99.8% |
| naamapadam_gu | 999 | 0.7266 | 0.7266 | 94.2 | 463.7 | 4.92x | 99.6% |
| naamapadam_hi | 864 | 0.7565 | 0.7570 | 97.9 | 489.0 | 4.99x | 99.5% |
| naamapadam_kn | 985 | 0.7425 | 0.7422 | 137.5 | 640.3 | 4.66x | 99.7% |
| naamapadam_ml | 965 | 0.7619 | 0.7623 | 169.0 | 649.3 | 3.84x | 99.7% |
| naamapadam_mr | 980 | 0.7002 | 0.7002 | 286.1 | 633.4 | 2.21x | 99.5% |
| naamapadam_or | 994 | 0.1172 | 0.1173 | 40.5 | 343.8 | 8.48x | 99.3% |
| naamapadam_pa | 993 | 0.6019 | 0.6021 | 106.7 | 484.3 | 4.54x | 99.6% |
| naamapadam_ta | 758 | 0.6774 | 0.6774 | 237.1 | 631.5 | 2.66x | 99.7% |
| naamapadam_te | 847 | 0.7443 | 0.7446 | 208.8 | 616.8 | 2.95x | 99.6% |

Naamapadam average micro F1 over 11 languages (PR #186 reports 70.5 for gliner2 over 9 languages): stock_fp16 62.3, glinert4 62.3

Person (glinert4): precision / recall / F1, gold names

| dataset | P | R | F1 | gold |
|---|---|---|---|---|
| naamapadam_as | 0.250 | 0.500 | 0.333 | 6 |
| naamapadam_bn | 0.745 | 0.732 | 0.739 | 459 |
| naamapadam_gu | 0.776 | 0.859 | 0.815 | 630 |
| naamapadam_hi | 0.836 | 0.886 | 0.861 | 790 |
| naamapadam_kn | 0.823 | 0.851 | 0.836 | 610 |
| naamapadam_ml | 0.822 | 0.891 | 0.855 | 714 |
| naamapadam_mr | 0.804 | 0.823 | 0.813 | 659 |
| naamapadam_or | 0.072 | 0.144 | 0.096 | 431 |
| naamapadam_pa | 0.635 | 0.736 | 0.682 | 561 |
| naamapadam_ta | 0.773 | 0.829 | 0.800 | 484 |
| naamapadam_te | 0.795 | 0.867 | 0.830 | 609 |


