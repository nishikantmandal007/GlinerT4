| layout | passes | chunks/s | speedup | Person R/P | Organization R/P | Street R/P | City R/P | identical chunks | accepted |
|---|---|---|---|---|---|---|---|---|---|
| current_3pass | 3 | 53.33 | 1.000x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.930 | 290/290 | yes |
| drop_city_pass | 2 | 77.71 | 1.457x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.947 | 286/290 | yes |
| merged_city_desc | 2 | 76.73 | 1.439x | 0.994/0.961 | 0.977/0.846 | 0.861/1.000 | 1.000/0.936 | 269/290 | yes |
| city_with_org | 2 | 75.79 | 1.421x | 0.994/0.961 | 0.977/0.857 | 0.861/1.000 | 1.000/0.884 | 277/290 | no: City precision |
| city_with_org_desc | 2 | 75.29 | 1.412x | 0.994/0.961 | 0.977/0.857 | 0.861/1.000 | 1.000/0.889 | 259/290 | no: City precision |

