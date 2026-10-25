### corpus_040sections, chunks (10 documents, 290 texts in, 37985 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 56.8 | 7441 | 0.994/0.961 · 0.997/1.000 · 0.980/0.979 | 0.977/0.846 · 0.993/1.000 · 0.955/0.941 | 0.861/1.000 · 0.833/1.000 · 0.085/0.101 | 1.000/0.930 · 1.000/0.973 · 0.995/0.964 | reference |
| single_street_0.5 | 136.4 | 17862 | 0.996/0.964 · 0.997/1.000 · 0.981/0.981 | 0.977/0.863 · 0.993/1.000 · 0.955/0.948 | 0.936/1.000 · 0.885/1.000 · 0.087/0.093 | 1.000/0.947 · 1.000/0.985 · 0.995/0.975 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 5.10 | 29 / 0 / 0 / 4 | 24 / 0 / 0 / 3 | 304 / 0 / 0 / 67 | 4 / 11 / 0 / 0 |
| single_street_0.5 | 2.13 | 26 / 0 / 0 / 4 | 21 / 0 / 0 / 3 | 342 / 0 / 0 / 46 | 4 / 6 / 0 / 0 |

### corpus_040sections, doc384-64 (10 documents, 10 texts in, 37985 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 1.5 | 5853 | 1.000/0.999 · 0.999/1.000 · 0.999/0.999 | 1.000/0.977 · 1.000/1.000 · 0.993/0.993 | 0.871/1.000 · 0.875/1.000 · 0.077/0.087 | 1.000/0.988 · 1.000/0.995 · 1.000/0.995 | reference |
| single_street_0.5 | 3.8 | 14595 | 1.000/0.997 · 1.000/1.000 · 0.999/0.999 | 1.000/0.992 · 1.000/1.000 · 0.998/0.998 | 0.871/1.000 · 0.880/1.000 · 0.077/0.083 | 1.000/0.994 · 1.000/0.998 · 1.000/0.998 | Person doc_set precision |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 6.49 | 1 / 0 / 0 / 1 | 3 / 0 / 0 / 0 | 324 / 0 / 0 / 50 | 0 / 2 / 0 / 0 |
| single_street_0.5 | 2.60 | 2 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 341 / 0 / 0 / 48 | 0 / 1 / 0 / 0 |

### corpus_040sections, doc384-128 (10 documents, 10 texts in, 37985 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 1.1 | 4348 | 1.000/1.000 · 0.999/1.000 · 0.999/1.000 | 1.000/0.963 · 1.000/0.990 · 0.995/0.985 | 0.882/1.000 · 0.880/0.997 · 0.080/0.087 | 1.000/0.988 · 1.000/0.995 · 1.000/0.995 | reference |
| single_street_0.5 | 2.9 | 11089 | 0.999/0.999 · 0.999/1.000 · 0.999/0.999 | 1.000/0.985 · 1.000/1.000 · 0.995/0.995 | 0.882/1.000 · 0.882/0.997 · 0.080/0.082 | 1.000/0.994 · 1.000/0.998 · 1.000/0.998 | Person doc_set recall, Person doc_set precision |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 8.74 | 0 / 0 / 0 / 1 | 2 / 4 / 0 / 0 | 336 / 0 / 1 / 48 | 0 / 2 / 0 / 0 |
| single_street_0.5 | 3.43 | 1 / 0 / 0 / 1 | 2 / 0 / 0 / 0 | 358 / 0 / 1 / 47 | 0 / 1 / 0 / 0 |

### corpus_040sections, doc256-64 (10 documents, 10 texts in, 37985 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 1.3 | 5069 | 0.999/0.997 · 0.999/1.000 · 0.998/0.999 | 1.000/0.949 · 1.000/0.985 · 0.995/0.980 | 0.882/1.000 · 0.877/0.997 · 0.080/0.090 | 1.000/0.964 · 1.000/0.973 · 1.000/0.973 | reference |
| single_street_0.5 | 3.6 | 13514 | 0.999/0.996 · 1.000/1.000 · 0.998/0.998 | 1.000/0.985 · 1.000/1.000 · 0.995/0.995 | 0.893/1.000 · 0.885/1.000 · 0.080/0.087 | 1.000/0.994 · 1.000/0.995 · 1.000/0.995 | Person doc_set precision |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 7.49 | 2 / 0 / 0 / 1 | 2 / 6 / 0 / 0 | 324 / 0 / 1 / 49 | 0 / 10 / 1 / 0 |
| single_street_0.5 | 2.81 | 3 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 336 / 0 / 0 / 46 | 0 / 1 / 1 / 0 |

### corpus_040sections, doc192-48 (10 documents, 10 texts in, 37985 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 1.4 | 5301 | 0.999/0.997 · 0.999/1.000 · 0.998/0.999 | 1.000/0.970 · 1.000/0.993 · 0.995/0.988 | 0.895/1.000 · 0.880/1.000 · 0.087/0.098 | 1.000/0.988 · 1.000/0.993 · 1.000/0.993 | reference |
| single_street_0.5 | 3.2 | 12269 | 0.999/0.997 · 0.999/1.000 · 0.998/0.999 | 1.000/0.977 · 1.000/1.000 · 0.993/0.993 | 0.909/1.000 · 0.897/1.000 · 0.090/0.096 | 1.000/0.994 · 1.000/0.995 · 1.000/0.995 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 7.17 | 2 / 0 / 0 / 1 | 2 / 3 / 0 / 0 | 323 / 0 / 0 / 48 | 0 / 2 / 1 / 0 |
| single_street_0.5 | 3.10 | 2 / 0 / 0 / 1 | 3 / 0 / 0 / 0 | 340 / 0 / 0 / 41 | 0 / 1 / 1 / 0 |

### corpus_020sections, chunks (10 documents, 200 texts in, 18954 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 67.9 | 6433 | 0.989/0.986 · 1.000/1.000 · 0.994/0.993 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.820/1.000 · 0.820/1.000 · 0.115/0.140 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 165.4 | 15678 | 0.992/0.992 · 1.000/1.000 · 0.996/0.996 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.965/1.000 · 0.960/1.000 · 0.115/0.116 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 2.95 | 5 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 141 / 0 / 0 / 36 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 1.21 | 3 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 175 / 0 / 0 / 8 | 0 / 0 / 0 / 0 |

### corpus_020sections, doc384-64 (10 documents, 10 texts in, 18954 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 2.8 | 5333 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.980 · 1.000/1.000 · 0.990/0.990 | 0.880/0.993 · 0.875/0.994 · 0.075/0.084 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 7.3 | 13802 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.990 · 1.000/1.000 · 0.995/0.995 | 0.880/0.994 · 0.880/0.995 · 0.070/0.074 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 3.55 | 0 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 163 / 0 / 1 / 25 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 1.37 | 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 173 / 0 / 1 / 24 | 0 / 0 / 0 / 0 |

### corpus_020sections, doc384-128 (10 documents, 10 texts in, 18954 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 2.4 | 4611 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.890/1.000 · 0.880/1.000 · 0.080/0.089 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 6.3 | 11888 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.890/1.000 · 0.885/1.000 · 0.080/0.087 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 4.11 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 163 / 0 / 0 / 24 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 1.59 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 168 / 0 / 0 / 23 | 0 / 0 / 0 / 0 |

### corpus_020sections, doc256-64 (10 documents, 10 texts in, 18954 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 2.7 | 5147 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.961 · 1.000/0.980 · 0.995/0.975 | 0.890/1.000 · 0.885/1.000 · 0.080/0.088 | 1.000/0.982 · 1.000/0.976 · 1.000/0.976 | reference |
| single_street_0.5 | 7.1 | 13503 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.980 · 1.000/1.000 · 0.990/0.990 | 0.900/1.000 · 0.890/1.000 · 0.080/0.087 | 1.000/0.994 · 1.000/0.995 · 1.000/0.995 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 3.68 | 0 / 0 / 0 / 0 | 1 / 4 / 0 / 0 | 165 / 0 / 0 / 23 | 0 / 5 / 0 / 0 |
| single_street_0.5 | 1.40 | 0 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 168 / 0 / 0 / 22 | 0 / 1 / 0 / 0 |

### corpus_020sections, doc192-48 (10 documents, 10 texts in, 18954 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 2.8 | 5267 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.970 · 1.000/0.985 · 0.995/0.980 | 0.885/1.000 · 0.885/1.000 · 0.100/0.110 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 6.7 | 12770 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.990 · 1.000/1.000 · 0.995/0.995 | 0.910/1.000 · 0.910/1.000 · 0.105/0.108 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 3.60 | 0 / 0 / 0 / 0 | 1 / 3 / 0 / 0 | 161 / 0 / 0 / 23 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 1.48 | 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 174 / 0 / 0 / 18 | 0 / 0 / 0 / 0 |

### corpus_010sections, chunks (10 documents, 100 texts in, 9514 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 67.7 | 6437 | 0.978/0.972 · 1.000/1.000 · 0.988/0.985 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 168.1 | 15989 | 0.983/0.983 · 1.000/1.000 · 0.991/0.991 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.940/1.000 · 0.930/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 1.48 | 5 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 80 / 0 / 0 / 20 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.60 | 3 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 99 / 0 / 0 / 7 | 0 / 0 / 0 / 0 |

### corpus_010sections, doc384-64 (10 documents, 10 texts in, 9514 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 5.9 | 5646 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.947 · 1.000/1.000 · 0.980/0.980 | 0.800/1.000 · 0.790/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 15.9 | 15159 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.973 · 1.000/1.000 · 0.990/0.990 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 1.69 | 0 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 81 / 0 / 0 / 21 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.63 | 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 89 / 0 / 0 / 20 | 0 / 0 / 0 / 0 |

### corpus_010sections, doc384-128 (10 documents, 10 texts in, 9514 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 4.8 | 4614 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.790/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 12.7 | 12071 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 2.06 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 81 / 0 / 0 / 21 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.79 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 85 / 0 / 0 / 20 | 0 / 0 / 0 / 0 |

### corpus_010sections, doc256-64 (10 documents, 10 texts in, 9514 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 5.3 | 5037 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.900 · 1.000/0.962 · 0.990/0.952 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/0.964 · 1.000/0.952 · 1.000/0.952 | reference |
| single_street_0.5 | 13.9 | 13272 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.947 · 1.000/1.000 · 0.980/0.980 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/0.988 · 1.000/0.990 · 1.000/0.990 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 1.89 | 0 / 0 / 0 / 0 | 1 / 4 / 0 / 0 | 83 / 0 / 0 / 20 | 0 / 5 / 0 / 0 |
| single_street_0.5 | 0.72 | 0 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 85 / 0 / 0 / 20 | 0 / 1 / 0 / 0 |

### corpus_010sections, doc192-48 (10 documents, 10 texts in, 9514 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 5.6 | 5314 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.947 · 1.000/0.971 · 1.000/0.971 | 0.820/1.000 · 0.820/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 13.8 | 13097 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.840/1.000 · 0.840/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 1.79 | 0 / 0 / 0 / 0 | 0 / 3 / 0 / 0 | 83 / 0 / 0 / 18 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.73 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 93 / 0 / 0 / 16 | 0 / 0 / 0 / 0 |

### corpus_005sections, chunks (10 documents, 50 texts in, 4746 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 66.1 | 6272 | 0.989/0.978 · 1.000/1.000 · 0.994/0.988 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 167.1 | 15858 | 0.989/0.989 · 1.000/1.000 · 0.994/0.994 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.920/1.000 · 0.920/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 0.76 | 2 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 40 / 0 / 0 / 10 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.30 | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 47 / 0 / 0 / 4 | 0 / 0 / 0 / 0 |

### corpus_005sections, doc384-64 (10 documents, 10 texts in, 4746 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 12.6 | 5960 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.780/1.000 · 0.780/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 32.9 | 15618 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 0.80 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 42 / 0 / 0 / 11 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.30 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 43 / 0 / 0 / 10 | 0 / 0 / 0 / 0 |

### corpus_005sections, doc384-128 (10 documents, 10 texts in, 4746 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 9.9 | 4706 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.780/1.000 · 0.780/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 26.1 | 12384 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | none |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 1.01 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 40 / 0 / 0 / 11 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.38 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 42 / 0 / 0 / 10 | 0 / 0 / 0 / 0 |

### corpus_005sections, doc256-64 (10 documents, 10 texts in, 4746 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 10.7 | 5099 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.909 · 1.000/0.943 · 1.000/0.943 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/0.943 · 1.000/0.909 · 1.000/0.909 | reference |
| single_street_0.5 | 27.7 | 13154 | 1.000/0.978 · 1.000/0.988 · 1.000/0.988 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/0.980 · 1.000/0.980 · 1.000/0.980 | Person doc_set precision, Person occurrence overlap_precision |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 0.93 | 0 / 0 / 0 / 0 | 0 / 3 / 0 / 0 | 40 / 0 / 0 / 10 | 0 / 5 / 0 / 0 |
| single_street_0.5 | 0.36 | 0 / 2 / 0 / 0 | 0 / 0 / 0 / 0 | 40 / 0 / 0 / 10 | 0 / 1 / 0 / 0 |

### corpus_005sections, doc192-48 (10 documents, 10 texts in, 4746 words)

| config | texts/s | words/s | Person doc R/P · occ R/P · exact R/P | Organization doc R/P · occ R/P · exact R/P | Street doc R/P · occ R/P · exact R/P | City doc R/P · occ R/P · exact R/P | worse than current |
|---|---|---|---|---|---|---|---|
| current_3pass | 11.0 | 5231 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 1.000/0.909 · 1.000/0.943 · 1.000/0.943 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | reference |
| single_street_0.5 | 27.4 | 13023 | 1.000/0.978 · 1.000/0.988 · 1.000/0.988 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | 0.800/1.000 · 0.800/1.000 · 0.000/0.000 | 1.000/1.000 · 1.000/1.000 · 1.000/1.000 | Person doc_set precision, Person occurrence overlap_precision |

Errors per label (boundary / wrong label / invented / missed), median seconds per run:

| config | seconds | Person | Organization | Street | City |
|---|---|---|---|---|---|
| current_3pass | 0.91 | 0 / 0 / 0 / 0 | 0 / 3 / 0 / 0 | 40 / 0 / 0 / 10 | 0 / 0 / 0 / 0 |
| single_street_0.5 | 0.36 | 0 / 2 / 0 / 0 | 0 / 0 / 0 / 0 | 41 / 0 / 0 / 10 | 0 / 0 / 0 / 0 |


