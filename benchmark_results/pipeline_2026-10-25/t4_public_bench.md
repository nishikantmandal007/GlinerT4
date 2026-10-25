### fastino/gliner2-base-v1

| dataset | sentences | stock_fp32 F1 | stock_fp16 F1 | glinert4 F1 | stock_fp32 sent/s | stock_fp16 sent/s | glinert4 sent/s | glinert4 speedup vs fp16 / fp32 | identical to fp16 |
|---|---|---|---|---|---|---|---|---|---|
| crossner_ai | 431 | 0.4866 | 0.4862 | 0.4865 | 90.1 | 65.2 | 250.8 | 3.84x / 2.78x | 97.9% |
| crossner_literature | 416 | 0.5243 | 0.5243 | 0.5246 | 95.1 | 132.8 | 326.5 | 2.46x / 3.43x | 99.3% |
| crossner_music | 465 | 0.6338 | 0.6337 | 0.6337 | 92.2 | 151.1 | 312.5 | 2.07x / 3.39x | 98.5% |
| crossner_politics | 651 | 0.6370 | 0.6374 | 0.6377 | 98.2 | 153.6 | 264.2 | 1.72x / 2.69x | 98.2% |
| crossner_science | 543 | 0.5094 | 0.5095 | 0.5093 | 83.3 | 107.8 | 312.3 | 2.90x / 3.75x | 98.9% |
| wikiann_en | 1000 | 0.5204 | 0.5212 | 0.5212 | 191.3 | 116.9 | 382.2 | 3.27x / 2.00x | 99.7% |

CrossNER average micro F1 (the paper reports 0.590 for GLiNER2): stock_fp32 0.5582, stock_fp16 0.5582, glinert4 0.5584

Person (glinert4): precision / recall / F1, gold names

| dataset | P | R | F1 | gold |
|---|---|---|---|---|
| crossner_ai | 0.254 | 0.866 | 0.393 | 67 |
| crossner_literature | 0.276 | 0.766 | 0.406 | 175 |
| crossner_music | 0.189 | 0.700 | 0.298 | 80 |
| crossner_politics | 0.570 | 0.893 | 0.696 | 354 |
| crossner_science | 0.293 | 0.887 | 0.440 | 230 |
| wikiann_en | 0.585 | 0.786 | 0.671 | 504 |

### fastino/gliner2-multi-v1

| dataset | sentences | stock_fp32 F1 | stock_fp16 F1 | glinert4 F1 | stock_fp32 sent/s | stock_fp16 sent/s | glinert4 sent/s | glinert4 speedup vs fp16 / fp32 | identical to fp16 |
|---|---|---|---|---|---|---|---|---|---|
| crossner_ai | 431 | 0.4909 | 0.4908 | 0.4908 | 78.2 | 97.3 | 376.7 | 3.87x / 4.82x | 99.1% |
| crossner_literature | 416 | 0.5346 | 0.5355 | 0.5353 | 75.1 | 96.5 | 364.6 | 3.78x / 4.85x | 98.6% |
| crossner_music | 465 | 0.6330 | 0.6330 | 0.6328 | 71.9 | 94.9 | 342.5 | 3.61x / 4.77x | 96.3% |
| crossner_politics | 651 | 0.6125 | 0.6129 | 0.6126 | 75.1 | 100.1 | 307.9 | 3.08x / 4.10x | 95.5% |
| crossner_science | 543 | 0.5211 | 0.5211 | 0.5211 | 64.9 | 74.2 | 307.3 | 4.14x / 4.73x | 97.8% |
| wikiann_en | 1000 | 0.5990 | 0.5988 | 0.5988 | 169.3 | 79.4 | 482.8 | 6.08x / 2.85x | 99.8% |
| wikiann_hi | 1000 | 0.6107 | 0.6107 | 0.6109 | 174.5 | 169.7 | 561.2 | 3.31x / 3.21x | 99.9% |
| wikiann_bn | 1000 | 0.5593 | 0.5593 | 0.5593 | 204.8 | 195.1 | 590.8 | 3.03x / 2.89x | 100.0% |
| wikiann_ta | 1000 | 0.5409 | 0.5409 | 0.5409 | 175.1 | 224.6 | 586.3 | 2.61x / 3.35x | 100.0% |
| wikiann_te | 1000 | 0.5519 | 0.5524 | 0.5533 | 154.3 | 171.6 | 541.9 | 3.16x / 3.51x | 99.5% |
| wikiann_mr | 1000 | 0.5374 | 0.5379 | 0.5374 | 171.4 | 302.8 | 568.9 | 1.88x / 3.32x | 99.7% |
| wikiann_gu | 100 | 0.6093 | 0.6093 | 0.6093 | 105.6 | 68.7 | 543.4 | 7.91x / 5.15x | 100.0% |
| wikiann_kn | 100 | 0.5174 | 0.5174 | 0.5174 | 162.8 | 295.3 | 576.8 | 1.96x / 3.55x | 100.0% |
| wikiann_ml | 1000 | 0.5888 | 0.5888 | 0.5888 | 161.1 | 239.1 | 568.2 | 2.38x / 3.53x | 100.0% |
| wikiann_pa | 100 | 0.4274 | 0.4255 | 0.4274 | 188.6 | 313.3 | 561.3 | 1.79x / 2.98x | 99.0% |
| wikiann_de | 1000 | 0.7014 | 0.7021 | 0.7019 | 165.8 | 263.5 | 556.2 | 2.11x / 3.35x | 99.6% |
| wikiann_fr | 1000 | 0.6392 | 0.6394 | 0.6394 | 182.3 | 327.8 | 584.3 | 1.78x / 3.21x | 99.8% |
| wikiann_es | 1000 | 0.6073 | 0.6070 | 0.6070 | 197.8 | 294.9 | 563.0 | 1.91x / 2.85x | 100.0% |

CrossNER average micro F1 (the paper reports 0.590 for GLiNER2): stock_fp32 0.5584, stock_fp16 0.5587, glinert4 0.5585

Person (glinert4): precision / recall / F1, gold names

| dataset | P | R | F1 | gold |
|---|---|---|---|---|
| crossner_ai | 0.267 | 0.881 | 0.410 | 67 |
| crossner_literature | 0.224 | 0.817 | 0.351 | 175 |
| crossner_music | 0.141 | 0.863 | 0.242 | 80 |
| crossner_politics | 0.409 | 0.955 | 0.573 | 354 |
| crossner_science | 0.288 | 0.865 | 0.433 | 230 |
| wikiann_en | 0.674 | 0.845 | 0.750 | 504 |
| wikiann_hi | 0.714 | 0.853 | 0.777 | 450 |
| wikiann_bn | 0.586 | 0.755 | 0.660 | 347 |
| wikiann_ta | 0.594 | 0.725 | 0.653 | 422 |
| wikiann_te | 0.537 | 0.822 | 0.649 | 381 |
| wikiann_mr | 0.561 | 0.720 | 0.631 | 375 |
| wikiann_gu | 0.559 | 0.808 | 0.661 | 47 |
| wikiann_kn | 0.489 | 0.667 | 0.564 | 33 |
| wikiann_ml | 0.609 | 0.840 | 0.706 | 407 |
| wikiann_pa | 0.434 | 0.590 | 0.500 | 39 |
| wikiann_de | 0.768 | 0.847 | 0.805 | 476 |
| wikiann_fr | 0.743 | 0.888 | 0.809 | 411 |
| wikiann_es | 0.750 | 0.844 | 0.794 | 410 |

Datasets that could not be loaded:

- naamapadam_as: ai4bharat/naamapadam/as could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'as' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_bn: ai4bharat/naamapadam/bn could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'bn' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_gu: ai4bharat/naamapadam/gu could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'gu' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_hi: ai4bharat/naamapadam/hi could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'hi' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_kn: ai4bharat/naamapadam/kn could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'kn' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_ml: ai4bharat/naamapadam/ml could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'ml' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_mr: ai4bharat/naamapadam/mr could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'mr' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_or: ai4bharat/naamapadam/or could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'or' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_pa: ai4bharat/naamapadam/pa could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'pa' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_ta: ai4bharat/naamapadam/ta could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'ta' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no
- naamapadam_te: ai4bharat/naamapadam/te could not be loaded: {}: RuntimeError: Dataset scripts are no longer supported, but found naamapadam.py | {'revision': 'refs/convert/parquet'}: ValueError: BuilderConfig 'te' not found. Available: ['default'] | {'trust_remote_code': True}: RuntimeError: Dataset scripts are no

