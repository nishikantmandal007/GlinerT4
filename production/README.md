# glinert4

GLiNER2 (`fastino/gliner2-multi-v1`, `gliner2==2.1.0`) entity detection tuned for one NVIDIA T4 (sm_75).

| Component | What it does | Output vs stock |
|---|---|---|
| `csrc/attention.cu` | Fused DeBERTa disentangled attention (c2p + p2c), `mma.sync` m16n8k8, padded and packed batches | within fp16 rounding of `disentangled_flash` |
| `csrc/add_layernorm.cu` | Residual add fused into LayerNorm (hidden 768) | bit-identical to PyTorch |
| `count.py` | Count head runs only the GRU step entity decoding keeps, same GEMM shapes | bit-identical |
| `pipeline.py` | One-thread CPU/GPU pipeline: encoder of batch *n+1* on its own stream while batch *n* is decoded | bit-identical |
| `detector.py` | Single label pass, whole documents in 384/64 word windows, table-cell confirmation, span resolution | accuracy measured, see below |

## Results (Tesla T4, 290-chunk realistic corpus, 38k words)

| Setup | Time | Words/s | Person R/P | Organization R/P | Street R/P | City R/P |
|---|---:|---:|---|---|---|---|
| 3 label passes, pre-cut chunks | 5.10 s | 7,441 | .994/.961 | .977/.846 | .861/1.000 | 1.000/.930 |
| **glinert4 defaults** | **2.60 s** | **14,595** | **1.000/.997** | **1.000/.992** | **.871/1.000** | **1.000/.994** |

Recall/precision are per document on distinct values, scored against the corpus ground truth. Occurrence-level
scores and the full error analysis are in `benchmark_results/pipeline_2026-10-25/` of the research repository. The
thresholds were chosen on this corpus, so confirm on held-out data before relying on them.

## Install

```bash
pip install .
python scripts/build_ops.py
```

`build_ops.py` compiles the CUDA ops for sm_75 into `glinert4/_glinert4.so`. It needs `nvcc` and the same torch as
the runtime. Without a prebuilt library, the ops are compiled on first use.

## Use

```python
from glinert4 import Detector

detector = Detector()
found = detector.detect(["Anna Müller works at Siemens AG, Hauptstraße 5, Berlin."])
found[0]["values"]
found[0]["occurrences"]
```

`detect` takes whole documents in any length and returns, per text, the distinct values per label and every
occurrence with character offsets and score. `labels=` restricts or reorders the labels for a call.

```python
from glinert4 import Detector, DetectorConfig

config = DetectorConfig(
    labels=("Person", "Organization", "Street", "City", "Insurance Number"),
    thresholds={"Organization": 0.85, "Street": 0.5},
    default_threshold=0.65,
    window_words=384,
    window_overlap=64,
    batch_size=16,
)
detector = Detector(config, device="cuda:0")
```

`Detector(..., fast=False)` runs plain gliner2 with the same settings, as a reference.
`glinert4.install()` and `glinert4.uninstall()` apply or remove the kernels and patches process-wide.

## Tests

```bash
pip install ".[test]"
pytest
```

`tests/test_parity.py` runs on CPU against a small random-weight model and checks bit equality of the count head,
the pipeline and the detector against stock gliner2. `tests/test_kernels.py` needs a GPU and the
`fastino/gliner2-multi-v1` weights. It checks that the fused add + LayerNorm is bit-identical to PyTorch, and that
encoder outputs match `disentangled_flash` on padded and unpadded batches of 96 to 1024 tokens.

## Layout

```
glinert4/
  __init__.py      Detector, DetectorConfig, install, uninstall
  detector.py      single-pass detection and span post-processing
  pipeline.py      pipelined long-document extraction
  count.py         exact count head
  attention.py     disentangled_flash integration of the attention kernel
  layernorm.py     DeBERTa output blocks on the fused add + LayerNorm
  runtime.py       model loading, length buckets, word-split cache, version check
  ops.py           op loading and building
  csrc/            CUDA kernels and the torch op registration
scripts/build_ops.py
tests/
```
