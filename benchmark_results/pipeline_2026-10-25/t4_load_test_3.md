Load test on Tesla T4: page requests (mean 542.6 words, 1100.4 tokens), batches up to 64, 5.0 ms collection window, 30 s per level, queue limit none, 3 retries

| setup | load | requests/min | tokens/min | words/min | p50 ms | p95 ms | p99 ms | mean batch | GPU busy | rejected | failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| glinert4 | 1 clients | 1,588 | 1,748,480 | 861,157 | 35.1 | 72.2 | 75.1 | 1 | 68.2% | 0.0% | 0 |
| glinert4 | 4 clients | 1,785 | 1,966,740 | 968,942 | 125.9 | 166.5 | 171.6 | 4 | 80.2% | 0.0% | 0 |
| glinert4 | 16 clients | 1,886 | 2,078,294 | 1,023,435 | 508.2 | 596.1 | 602.4 | 16 | 85.2% | 0.0% | 0 |
| glinert4 | 64 clients | 1,941 | 2,136,418 | 1,052,917 | 1984.5 | 2024.8 | 2937.4 | 61.9 | 91.0% | 0.0% | 0 |

{
 "device": "Tesla T4",
 "request": "page",
 "requests_in_pool": 70,
 "mean_words": 542.6,
 "mean_tokens": 1100.4,
 "max_batch": 64,
 "max_wait_ms": 5.0,
 "queue_limit": 0,
 "retries": 3,
 "backoff_ms": 100.0,
 "duration_s": 30.0,
 "levels": [
  {
   "setup": "glinert4",
   "mode": "closed",
   "level": 1,
   "requests": 794,
   "seconds": 30.01,
   "rejected_pct": 0.0,
   "failed": 0,
   "rpm": 1588,
   "tpm": 1748480,
   "wpm": 861157,
   "p50_ms": 35.1,
   "p95_ms": 72.2,
   "p99_ms": 75.1,
   "mean_ms": 37.8,
   "mean_batch": 1,
   "worker_busy": 0.861,
   "gpu_util": 68.2
  },
  {
   "setup": "glinert4",
   "mode": "closed",
   "level": 4,
   "requests": 896,
   "seconds": 30.11,
   "rejected_pct": 0.0,
   "failed": 0,
   "rpm": 1785,
   "tpm": 1966740,
   "wpm": 968942,
   "p50_ms": 125.9,
   "p95_ms": 166.5,
   "p99_ms": 171.6,
   "mean_ms": 134.4,
   "mean_batch": 4,
   "worker_busy": 0.961,
   "gpu_util": 80.2
  },
  {
   "setup": "glinert4",
   "mode": "closed",
   "level": 16,
   "requests": 944,
   "seconds": 30.03,
   "rejected_pct": 0.0,
   "failed": 0,
   "rpm": 1886,
   "tpm": 2078294,
   "wpm": 1023435,
   "p50_ms": 508.2,
   "p95_ms": 596.1,
   "p99_ms": 602.4,
   "mean_ms": 508.9,
   "mean_batch": 16,
   "worker_busy": 0.989,
   "gpu_util": 85.2
  },
  {
   "setup": "glinert4",
   "mode": "closed",
   "level": 64,
   "requests": 991,
   "seconds": 30.64,
   "rejected_pct": 0.0,
   "failed": 0,
   "rpm": 1941,
   "tpm": 2136418,
   "wpm": 1052917,
   "p50_ms": 1984.5,
   "p95_ms": 2024.8,
   "p99_ms": 2937.4,
   "mean_ms": 1977.7,
   "mean_batch": 61.9,
   "worker_busy": 0.999,
   "gpu_util": 91.0
  }
 ]
}
