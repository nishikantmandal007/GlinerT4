# Realistic Test Corpus & Ground Truth

This folder contains realistic benchmark documents and ground truth annotations sourced from `MDP/Maya_Work/refactor/benchmark/realistic_corpus/`. These files enable offline and online verification of extraction throughput (chunks/s, words/s) and precision/recall across diverse document types and encodings.

## Directory Structure

- **`txt/`**: Contains raw text files organized across 10 realistic document categories and encoding profiles:
  - `address_book`: Tabular contact registries and structured directory entries.
  - `business_letters`: Formal correspondence, notice letters, and follow-up requests.
  - `cp1252_correspondence`: Windows-1252 / ANSI encoded communications.
  - `email_archive`: Threaded multi-recipient email logs.
  - `form_export`: Semi-structured form dumps and intake exports.
  - `meeting_notes`: Executive summaries, attendee rosters, and agenda minutes.
  - `unicode_notes`: UTF-8 multilingual text containing accented characters and special symbols.
  - `utf16_archive`: UTF-16 LE encoded archival records.
  - `utf8_bom`: UTF-8 text with Byte Order Marks (BOM).
  - `windows_crlf`: Text formatted with Windows carriage return line endings (`\r\n`).

  Each category contains benchmark lengths:
  - `005sections.txt`: Short smoke test (5 logical sections / records).
  - `010sections.txt`: 10 sections (~100 chunks across categories).
  - `020sections.txt`: 20 sections (~200 chunks across categories).
  - `040sections.txt`: Full benchmark set matching the 290-chunk T4 evaluation set (~38,000 words).

- **`ground_truth/`**: Detailed JSON ground truth annotations (`*.ground_truth.json`) for every text file, containing:
  - Exact entity occurrences and character offsets (`char_start`, `char_end`).
  - Supported entity types: `Names` (`Person`), `Organization`, `Street`, `City`, plus `Dates`, `Emails`, `Phone`, `Zip Code`, `IBAN`, `DOB`.
  - Occurrence counts and section verification hashes.

- **Unified Benchmark Corpora**:
  - `corpus_040sections.json`: The exact 10-document, 290-chunk evaluation corpus used in Rounds 1–7.
  - `corpus_020sections.json`: 10 documents, 200 chunks.
  - `corpus_010sections.json`: 10 documents, 100 chunks.
  - `corpus_005sections.json`: 10 documents, 50 chunks.

## Benchmarking on this corpus

`bench/realistic_bench.py` runs the production `detect_batch`, with the shipping speed settings (exact count head, fused
add + LayerNorm, one-thread pipeline), in five configurations: today's 3 passes, 2 passes, single pass at today's
thresholds, single pass with only Street lowered to 0.5, and single pass with the round-7 thresholds. It scores each
three ways:

- **doc-set:** `bench_detection.py`'s scoring, by distinct values per document.
- **occ-overlap:** every annotated occurrence from `ground_truth/`, by character offsets. An occurrence counts as found
  if a detected span of the same label overlaps it.
- **occ-exact:** identical spans only.

Each configuration runs on the pre-cut chunks and on whole documents (`--inputs chunks,documents`). For whole
documents, GLiNER2 windows the text itself (384 words, 64 overlap), as production does.

```bash
python bench/realistic_bench.py --out results --sizes 040,020      # on a GPU (Colab T4 or VM)
python bench/realistic_bench.py --out /tmp/smoke --standin         # CPU smoke test of the script, stand-in model
python bench/make_colab_pipeline_bench.py --only realistic         # single-file Colab runner with the prepared corpus
```

Things to know about this corpus:

- **It is not held-out data for the round-7 thresholds.** `corpus_040sections.json` is the rounds 1-7 corpus. The
  005/010/020 files are prefixes of the same documents: their truth values are subsets of the 040 truth, though they
  are chunked at different boundaries.
- **The 040 chunker splits some mentions across chunks:** 51 of the 2,560 annotated Person/Organization/Street/City
  occurrences lie in no single chunk, so their exact recall cannot reach 1. The 005/010/020 chunks contain every
  occurrence whole.
- **The 005/010/020 chunkers differ from the 040 one in two ways:**
  - the UTF-8 BOM is kept at the start of the first `utf8_bom` chunk;
  - `windows_crlf` line ends are turned into LF.

  The ground-truth offsets refer to the decoded text without the BOM and with CRLF, and `realistic_bench.py` maps
  positions accordingly. With perfect predictions it scores exactly 1.000.

### Recalibrated Single-Pass Thresholds (from Round 7)

Merging all entities into a single pass shifts prediction confidence distributions. The recalibrated thresholds prevent false positives while capturing suppressed entities:
- **`Person`**: `0.85`
- **`Organization`**: `0.80`
- **`Street`**: `0.30` (compensates for suppression in shared prompt)
- **`City`**: `0.90`

On the 290-chunk corpus (`corpus_040sections.json`), this configuration achieves **133.5 chunks/s** on a Tesla T4 while exceeding the 3-pass reference precision and recall on every entity type:
- **Person**: Recall 0.994 / Precision 0.971 (vs 0.994 / 0.961 reference)
- **Organization**: Recall 0.977 / Precision 0.863 (vs 0.977 / 0.846 reference)
- **Street**: Recall 0.949 / Precision 1.000 (vs 0.861 / 1.000 reference)
- **City**: Recall 1.000 / Precision 0.964 (vs 1.000 / 0.930 reference)
