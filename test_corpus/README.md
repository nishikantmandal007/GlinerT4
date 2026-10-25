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

## Testing Single-Pass Speed and Precision

Use `bench/test_single_pass_realistic.py` to evaluate the recalibrated single-pass pipeline:

```bash
# Verify dataset structure and entity counts locally (CPU / stand-in mode):
python bench/test_single_pass_realistic.py --corpus test_corpus/corpus_040sections.json --standin

# Quick smoke test on short documents:
python bench/test_single_pass_realistic.py --corpus test_corpus/corpus_005sections.json --standin

# Full single-pass benchmark on a GPU runtime (Colab T4 or VM):
python bench/test_single_pass_realistic.py --corpus test_corpus/corpus_040sections.json \
                                           --layout single_person_first \
                                           --device cuda
```

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
