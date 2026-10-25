# Writes notebooks/glinert4_speed.ipynb (no outputs). Run it in Colab on a T4 to fill the outputs.
#   python notebooks/make_notebook.py
import json
from pathlib import Path

REPO = "https://github.com/nishikantmandal007/GlinerT4"
REF = "claude/upbeat-bell-km1ebk"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(f"""
# Making GLiNER2 fast on a T4 without losing accuracy

I wanted GLiNER2 to run fast on a Tesla T4, the cheapest GPU on most clouds, without changing a single prediction.
This notebook runs the same model on the same text four times, adding one optimisation at a time, and scores every
run against the gold labels. The starting point is GLiNER2 in fp16: in my tests fp32 was slower and no more accurate.

| Step | What changes |
|---|---|
| fp16 | GLiNER2 with half-precision weights (`quantize=True`) |
| + disentangled_flash | GLiNER2's fused Triton attention (`attention_backend="disentangled_flash"`) |
| + glinert4 kernels | my CUDA attention kernel for the T4, residual add fused into LayerNorm, a count head that skips work it throws away, a word-split cache |
| + pipeline | the CPU prepares batch *n+1* while the GPU runs batch *n*, and lengths are padded to multiples of 32 |

The text is [CrossNER](https://github.com/zliucr/CrossNER), the zero-shot NER benchmark from the GLiNER2 paper: its
five English test sets, once as single sentences and once joined into documents of about 1,000 words. Every label
of a domain goes in one prompt, threshold 0.5, strict F1 (label, start and end must match).

Pinned: gliner2 2.1.0, `fastino/gliner2-multi-v1`, disentangled-flash 1.1.1, glinert4 from [{REPO}]({REPO}).
Pick a GPU runtime (Runtime → Change runtime type → T4). It takes about 15 minutes.
""")

code(f"""
import sys

if "google.colab" in sys.modules:
    !pip install -q uv
    !uv pip install --system -q "gliner2[local]==2.1.0" disentangled-flash==1.1.1 ninja
    !uv pip install --system -q --no-deps "glinert4 @ git+{REPO}@{REF}#subdirectory=production"
""")

code("""
import html
import time
import urllib.request

import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
import transformers
from IPython.display import HTML, Markdown, display

import gliner2
import glinert4
from gliner2 import AutoExtractor
from glinert4 import pipeline, runtime
from glinert4.ops import ops

CHECKPOINT = "fastino/gliner2-multi-v1"
BATCH, THRESHOLD, DOC_WORDS = 16, 0.5, 1000
CROSSNER = "https://raw.githubusercontent.com/zliucr/CrossNER/main/ner_data/{}/test.txt"
DOMAINS = ("ai", "literature", "music", "politics", "science")
LABEL_NAMES = {"programlang": "programming language", "misc": "miscellaneous", "musicalartist": "musical artist",
               "musicalinstrument": "musical instrument", "musicgenre": "music genre", "literarygenre": "literary genre",
               "politicalparty": "political party", "academicjournal": "academic journal",
               "astronomicalobject": "astronomical object", "chemicalcompound": "chemical compound",
               "chemicalelement": "chemical element"}

ops()
print(f"gliner2 {gliner2.__version__} | transformers {transformers.__version__} | torch {torch.__version__} | "
      f"{torch.cuda.get_device_name()}")


# a markdown table
def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])
""")

md("""
## 1. Data

CrossNER's test sets straight from GitHub. Documents are consecutive sentences joined until they reach about 1,000
words, with the gold spans moved along, so long-document handling (GLiNER2's 384-word windows with 64 words of
overlap) is scored too.
""")

code("""
# one CrossNER test set as (text, gold spans), a gold span being (label, start, end)
def crossner(domain):
    sentences, tokens, tags = [], [], []
    for line in urllib.request.urlopen(CROSSNER.format(domain)).read().decode().splitlines() + [""]:
        parts = line.split("\\t")
        if len(parts) == 2:
            tokens.append(parts[0])
            tags.append(parts[1])
        elif tokens:
            sentences.append(from_bio(tokens, tags))
            tokens, tags = [], []
    return sentences


# the tokens joined by spaces, and the character spans of their BIO tags
def from_bio(tokens, tags):
    starts = [sum(len(token) + 1 for token in tokens[:i]) for i in range(len(tokens))]
    spans, current = [], None
    for i, tag in enumerate(tags + ["O"]):
        kind = tag[2:] if tag != "O" else None
        if current and (tag.startswith("B-") or kind != current[0]):
            spans.append((LABEL_NAMES.get(current[0], current[0]), starts[current[1]], starts[i - 1] + len(tokens[i - 1])))
            current = None
        if kind and current is None:
            current = (kind, i)
    return " ".join(tokens), spans


# consecutive sentences joined into documents of about DOC_WORDS words
def documents(sentences):
    docs, text, spans = [], "", []
    for sentence, gold in sentences:
        text += " " if text else ""
        spans += [(label, start + len(text), end + len(text)) for label, start, end in gold]
        text += sentence
        if len(text.split()) >= DOC_WORDS:
            docs.append((text, spans))
            text, spans = "", []
    if text:
        docs.append((text, spans))
    return docs


data = {}
for domain in DOMAINS:
    sentences = crossner(domain)
    data[domain] = {"labels": sorted({label for _, gold in sentences for label, _, _ in gold}),
                    "sentences": sentences, "documents": documents(sentences)}

display(Markdown(table(["Domain", "labels", "sentences", "documents", "words"],
                       [[d, len(v["labels"]), len(v["sentences"]), len(v["documents"]),
                         f"{sum(len(t.split()) for t, _ in v['sentences']):,}"] for d, v in data.items()])))
""")

md("""
## 2. Four runs, one optimisation at a time

Every run calls the same GLiNER2 long-text API with the same batch size, labels and threshold. Only the last run
swaps it for glinert4's pipelined version of that API.
""")

code("""
def load(**options):
    return runtime.freeze_eval(AutoExtractor.from_pretrained(CHECKPOINT, map_location="cuda", **options))


def predict(model, texts, labels, pipelined):
    if pipelined:
        return pipeline.extract_entities_long_many(model, [(texts, labels, THRESHOLD)], batch_size=BATCH,
                                                   include_confidence=True, include_spans=True)[0]
    return model.batch_extract_entities_long(texts, labels, batch_size=BATCH, threshold=THRESHOLD,
                                            include_confidence=True, include_spans=True)


def spans(results, labels):
    return [{(label, e["start"], e["end"]) for label in labels for e in result["entities"].get(label, [])}
            for result in results]


# seconds and predicted spans for every text, per kind of input
def measure(model, pipelined=False):
    run = {}
    for kind in ("sentences", "documents"):
        seconds, found = 0.0, []
        for domain in data.values():
            texts = [text for text, _ in domain[kind]]
            predict(model, texts[:BATCH], domain["labels"], pipelined)
            torch.cuda.synchronize()
            start = time.perf_counter()
            found += spans(predict(model, texts, domain["labels"], pipelined), domain["labels"])
            torch.cuda.synchronize()
            seconds += time.perf_counter() - start
        run[kind] = {"seconds": seconds, "found": found}
    return run


runs = {}
stock = load(quantize=True)
runs["fp16"] = measure(stock)
fast = load(quantize=True, attention_backend="disentangled_flash")
runs["+ disentangled_flash"] = measure(fast)
glinert4.install()
runs["+ glinert4 kernels"] = measure(fast)
runtime.add_length_buckets(fast, 32)
runs["+ pipeline"] = measure(fast, pipelined=True)
""")

md("""
## 3. Results

Speed is relative to GLiNER2 fp16. "Same as fp16" is the share of texts whose predicted spans are exactly the ones
GLiNER2 fp16 predicts; anything below 100% is rounding flipping a score that sits right at the threshold.
""")

code("""
def f1(kind, found):
    gold = [set(g) for domain in data.values() for _, g in domain[kind]]
    hits = sum(len(g & p) for g, p in zip(gold, found))
    predicted, expected = sum(map(len, found)), sum(map(len, gold))
    return 2 * hits / (predicted + expected)


for kind in ("sentences", "documents"):
    count = sum(len(domain[kind]) for domain in data.values())
    base = runs["fp16"][kind]
    rows = [[step, f"{run[kind]['seconds']:.1f} s", f"{count / run[kind]['seconds']:.1f}",
             f"**{base['seconds'] / run[kind]['seconds']:.1f}×**", f"{f1(kind, run[kind]['found']):.3f}",
             f"{sum(a == b for a, b in zip(run[kind]['found'], base['found'])) / count:.1%}"]
            for step, run in runs.items()]
    display(Markdown(f"**{count:,} {kind}**\\n\\n" + table(["Step", "time", f"{kind}/s", "speed-up", "F1",
                                                             "same as fp16"], rows)))
""")

code("""
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#d5d4cf", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True, sharey=True)
top = max(runs["fp16"][kind]["seconds"] / run[kind]["seconds"] for run in runs.values() for kind in ("sentences", "documents"))
for ax, kind in zip(axes, ("sentences", "documents")):
    speedup = [runs["fp16"][kind]["seconds"] / run[kind]["seconds"] for run in runs.values()]
    bars = ax.barh(list(runs), speedup, height=0.55, color="#2a78d6")
    ax.bar_label(bars, labels=[f"{s:.1f}×" for s in speedup], padding=5, color="#0b0b0b")
    ax.set_title(f"{kind.capitalize()}: times faster than GLiNER2 fp16", loc="left", color="#0b0b0b", pad=12)
    ax.set_xlim(0, top * 1.15)
    ax.grid(axis="x", color="#ecebe7")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].invert_yaxis()
fig.tight_layout()
plt.show()
""")

md("""
## 4. Try your own text

GLiNER2 fp16 against glinert4 on any text and labels. The timings are the best of three runs.
""")

code("""
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EXAMPLE = ("Satya Nadella, the chief executive of Microsoft, met Prime Minister Narendra Modi in New Delhi on "
           "Tuesday to discuss a cloud partnership with Reliance Industries.")


def timed(model, text, labels, pipelined):
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        found = spans(predict(model, [text], labels, pipelined), labels)[0]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return found, min(times) * 1000


def highlight(text, found, labels):
    parts, last = [], 0
    for label, start, end in sorted(found, key=lambda span: span[1]):
        if start < last:
            continue
        color = PALETTE[labels.index(label)]
        parts += [html.escape(text[last:start]),
                  f'<mark style="background:{color}26;border-bottom:2px solid {color};border-radius:3px;'
                  f'padding:1px 3px;color:inherit">{html.escape(text[start:end])}'
                  f'<span style="font-size:0.7em;opacity:0.7;margin-left:4px">{html.escape(label)}</span></mark>']
        last = end
    return "".join(parts + [html.escape(text[last:])])


def compare(text, labels):
    labels = [label.strip() for label in labels.split(",") if label.strip()][:len(PALETTE)]
    glinert4.uninstall()
    before, before_ms = timed(stock, text, labels, False)
    glinert4.install()
    after, after_ms = timed(fast, text, labels, True)
    verdict = "same entities" if before == after else "entities differ"
    rows = [("GLiNER2 fp16", before_ms, before), ("glinert4", after_ms, after)]
    display(HTML("".join(
        f'<div style="margin:10px 0;line-height:1.9"><b>{name}</b> · {ms:.0f} ms<br>{highlight(text, found, labels)}</div>'
        for name, ms, found in rows) + f"<p><b>{before_ms / after_ms:.1f}× faster</b>, {verdict}.</p>"))


compare(EXAMPLE, "person, organization, location, date")
""")

code("""
text_box = widgets.Textarea(value=EXAMPLE, layout=widgets.Layout(width="100%", height="110px"))
label_box = widgets.Text(value="person, organization, location, date", description="Labels",
                         layout=widgets.Layout(width="100%"))
button = widgets.Button(description="Compare", button_style="primary", icon="bolt")
output = widgets.Output()


def on_click(_):
    with output:
        output.clear_output()
        compare(text_box.value, label_box.value)


button.on_click(on_click)
display(widgets.VBox([text_box, label_box, button, output]))
""")

md("""
## What made the difference

- **Profile first.** On a T4, attention and the matrix multiplications each took about 45% of the GPU time. cuBLAS was
  already near the hardware limit, so the attention was the part worth rewriting.
- **Write for the GPU you have.** The T4 (sm_75) has fp16 tensor cores but no asynchronous copies, and recent Triton
  releases stop using its tensor cores. The kernel is CUDA C++ with `mma.sync` for this one chip.
- **Exact before clever.** Every optimisation here gives the same predictions up to fp16 rounding; the count head and
  the pipeline are bit-identical to stock GLiNER2.
- **Keep the GPU busy.** Once the kernels were fast, the GPU sat idle while Python prepared the next batch.
  Overlapping the two was the last big step.
""")

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                          "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                          "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}
target = Path(__file__).with_name("glinert4_speed.ipynb")
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(target)
