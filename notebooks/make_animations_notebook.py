# Writes notebooks/animations.ipynb from notebooks/animations/*.py: a Colab notebook that installs Manim Community
# Edition and the two typefaces, writes the shared style and each clip with %%writefile, renders 1080p MP4 and
# exports an 800 px GIF. The .py files are the source; edit them and re-run this script.
#   python notebooks/make_animations_notebook.py
import json
from pathlib import Path

here = Path(__file__).parent / "animations"
CLIPS = [("clip_a_read_once.py", "ReadOnce", "A. Read once, ask everything",
          "Instead of reading the document three times, read it once and ask all the questions."),
         ("clip_b_waiting.py", "Waiting", "B. The CPU wasn't working, it was waiting",
          "The slow line wasn't slow: it was where the CPU waited for the GPU."),
         ("clip_c_overlap.py", "Overlap", "C. Overlap keeps entities whole",
          "Overlap means no entity is ever only seen cut in half."),
         ("clip_d_deberta.py", "ThreeAttentions", "D. DeBERTa's three attentions",
          "DeBERTa looks at what words are and where they are, so it needs a special kernel."),
         ("clip_e_tiles.py", "Tiles", "E. Feeding the tensor cores",
          "The kernel re-packs the work into the exact shape the T4's fast units want.")]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# GLiNER2 on a T4: the animations

Each clip makes one claim, in one sentence. They share one visual system (`style.py`): a fixed palette (CPU blue,
GPU orange, wasted work grey, correct green, error red), Inter for labels, JetBrains Mono for numbers, a charcoal
background, and slow eased motion with one thing moving at a time.

Runs on a Colab CPU runtime; rendering a clip at 1080p takes a few minutes.
""")

code("""
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    !apt-get -qq install -y libcairo2-dev libpango1.0-dev ffmpeg fonts-inter fonts-jetbrains-mono > /dev/null 2>&1 || true
    !pip -q install manim
fonts = Path.home() / ".fonts"
fonts.mkdir(exist_ok=True)
for name, url in [("Inter.ttf", "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf"),
                  ("JetBrainsMono.ttf", "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf")]:
    if not (fonts / name).exists():
        !wget -q -O {fonts / name} "{url}"
!fc-cache -f > /dev/null && fc-list : family | grep -E "^(Inter|JetBrains Mono)$" | sort -u
""")

md("## The visual system")
code("%%writefile style.py\n" + (here / "style.py").read_text())

code("""
from IPython.display import Video, display


def render(source, scene):
    !manim -qh --disable_caching {source} {scene}
    video = f"media/videos/{Path(source).stem}/1080p30/{scene}.mp4"
    display(Video(video, embed=True, width=960))
    return video


def gif(video, out):
    !ffmpeg -v error -y -i {video} -vf "fps=15,scale=800:-1:flags=lanczos,palettegen=stats_mode=diff" palette.png
    !ffmpeg -v error -y -i {video} -i palette.png -lavfi "fps=15,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" {out}
    !ls -la {out}
""")

for source, scene, title, sentence in CLIPS:
    md(f"## {title}\n\n> {sentence}")
    code(f"%%writefile {source}\n" + (here / source).read_text())
    code(f'video = render("{source}", "{scene}")\ngif(video, "{Path(source).stem}.gif")')

notebook = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3",
                                                                                     "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
target = Path(__file__).with_name("animations.ipynb")
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(target)
