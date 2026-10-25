---
name: explainer-animations
description: Make 3Blue1Brown-style explainer animations with Manim Community Edition — silent, self-explanatory clips that teach one idea through gradual visual build-up, a narration line, measured numbers and a final one-sentence claim. Use when asked for an explainer video, animated diagram, technical animation, blog/post animation, Manim scene, or "animations like 3Blue1Brown / Grant Sanderson".
---

# Explainer animations, the 3Blue1Brown way

A clip teaches **one idea** to someone who has never seen the project and cannot hear a voiceover. It succeeds only if a
viewer can say the clip's sentence back after watching it once. Everything below serves that test.

This method comes from reading the code of 3b1b's neural-network (2017) and transformer (2024) videos
(`github.com/3b1b/videos`, `_2017/nn`, `_2024/transformers`). The measured numbers and the techniques, with the scenes
they come from, are in `references/3b1b-patterns.md`. That repository is CC BY-NC-SA: study it, never copy its code.

## 1. Before any code: script the idea

Write `references/storyboard-template.md` out for the clip. You need, in this order:

1. **The sentence.** One plain sentence the viewer should be able to repeat. If you cannot write it, there is no clip yet.
2. **The concrete example.** A real sentence, a real document line, real numbers from a real measurement. 3b1b opens with
   a handwritten 9, a real prompt ("The Computer History Museum is located in Mountain"), never with abstraction.
3. **Chapters (3–5)** that each answer one question the viewer would ask next ("what is this?", "what goes wrong?", "how
   is it fixed?", "how much does it help?").
4. **Beats (4–8 per chapter).** For each beat: the narration line the viewer reads, what moves, and what it is derived
   from. A beat that introduces something with nothing on screen to derive it from is a warning sign.
5. **Numbers and their source.** Every number on screen comes from a measurement you can name (file, run, row). Round
   honestly and state the setup in a footnote. If a figure is illustrative (a timeline's block lengths), the measured
   values must still be the ones printed.

Length follows content, not a target: a detailed concept clip is 60–120 s. 30 s is a teaser, not an explanation.

## 2. The ten rules (each one observed in 3b1b's code)

1. **Concrete first, abstraction second.** Show the real thing, then turn it into the diagram (`ReplacementTransform`
   from pixels to neurons, from words to boxes). The viewer watches the abstraction being made.
2. **Nothing appears from nowhere.** New objects grow out of existing ones: `TransformFromCopy`, a copy flying into
   place, a brace label copied into an equation. 3b1b uses `TransformFromCopy` 106 times in four files.
3. **One new thing per beat, and only one thing moving.** If two things must change, stagger them (`LaggedStart`,
   `lag_ratio` 0.1 for many small parts, 0.3–0.5 for a few large ones).
4. **Slow down.** Default animation 2 s (3b1b: median 2.0 s, mean 3.4 s, 87% ≥ 2 s), a pause after most animations
   (0.62 pauses per animation), 4–6 s for the key moment. Use `BEAT`, `PAUSE`, `read()` from `assets/style.py`.
5. **Focus by dimming, not by flashing.** Fade the rest to 25–40% (`set_opacity`, `fade`) and keep history visible but
   dim. 3b1b uses `Indicate` zero times in these files; do not use it either.
6. **Link symbol and substance.** A small box on the symbol, a big box on its concrete form, two connector lines between
   them (`connector()`); then rearrange the concrete form to compute with it.
7. **One worked example before the general case.** Enlarge one neuron, one word, one tile; change its value and show what
   the number means; only then show all of them.
8. **Values as colour, consistently.** One colour = one meaning for the whole project (the palette in `style.py`).
   Magnitude by brightness of one hue; sign by hue (blue/red). Never cycle colours for variety.
9. **Count with what is on screen.** Braces with counts, then copies of the counts flying into the equation (28 × 28 =
   784). Numbers count up with a `ValueTracker` (`counter()`), never by appearing.
10. **End on the claim.** Clear the board, show the sentence with its source footnote, hold 2–3 s, fade to the background
    (seamless loop).

## 3. Silent clips need a narration track

3b1b's visuals lean on his voice. A silent clip must put the voiceover on screen:

- `self.chapter(n, total, title)` — chapter label top left ("2 / 4  Pre-cut chunks").
- `self.say(text, *animations)` — the narration line at the bottom, replaced beat by beat, played **with** the animation
  it describes. Write it as speech: short, concrete, present tense, one idea ("Window 2 starts 64 words earlier, so it
  sees the whole name."). Max ~20 words; wraps to two lines.
- `self.read()` — waits until the line has been on screen long enough to read (3 words/s + 0.6 s), then a short pause.
- Reserve the bottom 1.3 units of the frame for narration; never place content there.
- On-screen labels name things ("chunk 1", "GPU idle"); the narration explains why. Do not repeat the narration in labels.

## 4. Workflow

1. Copy `assets/style.py` next to the clips; subclass `Clip`.
2. Storyboard (section 1), then write `construct()` as a list of chapter methods, like 3b1b's
   `break_up_image_as_neurons()`, `show_activation_of_one_neuron()`, ... One method per chapter.
3. Preview at 480p: `scripts/preview.sh <file.py> <Scene> 3x3 <t1> ... <t9>` renders and writes a contact sheet of the
   given timestamps. Look at it. Every review round, check: text inside the frame, no overlaps, nothing tiny, the active
   element obvious, the narration matches what moves.
4. Fix, re-render, repeat until a frame-by-frame look finds nothing. Expect 3–6 rounds per clip.
5. Final: `scripts/export.sh <file.py> <Scene> <out_dir>` renders 1080p and a 640 px, 12 fps GIF (about 6–8 MB for a 70–90 s clip).
6. Last test: show it to someone (or reason as someone) unfamiliar with the project. Can they say the sentence back?
   If not, the gap is almost always a missing beat, not a missing label.

## 5. Mistakes that cost time (details and fixes in `references/pitfalls.md`)

- Animating an `always_redraw` mobject (fade, opacity, colour): glyph counts change, `zip() argument 2 is shorter`.
  Animate its tracker only; call `.update()` before fading in labels next to it.
- `save_state`/`Restore` on groups whose children changed: same zip error. Animate explicit targets instead.
- `TransformFromCopy(a, text)` pads `text` with extra glyphs, so later `span()` highlights hit the wrong letters. Use
  `Derive(a, text)` from `style.py`, which leaves the target intact.
- Setting a chip's opacity to 1 also makes its translucent background solid; dim fill, stroke and text separately.
- Text running off the frame: check every frame of the contact sheet; long lines need a smaller size or wrapping.
- Moving words apart inside a sentence makes them collide with neighbours; show a split with a cut line instead.
- A render that fails leaves the previous video in place; check the render's exit and timestamp before reviewing frames.

## 6. Files

- `assets/style.py` — palette, typefaces, timing constants, `Clip` (chapters, narration, finale) and helpers (`Derive`, `span`,
  `box`, `chip`, `bracket`, `hatch`, `counter`, `heat_grid`, `connector`).
- `references/3b1b-patterns.md` — what 3b1b's code does, measured, with scene names to read.
- `references/pitfalls.md` — Manim CE errors and layout traps, with fixes.
- `references/storyboard-template.md` — fill this in before writing a scene.
- `scripts/preview.sh`, `scripts/export.sh` — render and review loop.

Setup: `pip install manim` (Python 3.12; needs `libcairo2-dev libpango1.0-dev ffmpeg`), fonts Inter and JetBrains Mono
(`fonts-inter fonts-jetbrains-mono`, or the Google Fonts TTFs). No LaTeX needed: use `Text`, and Unicode for symbols.
