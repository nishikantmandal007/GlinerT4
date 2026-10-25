# Manim CE pitfalls and layout traps (each one hit while making the GLiNER2-on-T4 clips)

## Errors

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: zip() argument 2 is shorter than argument 1` inside `interpolate_mobject` | Fading, recolouring or setting the opacity of an `always_redraw` mobject (a counter): its glyph count changes between the start copy and the target ("0" → "59") | Animate only the `ValueTracker`; keep colour in a variable the redraw reads; call `.update()` before fading in neighbours that follow it |
| Same zip error with `Restore` | `save_state()` taken before the group's children changed | Animate explicit targets (`chip_level`, `.animate.set_fill(...)`) instead of `Restore` |
| Highlights land on the wrong letters after a `TransformFromCopy` into a `Text`; transformed text panels look empty | `Transform` aligns submobjects by padding the *target* with duplicates, so the target's glyph indices no longer match characters | Use `Derive(source, target)` from `style.py`: it transforms copies and swaps the untouched target in at the end |
| Two narration lines overlap mid-transition | Old and new line cross-faded in the same spot | `Clip.say()` fades the old line out first (0.35 s), then fades the new one in |
| `TypeError: unsupported operand ... NoneType` | `.set(width=None)` or similar left in a chain | Remove it; size with `scale_to_fit_width` |
| `ModuleNotFoundError: manim` on Python 3.13 | ManimPango has no wheel for 3.13 | Use Python 3.12 and install `libpango1.0-dev libcairo2-dev` |
| A render "succeeds" but frames look unchanged | The render failed and the old MP4 is still there (or the manim path was wrong) | Check for `Error` in the output and the MP4's timestamp before reviewing; `scripts/preview.sh` stops on errors |

## Layout

- Long lines run off the frame at 1080p exactly as at 480p: check the left and right edges of every contact-sheet frame.
- Text glyph indexing skips whitespace: use `span(text, source, phrase)` to box a phrase.
- A chip at opacity 1 gets a solid background: dim/undim fill (×0.14), stroke and text separately.
- Labels placed relative to an object that moves later do not follow; position after layout, or use updaters.
- Moving words apart inside a sentence collides with neighbours; draw a cut line and box the halves instead.
- Narration needs the bottom 1.3 units; keep content above it.
- A shrunken "recap" of a busy diagram becomes unreadable; fade it out and give the next idea a clean board.
- Contact sheets from `f_*.png` sort lexicographically (f_10 before f_2); number frames with `%d` and tile in order.

## Pacing smells

- Beats around 1 s, no pauses: the viewer cannot read the narration and watch the motion. Use `BEAT` = 2 s and `read()`.
- Several things moving at once with no stagger: the eye does not know where to go.
- A number appearing at its final value: count it.
- A concept introduced only by a label: derive it on screen from something already there.
