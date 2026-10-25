# The visual system for explainer clips (Manim Community Edition). Every clip imports from here and never defines its
# own colours, fonts or timing. Copy this file next to the clips; see ../SKILL.md for how to use it.
from manim import *

# Surfaces and ink
BACKGROUND = "#1C1E21"
PANEL = "#26292E"
PANEL_EDGE = "#3A3F45"
INK = "#E8E6E1"
INK_DIM = "#8B9096"
INK_FAINT = "#4A4F55"

# Colours with fixed meanings: never reuse one for something else within a project
CPU = "#58C4DD"
GPU = "#FF9F43"
WASTE = "#6B7178"
CORRECT = "#83C167"
ERROR = "#FC6255"
NUMBER = "#FFD84D"
TERM_C2C = "#5CD0B3"
TERM_C2P = "#C792EA"
TERM_P2C = "#FF8FB1"
LABEL_COLORS = {"Person": "#58C4DD", "Organization": "#C792EA", "Street": "#83C167", "City": "#FF9F43"}

SANS = "Inter"
MONO = "JetBrains Mono"

# Timing, measured from 3b1b's 2024 transformer scenes: median animation 2 s, a pause after most animations
BEAT = 2.0
PAUSE = 1.0
HOLD = 2.5
READ_WORDS_PER_SECOND = 3.0

config.background_color = BACKGROUND
config.frame_rate = 30


def words(text, size=28, color=INK, weight=NORMAL, font=SANS):
    return Text(text, font=font, font_size=size, color=color, weight=weight)


def mono(text, size=26, color=INK):
    return Text(text, font=MONO, font_size=size, color=color)


def wrap(text, line_len=72):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > line_len:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return "\n".join(lines + [line])


def span(text_mob, source, phrase, occurrence=0):
    # the glyphs of `phrase` inside a Text built from `source` (Text keeps no glyphs for whitespace)
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(phrase, start + 1)
    first = len("".join(source[:start].split()))
    return text_mob[first:first + len("".join(phrase.split()))]


def box(mob, color=INK_DIM, fill=0.18, buff=0.07, stroke=2):
    rect = SurroundingRectangle(mob, buff=buff, corner_radius=0.06)
    rect.set_stroke(color, stroke).set_fill(color, fill)
    return rect


def panel(width, height, fill=PANEL, edge=PANEL_EDGE):
    return RoundedRectangle(width=width, height=height, corner_radius=0.18).set_fill(fill, 1).set_stroke(edge, 1.5)


def chip(label, color, size=22):
    text = words(label, size=size, color=color, weight=MEDIUM)
    back = RoundedRectangle(width=text.width + 0.36, height=text.height + 0.26, corner_radius=0.14)
    back.set_fill(color, 0.14).set_stroke(color, 1.5)
    return VGroup(back, text.move_to(back))


def chip_level(group, level):
    chips = [group] if isinstance(group[0], RoundedRectangle) else list(group)
    return AnimationGroup(*(AnimationGroup(c[0].animate.set_fill(opacity=0.14 * level).set_stroke(opacity=level),
                                           c[1].animate.set_opacity(level)) for c in chips))


def caption(sentence, size=34):
    line = words(sentence, size=size, color=INK, weight=MEDIUM)
    if line.width > config.frame_width - 1.2:
        line.scale_to_fit_width(config.frame_width - 1.2)
    return line.to_edge(DOWN, buff=0.55)


def bracket(left, right, y, color=INK_DIM, tick=0.12, up=True, stroke=2.5):
    sign = -1 if up else 1
    left, right = np.array([left, y, 0]), np.array([right, y, 0])
    return VGroup(Line(left, right), Line(left, left + sign * tick * UP), Line(right, right + sign * tick * UP)).set_stroke(
        color, stroke)


def hatch(rect, spacing=0.14, color=WASTE, stroke=1.6):
    x0, x1 = rect.get_left()[0], rect.get_right()[0]
    y0, y1 = rect.get_bottom()[1], rect.get_top()[1]
    lines = VGroup()
    offset = x0 - (y1 - y0)
    while offset < x1:
        start = np.array([offset, y0, 0])
        end = start + (y1 - y0) * np.array([1, 1, 0])
        if start[0] < x0:
            start = start + (x0 - start[0]) * np.array([1, 1, 0])
        if end[0] > x1:
            end = end - (end[0] - x1) * np.array([1, 1, 0])
        if end[0] > start[0]:
            lines.add(Line(start, end))
        offset += spacing
    return lines.set_stroke(color, stroke)


def counter(tracker, size=60, color=NUMBER, fmt="{:.0f}"):
    # animate only the tracker; never animate the counter itself (it is rebuilt every frame)
    paint = color if callable(color) else (lambda: color)
    return always_redraw(lambda: words(fmt.format(tracker.get_value()), size=size, color=paint(), weight=BOLD,
                                       font=MONO))


def heat_grid(values, color, cell=0.42, gap=0.04):
    grid = VGroup()
    rows, cols = len(values), len(values[0])
    for i in range(rows):
        for j in range(cols):
            square = Square(cell).set_stroke(width=0).set_fill(color, 0.08 + 0.85 * values[i][j])
            square.move_to([(j - (cols - 1) / 2) * (cell + gap), -(i - (rows - 1) / 2) * (cell + gap), 0])
            grid.add(square)
    return grid


def connector(small, big, color=INK_DIM, stroke=1.5):
    # 3b1b's link between a symbol and its concrete expansion: two lines from the small box's corners to the big box's
    if small.get_center()[1] > big.get_center()[1]:
        pairs = [(small.get_corner(DL), big.get_corner(UL)), (small.get_corner(DR), big.get_corner(UR))]
    else:
        pairs = [(small.get_corner(UL), big.get_corner(DL)), (small.get_corner(UR), big.get_corner(DR))]
    return VGroup(*(Line(a, b) for a, b in pairs)).set_stroke(color, stroke)


class Derive(Transform):
    """TransformFromCopy that ends with `target` itself on screen. Transform pads the target's submobjects to match the
    source, which breaks glyph indexing (span) on Text targets; transforming copies and swapping in the original keeps
    the target intact."""

    def __init__(self, source, target, **kwargs):
        self.final = target
        super().__init__(source.copy(), target.copy(), **kwargs)

    def clean_up_from_scene(self, scene):
        super().clean_up_from_scene(scene)
        scene.remove(self.mobject)
        scene.add(self.final)


class Clip(Scene):
    """A silent explainer: chapters top left, a narration line at the bottom that the viewer reads."""

    def setup(self):
        self.narration = None
        self.narration_start = 0.0
        self.chapter_title = None

    def beat(self, *animations, run_time=BEAT, **kwargs):
        self.play(*animations, run_time=run_time, rate_func=kwargs.pop("rate_func", smooth), **kwargs)

    def pause(self, seconds=PAUSE):
        self.wait(seconds)

    def chapter(self, number, total, title):
        label = VGroup(words(f"{number} / {total}", size=18, color=INK_FAINT, font=MONO),
                       words(title, size=26, color=INK, weight=MEDIUM)).arrange(RIGHT, buff=0.3)
        label.to_corner(UL, buff=0.55)
        if self.chapter_title is None:
            self.play(FadeIn(label, shift=0.15 * DOWN), run_time=0.8)
        else:
            self.play(FadeOut(self.chapter_title, shift=0.15 * UP), FadeIn(label, shift=0.15 * UP), run_time=0.8)
        self.chapter_title = label

    def say(self, text, *animations, run_time=None):
        """Replace the narration line, together with any animations that illustrate it."""
        line = words(wrap(text), size=25, color=INK).to_edge(DOWN, buff=0.45)
        if self.narration is not None:
            self.play(FadeOut(self.narration, shift=0.1 * UP), run_time=0.35)
        self.play(FadeIn(line, shift=0.12 * UP), *animations,
                  run_time=run_time or (BEAT if animations else 0.8), rate_func=smooth)
        self.narration = line
        self.narration_start = self.renderer.time - (run_time or (BEAT if animations else 0.8))

    def read(self, extra=0.0):
        """Wait until the current narration line has been on screen long enough to read, plus a pause."""
        if self.narration is None:
            self.wait(PAUSE + extra)
            return
        needed = len(self.narration.text.split()) / READ_WORDS_PER_SECOND + 0.6
        elapsed = self.renderer.time - self.narration_start
        self.wait(max(PAUSE * 0.6, needed - elapsed) + extra)

    def unsay(self):
        if self.narration is not None:
            self.play(FadeOut(self.narration), run_time=0.5)
            self.narration = None

    def finale(self, sentence, note=None, size=34):
        self.unsay()
        claim = caption(sentence, size=size)
        parts = [FadeIn(claim, shift=0.2 * UP)]
        if note:
            parts.append(FadeIn(words(note, size=20, color=INK_DIM).next_to(claim, DOWN, buff=0.12)))
        self.play(*parts, run_time=1.2)
        self.wait(HOLD)
        self.play(*(FadeOut(mob) for mob in self.mobjects), run_time=1.0)
