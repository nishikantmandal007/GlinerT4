# The visual system shared by every clip: one palette with fixed meanings, two typefaces, one background, one motion
# vocabulary. Clips import from here and never define their own colours or fonts.
from manim import *

BACKGROUND = "#1C1E21"
PANEL = "#26292E"
PANEL_EDGE = "#3A3F45"
INK = "#E8E6E1"
INK_DIM = "#8B9096"
INK_FAINT = "#4A4F55"

CPU = "#58C4DD"
GPU = "#FF9F43"
WASTE = "#6B7178"
CORRECT = "#83C167"
ERROR = "#FC6255"
NUMBER = "#FFD84D"

LABEL_COLORS = {"Person": "#58C4DD", "Organization": "#C792EA", "Street": "#83C167", "City": "#FF9F43"}

SANS = "Inter"
MONO = "JetBrains Mono"

BEAT = 1.0
HOLD = 2.0

config.background_color = BACKGROUND
config.frame_rate = 30


def words(text, size=28, color=INK, weight=NORMAL, font=SANS):
    return Text(text, font=font, font_size=size, color=color, weight=weight)


def mono(text, size=26, color=INK):
    return Text(text, font=MONO, font_size=size, color=color)


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
    line.to_edge(DOWN, buff=0.55)
    return line


class Clip(Scene):
    def beat(self, *animations, run_time=BEAT, **kwargs):
        self.play(*animations, run_time=run_time, rate_func=smooth, **kwargs)

    def pause(self, seconds=0.5):
        self.wait(seconds)


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
    paint = color if callable(color) else (lambda: color)
    return always_redraw(lambda: words(fmt.format(tracker.get_value()), size=size, color=paint(), weight=BOLD,
                                       font=MONO))


TERM_C2C = "#5CD0B3"
TERM_C2P = "#C792EA"
TERM_P2C = "#FF8FB1"


def heat_grid(values, color, cell=0.42, gap=0.04):
    grid = VGroup()
    rows, cols = len(values), len(values[0])
    for i in range(rows):
        for j in range(cols):
            square = Square(cell).set_stroke(width=0).set_fill(color, 0.08 + 0.85 * values[i][j])
            square.move_to([(j - (cols - 1) / 2) * (cell + gap), -(i - (rows - 1) / 2) * (cell + gap), 0])
            grid.add(square)
    return grid
