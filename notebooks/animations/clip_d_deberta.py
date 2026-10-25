# Clip D: "DeBERTa looks at what words are and where they are, so it needs a special kernel."
# Beats: ordinary attention scores every pair of words by content (one grid); DeBERTa adds content-to-position and
# position-to-content, whose patterns run along the diagonals because they depend on the distance between words; the
# three grids add up to one score; generic fast-attention kernels compute only the first, so the others were built
# separately; our kernel computes all three, tile by tile, in one pass.
# Numbers: Tesla T4, mDeBERTa-v3 encoder, batch 16 x 512 tokens: eager fp16 295 ms, Triton disentangled_flash 143 ms,
# glinert4 CUDA kernel 113 ms (kernel log rows 22 and 40).
from style import *

TOKENS = ["Lakshmi", "Varma", "signed", "the", "contract"]
C2C = [[1.0, .85, .25, .05, .2], [.85, 1.0, .3, .05, .25], [.3, .35, 1.0, .2, .8], [.05, .05, .25, 1.0, .45],
       [.2, .25, .85, .4, 1.0]]


def by_distance(weights, flip=False):
    n = len(TOKENS)
    return [[weights[(i - j if flip else j - i) + n - 1] for j in range(n)] for i in range(n)]


C2P = by_distance([.05, .1, .2, .7, .5, .95, .35, .15, .05])
P2C = by_distance([.05, .15, .3, .6, .45, .9, .4, .1, .05], flip=True)
TOTAL = [[min(1, (a + b + c) / 2.2) for a, b, c in zip(r1, r2, r3)] for r1, r2, r3 in zip(C2C, C2P, P2C)]


def labelled(grid, color, title, subtitle):
    head = words(title, size=22, color=color, weight=MEDIUM).next_to(grid, DOWN, buff=0.25)
    sub = words(subtitle, size=17, color=INK_DIM).next_to(head, DOWN, buff=0.08)
    return VGroup(head, sub)


class ThreeAttentions(Clip):
    def construct(self):
        heading = words("Ordinary attention asks one question", size=26, color=INK).to_corner(UL, buff=0.6)
        question = words("how related are these two words?", size=22, color=INK_DIM)
        question.next_to(heading, DOWN, buff=0.12, aligned_edge=LEFT)

        big = heat_grid(C2C, TERM_C2C, cell=0.62, gap=0.06).move_to(DOWN * 0.2 + RIGHT * 0.6)
        cols = VGroup(*(words(t, size=20, color=INK_DIM) for t in TOKENS))
        rows = VGroup(*(words(t, size=20, color=INK_DIM) for t in TOKENS))
        for k, (c, r) in enumerate(zip(cols, rows)):
            c.rotate(PI / 4).next_to(big[k], UP, buff=0.12).shift(RIGHT * 0.15)
            r.next_to(big[k * len(TOKENS)], LEFT, buff=0.2)
        self.beat(FadeIn(heading, shift=0.2 * DOWN), FadeIn(question, shift=0.2 * DOWN), run_time=0.8)
        self.beat(LaggedStart(*(FadeIn(t, shift=0.1 * DOWN) for t in cols), lag_ratio=0.08),
                  LaggedStart(*(FadeIn(t, shift=0.1 * RIGHT) for t in rows), lag_ratio=0.08), run_time=1.0)
        self.beat(LaggedStart(*(FadeIn(sq, scale=0.6) for sq in big), lag_ratio=0.02), run_time=1.6)
        pair = SurroundingRectangle(big[1], buff=0.03, corner_radius=0.04).set_stroke(INK, 2.5)
        pair_note = words("Lakshmi ↔ Varma: strongly related", size=20, color=INK)
        pair_note.next_to(big, DOWN, buff=0.35)
        self.beat(Create(pair), FadeIn(pair_note, shift=0.1 * UP), run_time=0.9)
        self.pause(0.8)

        grids = VGroup(heat_grid(C2C, TERM_C2C), heat_grid(C2P, TERM_C2P), heat_grid(P2C, TERM_P2C), heat_grid(TOTAL, INK))
        plus1, plus2, equals = (words(s, size=40, color=INK_DIM) for s in ("+", "+", "="))
        row = VGroup(grids[0], plus1, grids[1], plus2, grids[2], equals, grids[3]).arrange(RIGHT, buff=0.42)
        row.move_to(UP * 0.55)
        captions = VGroup(labelled(grids[0], TERM_C2C, "what ↔ what", "content to content"),
                          labelled(grids[1], TERM_C2P, "what → where", "content to position"),
                          labelled(grids[2], TERM_P2C, "where → what", "position to content"),
                          labelled(grids[3], INK, "the score", "summed, then softmax"))
        heading2 = words("DeBERTa asks three questions and adds the answers", size=26, color=INK)
        heading2.move_to(heading, aligned_edge=LEFT)
        question2 = words("what each word is, and where it sits relative to the other", size=22, color=INK_DIM)
        question2.move_to(question, aligned_edge=LEFT)
        self.beat(FadeOut(VGroup(cols, rows, pair, pair_note)), ReplacementTransform(big, grids[0]),
                  ReplacementTransform(heading, heading2), ReplacementTransform(question, question2), run_time=1.2)
        self.beat(FadeIn(captions[0], shift=0.1 * UP), run_time=0.5)

        self.beat(FadeIn(plus1), LaggedStart(*(FadeIn(sq, scale=0.6) for sq in grids[1]), lag_ratio=0.02),
                  FadeIn(captions[1], shift=0.1 * UP), run_time=1.3)
        diagonal = VGroup(*(grids[1][i * 5 + i + 1] for i in range(4)))
        stripe = VGroup(*(SurroundingRectangle(sq, buff=0.02, corner_radius=0.03).set_stroke(INK, 2) for sq in diagonal))
        stripe_note = words("same distance, same pattern: it follows the diagonals", size=20, color=TERM_C2P)
        stripe_note.next_to(row, UP, buff=0.35).align_to(grids[1], LEFT)
        self.beat(LaggedStart(*(Create(r) for r in stripe), lag_ratio=0.15), FadeIn(stripe_note, shift=0.1 * DOWN),
                  run_time=1.2)
        self.pause(0.6)
        self.beat(FadeOut(stripe), FadeOut(stripe_note), FadeIn(plus2),
                  LaggedStart(*(FadeIn(sq, scale=0.6) for sq in grids[2]), lag_ratio=0.02),
                  FadeIn(captions[2], shift=0.1 * UP), run_time=1.3)
        sources = VGroup(grids[0].copy(), grids[1].copy(), grids[2].copy())
        self.beat(FadeIn(equals), *(Transform(src, grids[3]) for src in sources), run_time=1.4)
        self.remove(*sources)
        self.add(grids[3])
        self.beat(FadeIn(captions[3], shift=0.1 * UP), run_time=0.5)
        self.pause(0.8)

        lower = row.get_bottom()[1] - 1.25
        generic = words("Generic fast attention (FlashAttention) builds only the first grid", size=22, color=INK_DIM)
        generic.move_to([0, lower, 0]).align_to(heading, LEFT)
        crosses = VGroup()
        for grid in grids[1:3]:
            crosses.add(VGroup(Line(grid.get_corner(UL), grid.get_corner(DR)), Line(grid.get_corner(UR), grid.get_corner(DL)))
                        .set_stroke(ERROR, 4))
        self.beat(FadeIn(generic, shift=0.1 * UP), LaggedStart(*(Create(c) for c in crosses), lag_ratio=0.3), run_time=1.1)
        self.pause(0.7)
        fused = SurroundingRectangle(VGroup(grids[0], grids[2]), buff=0.2, corner_radius=0.12).set_stroke(GPU, 3)
        ours = words("our kernel: all three terms, tile by tile, in one pass on the T4", size=22, color=GPU)
        ours.move_to(generic, aligned_edge=LEFT)
        self.beat(FadeOut(crosses), Create(fused), ReplacementTransform(generic, ours), run_time=1.2)
        self.pause(0.6)

        self.beat(FadeOut(VGroup(captions, ours)), VGroup(row, fused).animate.scale(0.7).to_edge(UP, buff=1.55),
                  run_time=1.0)
        chart_title = words("mDeBERTa encoder, batch 16 × 512 tokens, Tesla T4", size=20, color=INK_DIM)
        bars = VGroup()
        unit = 6.5 / 295
        for label, value, color in [("eager PyTorch fp16", 295, WASTE), ("Triton disentangled_flash", 143, INK_DIM),
                                    ("glinert4 CUDA kernel", 113, GPU)]:
            name = words(label, size=20, color=INK)
            bar = Rectangle(width=value * unit, height=0.36).set_fill(color, 0.9).set_stroke(width=0)
            number = words(f"{value} ms", size=20, color=NUMBER if color == GPU else INK, weight=BOLD, font=MONO)
            bars.add(VGroup(name, bar, number))
        bar_left, top = -1.6, -0.75
        for k, item in enumerate(bars):
            y = top - k * 0.58
            item[1].move_to([bar_left, y, 0], aligned_edge=LEFT)
            item[0].move_to([bar_left - 0.3, y, 0], aligned_edge=RIGHT)
            item[2].next_to(item[1], RIGHT, buff=0.2)
        chart_title.next_to(bars, UP, buff=0.3).align_to(bars, LEFT)
        self.beat(FadeIn(chart_title), LaggedStart(*(AnimationGroup(FadeIn(b[0]), GrowFromEdge(b[1], LEFT), FadeIn(b[2]))
                                                     for b in bars), lag_ratio=0.3), run_time=1.8)
        self.pause(0.6)

        claim = caption("DeBERTa looks at what words are and where they are, so it needs a special kernel.", size=23)
        self.beat(FadeIn(claim, shift=0.2 * UP), run_time=1.2)
        self.wait(HOLD)
        self.beat(*(FadeOut(mob) for mob in self.mobjects), run_time=1.0)
