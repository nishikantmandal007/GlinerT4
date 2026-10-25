# Clip D: "DeBERTa looks at what words are and where they are, so it needs a special kernel."
# Chapters: 1 ordinary attention (one worked pair: two word vectors, one dot product, one cell; then the whole grid);
# 2 where words are (the grid ignores order; DeBERTa measures the distance between two words, constant along each
# diagonal); 3 three questions (content-content, content-position, position-content, each a grid; they add up to the
# score); 4 a special kernel (fast generic kernels build only the first grid; eager PyTorch writes all three to memory;
# a fused kernel computes all three tile by tile).
# Numbers: Tesla T4, mDeBERTa-v3 encoder, batch 16 x 512 tokens: eager fp16 295 ms, Triton disentangled_flash 143 ms,
# glinert4 CUDA kernel 113 ms (kernel log rows 22 and 40). The grid values are illustrative.
from style import *

TOKENS = ["Lakshmi", "Varma", "signed", "the", "contract"]
N = len(TOKENS)
C2C = [[1.0, .85, .25, .05, .2], [.85, 1.0, .3, .05, .25], [.3, .35, 1.0, .2, .8], [.05, .05, .25, 1.0, .45],
       [.2, .25, .85, .4, 1.0]]
VECTORS = {"Lakshmi": [.9, .2, .7, .1, .8, .3], "Varma": [.8, .3, .6, .2, .9, .2]}


def by_distance(weights, flip=False):
    return [[weights[(i - j if flip else j - i) + N - 1] for j in range(N)] for i in range(N)]


C2P = by_distance([.05, .1, .2, .7, .5, .95, .35, .15, .05])
P2C = by_distance([.05, .15, .3, .6, .45, .9, .4, .1, .05], flip=True)
TOTAL = [[min(1, (a + b + c) / 2.2) for a, b, c in zip(r1, r2, r3)] for r1, r2, r3 in zip(C2C, C2P, P2C)]


def vector(values, color, cell=0.34):
    return VGroup(*(Square(cell).set_stroke(color, 1).set_fill(color, 0.1 + 0.85 * v) for v in values)).arrange(
        DOWN, buff=0.04)


def cell(grid, i, j):
    return grid[i * N + j]


class ThreeAttentions(Clip):
    def construct(self):
        self.ordinary()
        self.where_words_are()
        self.three_questions()
        self.special_kernel()
        self.finale("DeBERTa looks at what words are and where they are, so it needs a special kernel.",
                    "Tesla T4 · mDeBERTa-v3 encoder · batch 16 × 512 tokens", size=26)

    # 1 ------------------------------------------------------------------------------------------------------------
    def ordinary(self):
        self.chapter(1, 4, "Ordinary attention")
        sentence = words(" ".join(TOKENS), size=36).move_to(UP * 1.6)
        self.say("A sentence. To understand it, each word looks at every other word.",
                 LaggedStart(*(FadeIn(ch, shift=0.1 * UP) for ch in sentence), lag_ratio=0.02))
        self.read()

        grid = heat_grid(C2C, TERM_C2C, cell=0.5, gap=0.05).move_to(LEFT * 2.6 + DOWN * 0.35)
        for sq in grid:
            sq.set_fill(TERM_C2C, 0.04).set_stroke(PANEL_EDGE, 1)
        cols = VGroup(*(words(t, size=18, color=INK_DIM) for t in TOKENS))
        rows = VGroup(*(words(t, size=18, color=INK_DIM) for t in TOKENS))
        for k, (c, r) in enumerate(zip(cols, rows)):
            c.rotate(PI / 4).move_to(cell(grid, 0, k).get_top() + UP * 0.08 + LEFT * 0.1, aligned_edge=DL)
            r.next_to(cell(grid, k, 0), LEFT, buff=0.2)
        pieces = [span(sentence, " ".join(TOKENS), t) for t in TOKENS]
        self.say("Lay the words out twice, as rows and as columns. Every pair of words gets one cell.",
                 FadeOut(sentence), *(Derive(p, c) for p, c in zip(pieces, cols)),
                 *(Derive(p.copy(), r) for p, r in zip(pieces, rows)), FadeIn(grid), run_time=2.4)
        self.read()

        target = cell(grid, 0, 1)
        focus = SurroundingRectangle(target, buff=0.03, corner_radius=0.04).set_stroke(INK, 2.5)
        dim = [sq.animate.set_stroke(opacity=0.35) for sq in grid if sq is not target]
        self.say("Take one cell: the row word Lakshmi, looking at the column word Varma.",
                 Create(focus), rows[0].animate.set_color(INK), cols[1].animate.set_color(INK),
                 *(r.animate.set_opacity(0.4) for r in rows[1:]), *(c.animate.set_opacity(0.4) for c in cols if c is not cols[1]),
                 *dim)
        self.read()

        left = vector(VECTORS["Lakshmi"], TERM_C2C)
        right = vector(VECTORS["Varma"], TERM_C2C)
        dot = words("·", size=48, color=INK_DIM)
        eq = words("=", size=36, color=INK_DIM)
        score = ValueTracker(0)
        value = counter(score, size=40, color=NUMBER, fmt="{:.2f}")
        example = VGroup(left, dot, right, eq).arrange(RIGHT, buff=0.35).move_to(RIGHT * 3.3 + DOWN * 0.35)
        value.add_updater(lambda m: m.next_to(eq, RIGHT, buff=0.3))
        value.update()
        names = VGroup(words("Lakshmi", size=18, color=INK).next_to(left, UP, buff=0.15),
                       words("Varma", size=18, color=INK).next_to(right, UP, buff=0.15))
        self.say("Each word is stored as a vector: a list of numbers describing what it means.",
                 Derive(rows[0].copy(), names[0]), Derive(cols[1].copy(), names[1]),
                 LaggedStart(GrowFromEdge(left, UP), GrowFromEdge(right, UP), lag_ratio=0.3))
        self.read()
        self.add(value)
        self.say("Multiply the two vectors entry by entry and add up: a high score means the words belong together.",
                 FadeIn(dot), FadeIn(eq), score.animate.set_value(0.85), run_time=2.4)
        self.read()
        link = connector(target, VGroup(left, right), color=INK_DIM)
        self.say("That score colours the cell. Bright means strongly related.",
                 target.animate.set_fill(TERM_C2C, 0.08 + 0.85 * 0.85))
        self.read()

        self.say("Do this for every pair and the whole grid fills in. That is ordinary attention.",
                 FadeOut(VGroup(example, names, focus, link)), FadeOut(value),
                 *(r.animate.set_opacity(1).set_color(INK_DIM) for r in rows),
                 *(c.animate.set_opacity(1).set_color(INK_DIM) for c in cols),
                 LaggedStart(*(sq.animate.set_fill(TERM_C2C, 0.08 + 0.85 * C2C[k // N][k % N]).set_stroke(width=0)
                               for k, sq in enumerate(grid)), lag_ratio=0.03), run_time=2.6)
        self.read()
        self.grid, self.cols, self.rows = grid, cols, rows

    # 2 ------------------------------------------------------------------------------------------------------------
    def where_words_are(self):
        self.chapter(2, 4, "Where words are")
        grid = self.grid
        swapped = VGroup(words("Lakshmi Varma", size=24, color=INK), words("Varma Lakshmi", size=24, color=INK))
        swapped.arrange(DOWN, buff=0.3).move_to(RIGHT * 3.6 + UP * 0.6)
        same = words("same scores", size=20, color=ERROR).next_to(swapped, DOWN, buff=0.3)
        self.say("But the score only depends on what the two words are, not on where they are.",
                 FadeIn(swapped[0], shift=0.1 * UP))
        self.read()
        self.say("Swap the words around, and ordinary attention sees exactly the same pairs.",
                 Derive(swapped[0].copy(), swapped[1]), FadeIn(same, shift=0.1 * UP))
        self.read()

        numbers = VGroup()
        for i in range(N):
            for j in range(N):
                numbers.add(words(f"{j - i:+d}" if j != i else "0", size=18, color=TERM_C2P, font=MONO)
                            .move_to(cell(grid, i, j)))
        board = VGroup(*(sq.copy().set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1) for sq in grid))
        self.say("DeBERTa also measures where: how far the column word is from the row word.",
                 FadeOut(VGroup(swapped, same)), FadeIn(board), LaggedStart(*(FadeIn(n) for n in numbers), lag_ratio=0.03),
                 run_time=2.6)
        self.read()

        diagonal = VGroup(*(cell(grid, i, i + 1) for i in range(N - 1)))
        stripe = VGroup(*(SurroundingRectangle(sq, buff=0.02, corner_radius=0.03).set_stroke(INK, 2) for sq in diagonal))
        note = words("+1: the next word", size=20, color=TERM_C2P).move_to(RIGHT * 3.6 + UP * 0.6)
        self.say("Every cell on this diagonal holds the same distance: the next word along.",
                 LaggedStart(*(Create(r) for r in stripe), lag_ratio=0.15), FadeIn(note, shift=0.1 * UP))
        self.read()
        note2 = words("−1: the word before", size=20, color=TERM_C2P).next_to(note, DOWN, buff=0.2).align_to(note, LEFT)
        back = cell(grid, 1, 0)
        mark = SurroundingRectangle(back, buff=0.02, corner_radius=0.03).set_stroke(INK, 2)
        self.say("And now order matters: Varma is +1 from Lakshmi, Lakshmi is −1 from Varma.",
                 Create(mark), FadeIn(note2, shift=0.1 * UP))
        self.read()
        self.say("Each distance has its own learned vector, just like each word.",
                 FadeOut(VGroup(stripe, mark, note, note2)))
        self.read()
        self.board, self.numbers = board, numbers

    # 3 ------------------------------------------------------------------------------------------------------------
    def three_questions(self):
        self.chapter(3, 4, "Three questions")
        grids = VGroup(heat_grid(C2C, TERM_C2C), heat_grid(C2P, TERM_C2P), heat_grid(P2C, TERM_P2C), heat_grid(TOTAL, INK))
        plus1, plus2, equals = (words(s, size=40, color=INK_DIM) for s in ("+", "+", "="))
        row = VGroup(grids[0], plus1, grids[1], plus2, grids[2], equals, grids[3]).arrange(RIGHT, buff=0.42)
        row.move_to(UP * 0.55)
        heads = [("what ↔ what", "content to content", TERM_C2C), ("what → where", "content to position", TERM_C2P),
                 ("where → what", "position to content", TERM_P2C), ("the score", "summed, then softmax", INK)]
        captions = VGroup()
        for grid, (head, sub, color) in zip(grids, heads):
            h = words(head, size=21, color=color, weight=MEDIUM).next_to(grid, DOWN, buff=0.25)
            captions.add(VGroup(h, words(sub, size=16, color=INK_DIM).next_to(h, DOWN, buff=0.08)))

        distances = VGroup(self.board, self.numbers)
        ghosts = [distances.copy().scale(g.width / self.board.width).move_to(g) for g in grids[1:3]]
        self.say("Question one is the grid we already have: how related are the two words' meanings?",
                 FadeOut(distances), FadeOut(VGroup(self.cols, self.rows)), ReplacementTransform(self.grid, grids[0]),
                 FadeIn(captions[0], shift=0.1 * UP), run_time=2.4)
        self.read()

        ghost = ghosts[0]
        self.say("Question two: how does this word relate to that distance? Lakshmi asks: what usually sits one word after me?",
                 FadeIn(plus1), FadeIn(ghost), run_time=1.6)
        self.beat(FadeOut(ghost[1]), ReplacementTransform(ghost[0], grids[1]), FadeIn(captions[1], shift=0.1 * UP),
                  run_time=2.0)
        self.read()
        diagonal = VGroup(*(cell(grids[1], i, i + 1) for i in range(N - 1)))
        stripe = VGroup(*(SurroundingRectangle(sq, buff=0.02, corner_radius=0.03).set_stroke(INK, 2) for sq in diagonal))
        self.say("Because it is built from distances, its pattern runs along the diagonals.",
                 LaggedStart(*(Create(r) for r in stripe), lag_ratio=0.15))
        self.read()

        ghost = ghosts[1]
        self.say("Question three is the mirror image: how does that distance relate to the other word?",
                 FadeOut(stripe), FadeIn(plus2), FadeIn(ghost), run_time=1.6)
        self.beat(FadeOut(ghost[1]), ReplacementTransform(ghost[0], grids[2]), FadeIn(captions[2], shift=0.1 * UP),
                  run_time=2.0)
        self.read()

        sources = VGroup(grids[0].copy(), grids[1].copy(), grids[2].copy())
        self.say("Add the three grids cell by cell. That sum is the attention score DeBERTa uses.",
                 FadeIn(equals), *(Transform(src, grids[3]) for src in sources), FadeIn(captions[3], shift=0.1 * UP),
                 run_time=2.6)
        self.remove(*sources)
        self.add(grids[3])
        self.read(extra=0.5)
        self.row, self.grids, self.captions = row, grids, captions

    # 4 ------------------------------------------------------------------------------------------------------------
    def special_kernel(self):
        self.chapter(4, 4, "Why a special kernel")
        grids = self.grids
        crosses = VGroup(*(VGroup(Line(g.get_corner(UL), g.get_corner(DR)), Line(g.get_corner(UR), g.get_corner(DL)))
                           .set_stroke(ERROR, 4) for g in grids[1:3]))
        self.say("Fast generic attention kernels, like FlashAttention, only know how to build the first grid.",
                 LaggedStart(*(Create(c) for c in crosses), lag_ratio=0.3))
        self.read()
        fused = SurroundingRectangle(VGroup(grids[0], grids[2]), buff=0.2, corner_radius=0.12).set_stroke(GPU, 3)
        self.say("So DeBERTa needs its own kernel: one that computes all three, piece by piece, without writing them out.",
                 FadeOut(crosses), Create(fused), run_time=2.2)
        self.read()

        self.beat(FadeOut(self.captions), VGroup(self.row, fused).animate.scale(0.62).to_edge(UP, buff=1.2), run_time=1.4)
        chart_title = words("whole encoder, batch 16 × 512 tokens, Tesla T4", size=20, color=INK_DIM)
        bars = VGroup()
        unit = 6.0 / 295
        rows = [("eager PyTorch fp16", 295, WASTE), ("Triton disentangled_flash", 143, INK_DIM),
                ("glinert4 CUDA kernel", 113, GPU)]
        for label, value, color in rows:
            name = words(label, size=20, color=INK)
            bar = Rectangle(width=value * unit, height=0.36).set_fill(color, 0.9).set_stroke(width=0)
            number = words(f"{value} ms", size=20, color=NUMBER if color == GPU else INK, weight=BOLD, font=MONO)
            bars.add(VGroup(name, bar, number))
        bar_left, top = -1.4, -0.45
        for k, item in enumerate(bars):
            y = top - k * 0.62
            item[1].move_to([bar_left, y, 0], aligned_edge=LEFT)
            item[0].move_to([bar_left - 0.3, y, 0], aligned_edge=RIGHT)
            item[2].next_to(item[1], RIGHT, buff=0.2)
        chart_title.next_to(bars, UP, buff=0.3).align_to(bars, LEFT)
        texts = ["Plain PyTorch writes all three grids to GPU memory and reads them back: 295 ms for the encoder.",
                 "A Triton kernel that fuses the three terms halves that: 143 ms.",
                 "Our CUDA kernel, written for the T4's tensor cores: 113 ms."]
        for k, item in enumerate(bars):
            extra = [FadeIn(chart_title)] if k == 0 else []
            self.say(texts[k], *extra, FadeIn(item[0]), GrowFromEdge(item[1], LEFT), FadeIn(item[2]), run_time=2.0)
            self.read()
