# Clip E: "The kernel re-packs the work into the exact shape the T4's fast units want."
# Chapters: 1 a tensor core (one worked output cell: a row times a column; one instruction does the whole 16x8 tile);
# 2 where the data lives (GPU memory: large and far; on-chip memory: small and fast; tensor cores need tiles close by);
# 3 before (the Triton kernel read Q, K and the position tables in scattered pieces, so the tensor cores idled);
# 4 now (one wide read per tile into on-chip memory, only the distances a tile needs, the next tile prefetched into
# registers by hand because the T4 has no asynchronous copy; tiles stream into the tensor cores).
# Numbers: Tesla T4, 16 x 12 heads x 512 tokens, per layer (kernel log rows 24, 25, 31, 38): Triton kernel 4.80 ms +
# position tables 2.2 ms = 7.0 ms; ours 2.84 ms; Triton disentangled kernel 2.7 TFLOP/s vs plain attention (SDPA)
# 17.0 TFLOP/s. T4 spec: 65 TFLOP/s fp16 tensor-core peak, 320 GB/s memory, 64 KB on-chip memory per SM.
from style import *

A_VALUES = [[((r * 5 + c * 3) % 7) / 6 for c in range(8)] for r in range(16)]
B_VALUES = [[((r * 3 + c * 5) % 7) / 6 for c in range(8)] for r in range(8)]


def tile(rows, cols, color, cell=0.11, gap=0.018, fill=0.85, values=None):
    grid = VGroup(*(Square(cell).set_stroke(width=0).set_fill(color, fill if values is None else
                                                              0.25 + 0.7 * values[k // cols][k % cols])
                    for k in range(rows * cols)))
    return grid.arrange_in_grid(rows=rows, cols=cols, buff=gap)


def area(width, height, title, note, edge=PANEL_EDGE):
    box = panel(width, height, edge=edge)
    head = words(title, size=21, color=INK, weight=MEDIUM).next_to(box, UP, buff=0.14).align_to(box, LEFT)
    sub = words(note, size=16, color=INK_DIM).next_to(head, RIGHT, buff=0.2).align_to(head, DOWN)
    return VGroup(box, head, sub)


class Tiles(Clip):
    def construct(self):
        self.tensor_core()
        self.where_data_lives()
        self.before()
        self.now()
        self.finale("The kernel re-packs the work into the exact shape the T4's fast units want.",
                    "Tesla T4 · attention per layer, 16 × 12 heads × 512 tokens", size=28)

    # 1 ------------------------------------------------------------------------------------------------------------
    def tensor_core(self):
        self.chapter(1, 4, "What a tensor core does")
        cell = 0.17
        a = tile(16, 8, CPU, cell=cell, gap=0.025, values=A_VALUES)
        b = tile(8, 8, TERM_C2P, cell=cell, gap=0.025, values=B_VALUES)
        c = tile(16, 8, GPU, cell=cell, gap=0.025, fill=0.0)
        for sq in c:
            sq.set_stroke(PANEL_EDGE, 1)
        times, equals = words("×", size=40, color=INK_DIM), words("=", size=40, color=INK_DIM)
        product = VGroup(a, times, b, equals, c).arrange(RIGHT, buff=0.45).move_to(UP * 0.35)
        sizes = VGroup(*(words(n, size=18, color=INK_DIM, font=MONO).next_to(g, DOWN, buff=0.15)
                         for g, n in ((a, "16 × 8"), (b, "8 × 8"), (c, "16 × 8"))))
        self.say("Attention is mostly multiplying blocks of numbers. Here is one block: 16 rows of 8 numbers.",
                 LaggedStart(*(FadeIn(sq, scale=0.5) for sq in a), lag_ratio=0.01), FadeIn(sizes[0]))
        self.read()
        self.say("And a second block, 8 by 8, to multiply it with.",
                 FadeIn(times), LaggedStart(*(FadeIn(sq, scale=0.5) for sq in b), lag_ratio=0.01), FadeIn(sizes[1]))
        self.read()

        row = VGroup(*a[16:24])
        col = VGroup(*(b[k * 8 + 3] for k in range(8)))
        out = c[2 * 8 + 3]
        row_box = SurroundingRectangle(row, buff=0.03, corner_radius=0.03).set_stroke(INK, 2)
        col_box = SurroundingRectangle(col, buff=0.03, corner_radius=0.03).set_stroke(INK, 2)
        self.say("One cell of the answer takes one row of the first block and one column of the second.",
                 FadeIn(equals), FadeIn(c), FadeIn(sizes[2]), Create(row_box), Create(col_box),
                 *(sq.animate.set_opacity(0.3) for sq in a if sq not in row),
                 *(sq.animate.set_opacity(0.3) for sq in b if sq not in col))
        self.read()
        pairs = VGroup(*(Line(r.get_center(), k.get_center()).set_stroke(INK, 1.2, opacity=0.6) for r, k in zip(row, col)))
        self.say("Eight multiplications, added up, give one number.",
                 LaggedStart(*(ShowPassingFlash(p.copy().set_stroke(NUMBER, 3, 1), time_width=0.5) for p in pairs),
                             lag_ratio=0.12),
                 out.animate.set_fill(GPU, 0.85).set_stroke(GPU, 2), run_time=2.4)
        self.read()

        instruction = words("mma.sync m16n8k8: fp16 in, fp32 sums", size=20, color=GPU, font=MONO)
        instruction.next_to(sizes, DOWN, buff=0.35)
        self.say("A tensor core does that for all 128 cells at once, in a single instruction: 1,024 multiplications.",
                 FadeOut(row_box), FadeOut(col_box), *(sq.animate.set_opacity(1) for sq in [*a, *b]),
                 LaggedStart(*(sq.animate.set_fill(GPU, 0.85).set_stroke(width=0) for sq in c), lag_ratio=0.01),
                 FadeIn(instruction, shift=0.1 * UP), run_time=2.6)
        self.read()
        self.say("Kept fed, the T4's tensor cores can do 65 trillion of these a second. The hard part is keeping them fed.")
        self.read(extra=0.5)
        self.product = VGroup(product, sizes, instruction)

    # 2 ------------------------------------------------------------------------------------------------------------
    def where_data_lives(self):
        self.chapter(2, 4, "Where the data lives")
        memory = area(3.0, 3.2, "GPU memory", "large, far away").move_to(LEFT * 4.9 + DOWN * 0.1)
        chip_mem = area(3.0, 3.2, "on-chip memory", "64 KB, fast").move_to(LEFT * 0.4 + DOWN * 0.1)
        core_box = RoundedRectangle(width=2.6, height=1.6, corner_radius=0.16).set_fill(PANEL, 1).set_stroke(GPU, 2.5)
        core_box.move_to(RIGHT * 4.5 + DOWN * 0.1)
        core = VGroup(core_box, words("tensor cores", size=22, color=GPU, weight=MEDIUM).move_to(core_box))
        small = self.product[0][0].copy()
        self.say("The tensor cores sit at the end of a chain.",
                 FadeOut(self.product), Derive(small, core), run_time=2.0)
        self.read()

        q = Rectangle(width=0.5, height=2.2).set_fill(CPU, 0.35).set_stroke(CPU, 1.5)
        k = Rectangle(width=0.5, height=2.2).set_fill(CPU, 0.35).set_stroke(CPU, 1.5)
        pos = Rectangle(width=0.85, height=2.2).set_fill(TERM_C2P, 0.3).set_stroke(TERM_C2P, 1.5)
        stores = VGroup(q, k, pos).arrange(RIGHT, buff=0.28).move_to(memory[0]).shift(UP * 0.12)
        names = VGroup(*(words(n, size=17, color=INK_DIM).next_to(m, DOWN, buff=0.08)
                         for n, m in zip(("Q", "K", "positions"), stores)))
        self.say("The words' vectors, Q and K, and the position vectors live in GPU memory: plenty of room, but far away.",
                 FadeIn(memory), LaggedStart(*(GrowFromEdge(s, DOWN) for s in stores), lag_ratio=0.2), FadeIn(names),
                 run_time=2.4)
        self.read()
        self.say("Next to the tensor cores is a small, fast on-chip memory. Tiles must be there when the cores want them.",
                 FadeIn(chip_mem))
        self.read()
        self.board = VGroup(memory, chip_mem, core, stores, names)
        self.memory, self.chip_mem, self.core_box, self.core_label, self.stores = memory, chip_mem, core_box, core[1], stores

    # 3 ------------------------------------------------------------------------------------------------------------
    def before(self):
        self.chapter(3, 4, "Before: starved")
        stores, core_box = self.stores, self.core_box
        rng = np.random.default_rng(3)
        reads = VGroup()
        for _ in range(9):
            for source in stores:
                start = source.get_center() + np.array([rng.uniform(-0.2, 0.2), rng.uniform(-1.1, 1.1), 0])
                end = core_box.get_left() + np.array([0, rng.uniform(-0.6, 0.6), 0])
                reads.add(Line(start, end).set_stroke(source.get_stroke_color(), 1.4, opacity=0.55))
        self.say("The previous kernel fetched Q, K and a position vector separately for each pair of words.",
                 LaggedStart(*(ShowPassingFlash(r.copy().set_stroke(width=3, opacity=1), time_width=0.6) for r in reads),
                             lag_ratio=0.06), LaggedStart(*(Create(r) for r in reads), lag_ratio=0.06), run_time=3.0)
        self.read()
        idle = words("mostly idle", size=18, color=INK_DIM).next_to(core_box, DOWN, buff=0.15)
        self.say("Small scattered reads from far away. The tensor cores spent most of their time waiting.",
                 core_box.animate.set_stroke(WASTE, 2.5), self.core_label.animate.set_color(WASTE), FadeIn(idle))
        self.read()

        self.rate = ValueTracker(0)
        rate = counter(self.rate, size=40, color=NUMBER, fmt="{:.1f}")
        unit = words("TFLOP/s", size=20, color=INK_DIM)
        rate.add_updater(lambda m: m.next_to(idle, DOWN, buff=0.3).shift(LEFT * 0.55))
        unit.add_updater(lambda m: m.next_to(rate, RIGHT, buff=0.12).align_to(rate, DOWN))
        rate.update(), unit.update()
        compare = words("plain attention on the same T4: 17", size=16, color=INK_DIM).next_to(unit, DOWN, buff=0.3)
        compare.match_x(core_box)
        self.add(rate)
        self.say("Measured: 2.7 trillion operations a second, where plain attention on the same T4 reaches 17.",
                 FadeIn(unit), self.rate.animate.set_value(2.7), FadeIn(compare), run_time=2.4)
        self.read(extra=0.5)
        self.beat(FadeOut(VGroup(reads, idle, rate, unit, compare)), core_box.animate.set_stroke(GPU, 2.5),
                  self.core_label.animate.set_color(GPU), run_time=1.0)
        self.remove(rate)

    # 4 ------------------------------------------------------------------------------------------------------------
    def now(self):
        self.chapter(4, 4, "Now: load once, reuse")
        memory, chip_mem, core_box, stores = self.memory, self.chip_mem, self.core_box, self.stores
        slices = VGroup(*(Rectangle(width=m.width, height=0.6).set_fill(m.get_fill_color(), 0.9).set_stroke(width=0)
                          .move_to(m.get_top() + DOWN * 0.45) for m in stores))
        landed = VGroup(*(s.copy() for s in slices)).arrange(RIGHT, buff=0.15).move_to(chip_mem[0]).shift(UP * 0.75)
        load = Arrow(memory[0].get_right(), chip_mem[0].get_left(), buff=0.12, stroke_width=5,
                     max_tip_length_to_length_ratio=0.2).set_color(INK).shift(UP * 0.75)
        self.say("Our kernel takes a whole tile of 32 words at once: one wide, contiguous read.",
                 FadeIn(slices), run_time=1.4)
        self.beat(GrowArrow(load), *(Derive(s, l) for s, l in zip(slices, landed)), run_time=2.0)
        self.read()
        window = words("only the distances\nthis tile needs", size=16, color=TERM_C2P).next_to(landed, DOWN, buff=0.12)
        self.say("For positions it loads only the distances that occur between these two tiles, not the whole table.",
                 FadeIn(window, shift=0.1 * UP))
        self.read()

        feed = Arrow(chip_mem[0].get_right(), core_box.get_left(), buff=0.1, stroke_width=5,
                     max_tip_length_to_length_ratio=0.15).set_color(GPU)
        minis = VGroup(*(tile(4, 2, GPU, cell=0.09, gap=0.015).move_to(landed.get_center()) for _ in range(10)))
        self.say("Then it cuts the tile into the 16-by-8 pieces the tensor cores take, and reuses it for every pair.",
                 GrowArrow(feed), core_box.animate.set_fill(GPU, 0.25),
                 LaggedStart(*(Succession(FadeIn(m, run_time=0.15),
                                          m.animate(run_time=0.55).move_to(core_box.get_center()).set_opacity(0))
                               for m in minis), lag_ratio=0.12), run_time=2.8)
        self.read()

        prefetch = Arrow(memory[0].get_right(), chip_mem[0].get_left(), buff=0.12, stroke_width=3,
                         max_tip_length_to_length_ratio=0.2).set_color(INK_DIM).shift(DOWN * 0.9)
        next_tile = VGroup(*(Rectangle(width=m.width, height=0.6).set_fill(m.get_fill_color(), 0.5).set_stroke(width=0)
                             .move_to(m.get_center() + DOWN * 0.2) for m in stores))
        note = words("the T4 has no asynchronous copy: the kernel overlaps by hand", size=16, color=INK_FAINT)
        note.next_to(VGroup(memory[0], chip_mem[0]), DOWN, buff=0.3).align_to(memory[0], LEFT)
        busy = words("busy", size=18, color=GPU).next_to(core_box, DOWN, buff=0.15)
        minis2 = VGroup(*(tile(4, 2, GPU, cell=0.09, gap=0.015).move_to(landed.get_center()) for _ in range(10)))
        self.say("While the cores work, the next tile is already on its way into registers, so they never wait.",
                 FadeIn(next_tile), GrowArrow(prefetch), FadeIn(note), FadeIn(busy),
                 LaggedStart(*(Succession(FadeIn(m, run_time=0.15),
                                          m.animate(run_time=0.55).move_to(core_box.get_center()).set_opacity(0))
                               for m in minis2), lag_ratio=0.12), run_time=2.8)
        self.read()

        self.beat(FadeOut(VGroup(self.board, slices, landed, load, window, feed, prefetch, next_tile, note, busy)),
                  run_time=1.0)
        chart_title = words("attention per layer, 16 × 12 heads × 512 tokens, Tesla T4", size=20, color=INK_DIM)
        unit = 6.0 / 7.0
        rows = VGroup()
        for k, (label, value, color, text) in enumerate([("before: Triton kernel + position tables", 7.0, WASTE, "7.0 ms"),
                                                          ("glinert4 kernel", 2.84, GPU, "2.84 ms")]):
            y = 0.6 - k * 0.7
            bar = Rectangle(width=value * unit, height=0.38).set_fill(color, 0.9).set_stroke(width=0)
            bar.move_to([-1.2, y, 0], aligned_edge=LEFT)
            name = words(label, size=20, color=INK).move_to([-1.5, y, 0], aligned_edge=RIGHT)
            number = words(text, size=20, color=NUMBER if color == GPU else INK, weight=BOLD, font=MONO)
            rows.add(VGroup(name, bar, number.next_to(bar, RIGHT, buff=0.2)))
        chart_title.next_to(rows, UP, buff=0.3).align_to(rows, LEFT)
        self.say("Before: 7.0 milliseconds of attention per layer, counting the position tables built beforehand.",
                 FadeIn(chart_title), FadeIn(rows[0][0]), GrowFromEdge(rows[0][1], LEFT), FadeIn(rows[0][2]))
        self.read()
        gain = chip("2.5× faster", NUMBER, size=22).next_to(rows[1][2], RIGHT, buff=0.35)
        self.say("With the tiles re-packed: 2.84 milliseconds. The same arithmetic, two and a half times faster.",
                 FadeIn(rows[1][0]), GrowFromEdge(rows[1][1], LEFT), FadeIn(rows[1][2]), run_time=2.0)
        self.beat(FadeIn(gain, scale=1.15), run_time=0.8)
        self.read()
