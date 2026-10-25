# Clip E: "The kernel re-packs the work into the exact shape the T4's fast units want."
# Beats: a tensor core multiplies a 16x8 tile by an 8x8 tile in one instruction; the old kernel fed it with three
# scattered reads from GPU memory per pair of words, so it sat idle; our kernel loads a tile once into on-chip memory,
# prefetches the next one into registers (the T4 has no asynchronous copy), and streams 16x8 tiles into the tensor
# cores; attention per layer drops from 7.0 ms to 2.84 ms.
# Numbers: Tesla T4, 16 x 12 heads x 512 tokens, per layer (kernel log rows 24, 25, 31, 38): Triton kernel 4.80 ms +
# position tables 2.2 ms; ours 2.84 ms; Triton disentangled kernel 2.7 TFLOP/s vs plain attention (SDPA) 17.0 TFLOP/s.
from style import *


def tile(rows, cols, color, cell=0.11, gap=0.018, fill=0.85):
    grid = VGroup(*(Square(cell).set_stroke(width=0).set_fill(color, fill) for _ in range(rows * cols)))
    return grid.arrange_in_grid(rows=rows, cols=cols, buff=gap)


def area(width, height, title, note, edge=PANEL_EDGE):
    box = panel(width, height, edge=edge)
    head = words(title, size=21, color=INK, weight=MEDIUM).next_to(box, UP, buff=0.14).align_to(box, LEFT)
    sub = words(note, size=16, color=INK_DIM).next_to(head, RIGHT, buff=0.2).align_to(head, DOWN)
    return VGroup(box, head, sub)


class Tiles(Clip):
    def construct(self):
        heading = words("The T4's fastest units multiply small tiles", size=26, color=INK).to_corner(UL, buff=0.6)
        a, b, c = tile(16, 8, CPU), tile(8, 8, TERM_C2P), tile(16, 8, GPU, fill=0.0)
        times, equals = words("×", size=36, color=INK_DIM), words("=", size=36, color=INK_DIM)
        product = VGroup(a, times, b, equals, c).arrange(RIGHT, buff=0.35).move_to(UP * 0.3)
        for grid, name in ((a, "16 × 8"), (b, "8 × 8"), (c, "16 × 8")):
            grid.add(words(name, size=18, color=INK_DIM, font=MONO).next_to(grid, DOWN, buff=0.15))
        instruction = words("one instruction: mma.sync m16n8k8 (fp16 in, fp32 accumulate)", size=20, color=GPU, font=MONO)
        instruction.next_to(product, DOWN, buff=0.55)
        self.beat(FadeIn(heading, shift=0.2 * DOWN), run_time=0.7)
        self.beat(LaggedStart(FadeIn(a, shift=0.2 * RIGHT), FadeIn(times), FadeIn(b, shift=0.2 * RIGHT), FadeIn(equals),
                              lag_ratio=0.25), run_time=1.2)
        self.beat(LaggedStart(*(sq.animate.set_fill(GPU, 0.85) for sq in c[:-1]), lag_ratio=0.01), FadeIn(c[-1]),
                  FadeIn(instruction, shift=0.1 * UP), run_time=1.2)
        self.pause(1.0)

        memory = area(3.0, 3.2, "GPU memory", "large, far away").move_to(LEFT * 4.9 + DOWN * 0.3)
        chip_mem = area(3.0, 3.2, "on-chip memory", "small, fast").move_to(LEFT * 0.4 + DOWN * 0.3)
        core_box = RoundedRectangle(width=2.6, height=1.6, corner_radius=0.16).set_fill(PANEL, 1).set_stroke(GPU, 2.5)
        core_box.move_to(RIGHT * 4.5 + DOWN * 0.3)
        core_label = words("tensor cores", size=22, color=GPU, weight=MEDIUM).move_to(core_box)
        core = VGroup(core_box, core_label)
        q = Rectangle(width=0.5, height=2.2).set_fill(CPU, 0.35).set_stroke(CPU, 1.5)
        k = Rectangle(width=0.5, height=2.2).set_fill(CPU, 0.35).set_stroke(CPU, 1.5)
        pos = Rectangle(width=0.85, height=2.2).set_fill(TERM_C2P, 0.3).set_stroke(TERM_C2P, 1.5)
        stores = VGroup(q, k, pos).arrange(RIGHT, buff=0.28).move_to(memory[0]).shift(UP * 0.12)
        store_names = VGroup(*(words(n, size=17, color=INK_DIM).next_to(m, DOWN, buff=0.08)
                               for n, m in zip(("Q", "K", "positions"), stores)))
        heading2 = words("Before: the kernel starved them", size=26, color=INK).move_to(heading, aligned_edge=LEFT)
        self.beat(FadeOut(VGroup(product, instruction)), ReplacementTransform(heading, heading2),
                  FadeIn(memory), FadeIn(stores), FadeIn(store_names), FadeIn(chip_mem), FadeIn(core), run_time=1.2)

        rng = np.random.default_rng(3)
        reads = VGroup()
        for _ in range(9):
            for source in stores:
                start = source.get_center() + np.array([rng.uniform(-0.2, 0.2), rng.uniform(-1.1, 1.1), 0])
                end = core_box.get_left() + np.array([0, rng.uniform(-0.6, 0.6), 0])
                reads.add(Line(start, end).set_stroke(source.get_stroke_color(), 1.4, opacity=0.55))
        scattered = words("3 scattered reads for every pair of words", size=20, color=INK)
        scattered.next_to(memory, DOWN, buff=0.35).align_to(memory, LEFT)
        self.beat(LaggedStart(*(ShowPassingFlash(r.copy().set_stroke(width=3, opacity=1), time_width=0.6) for r in reads),
                              lag_ratio=0.06), LaggedStart(*(Create(r) for r in reads), lag_ratio=0.06),
                  FadeIn(scattered, shift=0.1 * UP), run_time=2.4)
        idle = words("mostly idle", size=18, color=INK_DIM).next_to(core_box, DOWN, buff=0.15)
        figure = VGroup(words("2.7", size=40, color=NUMBER, weight=BOLD, font=MONO),
                        words("TFLOP/s", size=20, color=INK_DIM)).arrange(RIGHT, buff=0.12)
        figure[1].align_to(figure[0], DOWN)
        rate = VGroup(figure, words("plain attention reaches 17", size=17, color=INK_DIM)).arrange(DOWN, buff=0.08)
        rate.next_to(idle, DOWN, buff=0.3)
        self.beat(core_box.animate.set_stroke(WASTE, 2.5), core_label.animate.set_color(WASTE), FadeIn(idle),
                  FadeIn(rate, shift=0.1 * UP), run_time=1.0)
        self.pause(1.0)

        heading3 = words("Now: load a tile once, reuse it many times", size=26, color=INK).move_to(heading, aligned_edge=LEFT)
        self.beat(FadeOut(reads), FadeOut(scattered), FadeOut(idle), FadeOut(rate),
                  core_box.animate.set_stroke(GPU, 2.5), core_label.animate.set_color(GPU),
                  ReplacementTransform(heading2, heading3), run_time=0.9)
        slices = VGroup(*(Rectangle(width=m.width, height=0.6).set_fill(m.get_fill_color(), 0.9).set_stroke(width=0)
                          .move_to(m.get_top() + DOWN * 0.45) for m in stores))
        landed = VGroup(*(s.copy() for s in slices)).arrange(RIGHT, buff=0.15).move_to(chip_mem[0]).shift(UP * 0.75)
        loads = VGroup(Arrow(memory[0].get_right(), chip_mem[0].get_left(), buff=0.12, stroke_width=5,
                             max_tip_length_to_length_ratio=0.2).set_color(INK).shift(UP * 0.75))
        once = words("one wide read", size=17, color=INK).next_to(loads, UP, buff=0.08)
        window = words("only the distances it needs", size=16, color=TERM_C2P).next_to(landed, DOWN, buff=0.12)
        self.beat(FadeIn(slices), run_time=0.4)
        self.beat(GrowArrow(loads[0]), FadeIn(once), *(TransformFromCopy(s, l) for s, l in zip(slices, landed)),
                  run_time=1.2)
        self.beat(FadeIn(window, shift=0.1 * UP), run_time=0.6)

        prefetch = Arrow(memory[0].get_right(), chip_mem[0].get_left(), buff=0.12, stroke_width=3,
                         max_tip_length_to_length_ratio=0.2).set_color(INK_DIM).shift(DOWN * 0.9)
        prefetch_note = words("meanwhile, the next tile is fetched into registers", size=17, color=INK_DIM)
        prefetch_note.next_to(VGroup(memory[0], chip_mem[0]), DOWN, buff=0.3)
        no_async = words("the T4 has no asynchronous copy, so the kernel overlaps it by hand", size=15, color=INK_FAINT)
        no_async.next_to(prefetch_note, DOWN, buff=0.06)
        feed = Arrow(chip_mem[0].get_right(), core_box.get_left(), buff=0.1, stroke_width=5,
                     max_tip_length_to_length_ratio=0.15).set_color(GPU)
        minis = VGroup(*(tile(4, 2, GPU, cell=0.09, gap=0.015).move_to(landed.get_center()) for _ in range(10)))
        self.add(feed)
        self.beat(GrowArrow(feed), GrowArrow(prefetch), FadeIn(prefetch_note), FadeIn(no_async),
                  core_box.animate.set_fill(GPU, 0.25), run_time=0.8)
        self.beat(LaggedStart(*(Succession(FadeIn(m, run_time=0.15), m.animate(run_time=0.55).move_to(core_box.get_center())
                                           .set_opacity(0)) for m in minis), lag_ratio=0.12),
                  run_time=2.6)
        busy = words("busy", size=18, color=GPU).next_to(core_box, DOWN, buff=0.15)
        self.beat(FadeIn(busy), run_time=0.4)
        self.pause(0.6)

        board = VGroup(memory, chip_mem, core, stores, store_names, slices, landed, loads, once, window, prefetch,
                       prefetch_note, no_async, feed, busy)
        self.beat(FadeOut(board), run_time=0.8)
        chart_title = words("attention per layer, 16 × 12 heads × 512 tokens, Tesla T4", size=20, color=INK_DIM)
        unit = 6.0 / 7.0
        rows = VGroup()
        for k, (label, value, color, text) in enumerate([("before: Triton kernel + position tables", 7.0, WASTE, "7.0 ms"),
                                                          ("glinert4 kernel", 2.84, GPU, "2.84 ms")]):
            y = 0.2 - k * 0.7
            bar = Rectangle(width=value * unit, height=0.38).set_fill(color, 0.9).set_stroke(width=0)
            bar.move_to([-1.2, y, 0], aligned_edge=LEFT)
            name = words(label, size=20, color=INK).move_to([-1.5, y, 0], aligned_edge=RIGHT)
            number = words(text, size=20, color=NUMBER if color == GPU else INK, weight=BOLD, font=MONO)
            rows.add(VGroup(name, bar, number.next_to(bar, RIGHT, buff=0.2)))
        chart_title.next_to(rows, UP, buff=0.3).align_to(rows, LEFT)
        gain = chip("2.5× faster", NUMBER, size=22).next_to(rows[1][2], RIGHT, buff=0.35)
        self.beat(FadeIn(chart_title), LaggedStart(*(AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))
                                                     for r in rows), lag_ratio=0.35), run_time=1.6)
        self.beat(FadeIn(gain, scale=1.15), run_time=0.6)
        self.pause(0.6)

        claim = caption("The kernel re-packs the work into the exact shape the T4's fast units want.", size=25)
        self.beat(FadeIn(claim, shift=0.2 * UP), run_time=1.2)
        self.wait(HOLD)
        self.beat(*(FadeOut(mob) for mob in self.mobjects), run_time=1.0)
