# Clip B: "The slow line wasn't slow: it was where the CPU waited for the GPU."
# Chapters: 1 the profile (four lines of serving code; cProfile blames .tolist() for half the time); 2 two workers
# (launching the encoder only queues work on the GPU and returns); 3 .tolist() is waiting (the line that needs the
# numbers blocks until the GPU is done, the profiler books that wait to it; between batches the GPU sits idle);
# 4 the fix (prepare and launch the next batch before asking for this one's result: the GPU's gaps close).
# Numbers: Tesla T4, benchmark_results/pipeline_2026-10-25/README.md: cProfile of the queue path, .tolist() 2.538 s =
# 50.3% of profiled time (round 5), 0.778 s with the one-thread pipeline (round 6); GPU busy (kernel time over wall
# time, torch.profiler) 70.0% before, 94.2% with the pipelined configuration. Block lengths are illustrative.
from style import *

PREPARE, ENCODE, DECODE = 0.5, 2.1, 0.4
PERIOD = PREPARE + ENCODE + DECODE
SCALE = 1.15
X0 = -4.6
CPU_Y, GPU_Y = 0.6, -0.55
CODE = ["batch = prepare(texts)", "out   = encoder(batch)", "spans = out.tolist()", "found = decode(spans)"]
NOTES = ["# tokenize and pad", "# run the 12 layers", "# scores as numbers", "# spans and labels"]


def x(t):
    return X0 + t * SCALE


def block(y, t0, t1, color, label="", fill=0.85, height=0.6, ink=BACKGROUND):
    rect = RoundedRectangle(width=max((t1 - t0) * SCALE - 0.04, 0.04), height=height, corner_radius=0.05)
    rect.set_fill(color, fill).set_stroke(color, 1.5).move_to([(x(t0) + x(t1)) / 2, y, 0])
    if not label or rect.width < 0.9:
        return VGroup(rect)
    text = words(label, size=17, color=ink, weight=MEDIUM, font=MONO if label.startswith(".") else SANS)
    if text.width > rect.width - 0.12:
        text.scale_to_fit_width(rect.width - 0.12)
    return VGroup(rect, text.move_to(rect))


def waiting(rect, label="waiting"):
    shell = rect.copy().set_fill(BACKGROUND, 1).set_stroke(WASTE, 1.5)
    group = VGroup(shell, hatch(rect, color=WASTE))
    if label and rect.width > 0.9:
        group.add(words(label, size=17, color=INK_DIM, weight=MEDIUM).move_to(rect))
    return group


class Waiting(Clip):
    def construct(self):
        self.the_profile()
        self.two_workers()
        self.tolist_is_waiting()
        self.the_fix()
        self.finale("The slow line wasn't slow: it was where the CPU waited for the GPU.",
                    "Tesla T4 · cProfile and torch.profiler of the serving queue · rounds 5 and 6")

    # 1 ------------------------------------------------------------------------------------------------------------
    def the_profile(self):
        self.chapter(1, 4, "What the profiler said")
        lines = VGroup(*(VGroup(mono(c, size=26, color=INK), mono(n, size=22, color=INK_FAINT)) for c, n in zip(CODE, NOTES)))
        for line in lines:
            line[1].next_to(line[0], RIGHT, buff=0.5)
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        for line in lines[1:]:
            line[1].align_to(lines[0][1], LEFT)
        back = panel(lines.width + 0.8, lines.height + 0.6).move_to(lines)
        self.code = VGroup(back, lines).move_to(UP * 1.25)
        self.say("This is the serving code for one batch of text: four lines of Python.",
                 FadeIn(back), LaggedStart(*(FadeIn(l, shift=0.1 * RIGHT) for l in lines), lag_ratio=0.25))
        self.read()

        total, left, bar_y = 10.0, -5.0, -1.3
        shares = [(".tolist()", 0.503, CPU), (".cpu()", 0.075, "#3D8DA6"), ("everything else", 0.422, INK_FAINT)]
        parts = VGroup()
        for label, share, color in shares:
            width = total * share
            rect = Rectangle(width=width - 0.03, height=0.5).set_stroke(width=0)
            rect.set_fill(color, 0.9 if color != INK_FAINT else 0.5).move_to([left + width / 2, bar_y, 0])
            text = words(label, size=18, color=BACKGROUND if color != INK_FAINT else INK, weight=MEDIUM,
                         font=MONO if label.startswith(".") else SANS)
            if text.width > width - 0.15:
                text.scale_to_fit_width(width - 0.15)
            parts.add(VGroup(rect, text.move_to(rect)))
            left += width
        title = words("cProfile of the serving thread, Tesla T4", size=19, color=INK_DIM)
        title.next_to(parts, UP, buff=0.18).align_to(parts, LEFT)
        self.say("We profiled it on the T4 to see where the time goes.",
                 FadeIn(title), LaggedStart(*(GrowFromEdge(p, LEFT) for p in parts), lag_ratio=0.2), run_time=2.4)
        self.read()

        target = lines[2][0][-9:]
        mark = box(target, CPU, fill=0.15)
        link = connector(mark, parts[0][0], color=CPU)
        share = ValueTracker(0)
        seconds = ValueTracker(0)
        number = counter(share, size=34, color=NUMBER, fmt="{:.0f}% of the time")
        number.add_updater(lambda m: m.next_to(parts, DOWN, buff=0.22).align_to(parts, LEFT))
        secs = counter(seconds, size=24, color=INK_DIM, fmt="{:.2f} s")
        secs.add_updater(lambda m: m.next_to(number, RIGHT, buff=0.35).align_to(number, DOWN))
        number.update(), secs.update()
        self.add(number, secs)
        self.say("One line took half of all the time: the call that turns the result into Python numbers.",
                 DrawBorderThenFill(mark), Create(link), share.animate.set_value(50.3), seconds.animate.set_value(2.54),
                 run_time=2.6)
        self.read()
        self.say("But it only copies a small array of scores to the CPU. That should take microseconds, not seconds.",
                 lines[0].animate.set_opacity(0.35), lines[1].animate.set_opacity(0.35), lines[3].animate.set_opacity(0.35))
        self.read(extra=0.5)
        self.profile = VGroup(title, parts)
        self.parts = parts
        self.beat(FadeOut(VGroup(mark, link, number, secs, self.profile)), *(l.animate.set_opacity(1) for l in lines),
                  run_time=1.0)
        self.remove(number, secs)

    # 2 ------------------------------------------------------------------------------------------------------------
    def two_workers(self):
        self.chapter(2, 4, "Two workers")
        lines = self.code[1]
        self.beat(self.code.animate.scale(0.62).to_corner(UR, buff=0.4), run_time=1.4)
        end = x(3 * PERIOD + 0.3)
        lanes = VGroup(*(Line([X0 - 0.15, y, 0], [end, y, 0]).set_stroke(PANEL_EDGE, 1.5) for y in (CPU_Y, GPU_Y)))
        names = VGroup(words("CPU", size=24, color=CPU, weight=MEDIUM).next_to(lanes[0], LEFT, buff=0.3),
                       words("GPU", size=24, color=GPU, weight=MEDIUM).next_to(lanes[1], LEFT, buff=0.3))
        arrow = Arrow([X0 - 0.15, GPU_Y - 0.62, 0], [end, GPU_Y - 0.62, 0], buff=0, stroke_width=2,
                      max_tip_length_to_length_ratio=0.02).set_color(INK_FAINT)
        clock = words("time", size=17, color=INK_FAINT).next_to(arrow, DOWN, buff=0.06).align_to(arrow, RIGHT)
        self.say("Two workers run this code. The CPU runs Python; the GPU runs the encoder's arithmetic.",
                 LaggedStart(*(Create(l) for l in lanes), lag_ratio=0.3), FadeIn(names, shift=0.2 * RIGHT),
                 GrowArrow(arrow), FadeIn(clock), run_time=2.2)
        self.read()
        self.timeline = VGroup(lanes, names, arrow, clock)

        pointer = Triangle(fill_color=CPU, fill_opacity=1, stroke_width=0).rotate(-PI / 2).scale(0.09)
        pointer.next_to(lines[0], LEFT, buff=0.12)
        self.pointer = pointer
        prepare = block(CPU_Y, 0, PREPARE, CPU)
        self.say("Line 1 runs on the CPU: it tokenizes the texts and pads them into one batch.",
                 FadeIn(pointer, shift=0.1 * RIGHT), GrowFromEdge(prepare, LEFT))
        self.read()

        launch = Arrow([x(PREPARE), CPU_Y - 0.3, 0], [x(PREPARE), GPU_Y + 0.3, 0], buff=0.02, stroke_width=3,
                       max_tip_length_to_length_ratio=0.3).set_color(INK)
        queued = words("queued", size=17, color=INK).next_to(launch, RIGHT, buff=0.08)
        self.say("Line 2 does not wait for the encoder. It puts the work in the GPU's queue and returns at once.",
                 pointer.animate.next_to(lines[1], LEFT, buff=0.12), GrowArrow(launch), FadeIn(queued))
        self.read()

        encode = block(GPU_Y, PREPARE, PREPARE + ENCODE, GPU, "encode batch 1")
        self.say("The GPU starts encoding, and the CPU moves straight on to line 3.",
                 pointer.animate.next_to(lines[2], LEFT, buff=0.12), GrowFromEdge(encode, LEFT), FadeOut(queued),
                 run_time=2.4)
        self.read()
        self.first = VGroup(prepare, encode, launch)

    # 3 ------------------------------------------------------------------------------------------------------------
    def tolist_is_waiting(self):
        self.chapter(3, 4, "The slow line is waiting")
        prepare, encode, launch = self.first
        lines = self.code[1]
        wait = block(CPU_Y, PREPARE, PREPARE + ENCODE, CPU, ".tolist()")
        self.say("Line 3 needs the actual numbers. They exist only once the GPU finishes, so the CPU stops here.",
                 GrowFromEdge(wait, LEFT), run_time=2.6)
        self.read()

        guides = VGroup()
        for side in (LEFT, RIGHT):
            p = wait.get_corner(DOWN + side)
            guides.add(DashedLine(p + DOWN * 0.03, [p[0], encode.get_top()[1] + 0.03, 0], dash_length=0.06)
                       .set_stroke(INK, 1.5))
        self.say("Look at the two blocks: the CPU's wait lasts exactly as long as the GPU's work.",
                 FadeOut(launch), Create(guides))
        self.read()

        self.profile.move_to(DOWN * 2.0 + RIGHT * 0.55)
        link = VGroup()
        self.say("The profiler books that wait to .tolist(). This is where its 50% came from.",
                 FadeIn(VGroup(self.profile[0], self.parts[1:])), Derive(wait, self.parts[0]), run_time=2.2)
        self.read()

        hatched = waiting(wait[0])
        bar = waiting(self.parts[0][0], "waiting for the GPU")
        self.say("So .tolist() was never slow. It was the GPU's work, seen from the CPU.",
                 ReplacementTransform(wait, hatched), ReplacementTransform(self.parts[0], bar), run_time=2.2)
        self.read(extra=0.5)

        decode = block(CPU_Y, PREPARE + ENCODE, PERIOD, CPU)
        self.say("Then the CPU decodes the spans, line 4, and starts on the next batch.",
                 FadeOut(VGroup(guides, link, self.profile, bar)),
                 self.pointer.animate.next_to(lines[3], LEFT, buff=0.12), GrowFromEdge(decode, LEFT))
        self.read()

        cpu_blocks, gpu_blocks, waits = VGroup(prepare, decode), VGroup(encode), VGroup(hatched)
        later = []
        for n in (1, 2):
            t = n * PERIOD
            later.append((block(CPU_Y, t, t + PREPARE, CPU), block(GPU_Y, t + PREPARE, t + PREPARE + ENCODE, GPU,
                                                                   f"encode batch {n + 1}"),
                          VGroup(waiting(block(CPU_Y, t + PREPARE, t + PREPARE + ENCODE, CPU)[0])),
                          block(CPU_Y, t + PREPARE + ENCODE, t + PERIOD, CPU)))
        self.say("Batch after batch, the same rhythm: prepare, launch, wait, decode.", FadeOut(self.pointer),
                 Succession(*(Succession(GrowFromEdge(p, LEFT), AnimationGroup(GrowFromEdge(e, LEFT), FadeIn(w)),
                                         GrowFromEdge(d, LEFT)) for p, e, w, d in later)), run_time=3.6)
        for p, e, w, d in later:
            cpu_blocks.add(p, d)
            gpu_blocks.add(e)
            waits.add(w)
        self.read()

        gaps = VGroup()
        for n in range(3):
            start = n * PERIOD
            for t0, t1 in [(start, start + PREPARE), (start + PREPARE + ENCODE, start + PERIOD)]:
                gap = Rectangle(width=(t1 - t0) * SCALE - 0.04, height=0.6).move_to([(x(t0) + x(t1)) / 2, GPU_Y, 0])
                gaps.add(VGroup(gap.set_stroke(INK_FAINT, 1).set_fill(opacity=0), hatch(gap, spacing=0.12, color=INK_FAINT)))
        self.say("And while the CPU decodes and prepares, the GPU has nothing queued. It sits idle.",
                 LaggedStart(*(FadeIn(g) for g in gaps), lag_ratio=0.12), run_time=2.4)
        self.read()

        self.busy = ValueTracker(0)
        self.waited = ValueTracker(0)
        busy_title = words("GPU busy", size=20, color=INK_DIM).move_to(LEFT * 4.6 + DOWN * 2.05, aligned_edge=LEFT)
        wait_title = words("CPU waiting in .tolist()", size=20, color=INK_DIM).move_to(RIGHT * 0.2 + DOWN * 2.05,
                                                                                      aligned_edge=LEFT)
        busy = counter(self.busy, size=36, color=NUMBER, fmt="{:.0f}%")
        busy.add_updater(lambda m: m.next_to(busy_title, RIGHT, buff=0.25))
        waited = counter(self.waited, size=36, color=NUMBER, fmt="{:.2f} s")
        waited.add_updater(lambda m: m.next_to(wait_title, RIGHT, buff=0.25))
        busy.update(), waited.update()
        self.add(busy, waited)
        self.say("Measured on the T4: the GPU was busy only 70% of the time, and the CPU waited 2.54 seconds.",
                 FadeIn(busy_title), FadeIn(wait_title), self.busy.animate.set_value(70),
                 self.waited.animate.set_value(2.54), run_time=2.4)
        self.read()
        self.old = VGroup(cpu_blocks, gpu_blocks, waits, gaps)

    # 4 ------------------------------------------------------------------------------------------------------------
    def the_fix(self):
        self.chapter(4, 4, "The fix: overlap")
        encode_at = [PREPARE + n * ENCODE for n in range(3)]
        before_end = DashedLine([x(3 * PERIOD), CPU_Y + 0.5, 0], [x(3 * PERIOD), GPU_Y - 0.4, 0], dash_length=0.07)
        before_end.set_stroke(INK_FAINT, 1.5)
        before = words("before", size=16, color=INK_FAINT).next_to(before_end, UP, buff=0.06)
        self.say("Keep the old finish line, and clear the timeline.", Create(before_end), FadeIn(before),
                 FadeOut(self.old, shift=0.1 * DOWN))
        self.read()

        p1 = block(CPU_Y, 0, PREPARE, CPU)
        e1 = block(GPU_Y, encode_at[0], encode_at[0] + ENCODE, GPU, "encode batch 1")
        p2 = block(CPU_Y, encode_at[0], encode_at[0] + PREPARE, CPU)
        self.say("New order: launch batch 1, then prepare batch 2 at once, while the GPU is still busy.",
                 Succession(GrowFromEdge(p1, LEFT), AnimationGroup(GrowFromEdge(e1, LEFT), GrowFromEdge(p2, LEFT))),
                 run_time=2.6)
        self.read()

        e2 = block(GPU_Y, encode_at[1], encode_at[1] + ENCODE, GPU, "encode batch 2")
        w1 = waiting(block(CPU_Y, encode_at[0] + PREPARE, encode_at[1], CPU)[0])
        self.say("Batch 2 is already queued when batch 1 finishes, so the GPU goes straight on.",
                 FadeIn(w1), GrowFromEdge(e2, LEFT), run_time=2.4)
        self.read()

        rest_cpu = VGroup(block(CPU_Y, encode_at[1], encode_at[1] + DECODE, CPU),
                          block(CPU_Y, encode_at[1] + DECODE, encode_at[1] + DECODE + PREPARE, CPU),
                          block(CPU_Y, encode_at[2], encode_at[2] + DECODE, CPU),
                          block(CPU_Y, encode_at[2] + ENCODE, encode_at[2] + ENCODE + DECODE, CPU))
        e3 = block(GPU_Y, encode_at[2], encode_at[2] + ENCODE, GPU, "encode batch 3")
        rest_wait = VGroup(*(waiting(block(CPU_Y, t0, t1, CPU)[0], label="")
                             for t0, t1 in [(encode_at[1] + DECODE + PREPARE, encode_at[2]),
                                            (encode_at[2] + DECODE, encode_at[2] + ENCODE)]))
        self.say("The CPU decodes one batch while the GPU encodes the next. Its waits shrink to the leftovers.",
                 LaggedStart(GrowFromEdge(rest_cpu[0], LEFT), GrowFromEdge(rest_cpu[1], LEFT), GrowFromEdge(e3, LEFT),
                             FadeIn(rest_wait[0]), GrowFromEdge(rest_cpu[2], LEFT), FadeIn(rest_wait[1]),
                             GrowFromEdge(rest_cpu[3], LEFT), lag_ratio=0.35), run_time=3.4)
        self.read()

        finish = encode_at[2] + ENCODE + DECODE
        now_end = DashedLine([x(finish), CPU_Y + 0.5, 0], [x(finish), GPU_Y - 0.4, 0], dash_length=0.07)
        now_end.set_stroke(CORRECT, 2)
        now = words("now", size=16, color=CORRECT).next_to(now_end, UP, buff=0.06)
        span_arrow = DoubleArrow([x(finish), CPU_Y + 0.95, 0], [x(3 * PERIOD), CPU_Y + 0.95, 0], buff=0, stroke_width=2,
                                 tip_length=0.12).set_color(CORRECT)
        self.say("The GPU's gaps close: the three batches run back to back and finish earlier.",
                 Create(now_end), FadeIn(now), GrowFromCenter(span_arrow), run_time=2.2)
        self.read()

        self.say("Measured on the T4, the GPU is now busy 94% of the time.", self.busy.animate.set_value(94.2),
                 run_time=2.4)
        self.read()
        self.say("And the time booked to .tolist() fell from 2.54 to 0.78 seconds. Same code, new order.",
                 self.waited.animate.set_value(0.78), run_time=2.4)
        self.read(extra=0.5)
