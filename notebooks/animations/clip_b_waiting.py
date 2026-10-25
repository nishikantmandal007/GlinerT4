# Clip B: "The slow line wasn't slow; it was where the CPU finally waited for the GPU."
# Beats: CPU and GPU lanes on a timeline; Python prepares a batch, queues the encoder and asks for the result with
# .tolist(); the profiler blames .tolist() for half the time; that block is the GPU's work seen from the CPU, so it
# turns into waiting; the pipeline prepares batch n+1 while the GPU encodes batch n, and the GPU's gaps close.
# Numbers: Tesla T4, cProfile of the queue path (round 5): .tolist() 2.54 s = 50.3% of profiled time; after the
# one-thread pipeline (round 6) 0.78 s; GPU busy 70% -> 94% (torch.profiler, kernel time over wall time).
from style import *

PREPARE, ENCODE, DECODE = 0.5, 2.1, 0.4
SCALE = 1.05
X0 = -3.9
CPU_Y, GPU_Y = 1.15, -0.05


def x(t):
    return X0 + t * SCALE


def block(y, t0, t1, color, label="", fill=0.85, height=0.62, ink=BACKGROUND):
    rect = RoundedRectangle(width=(t1 - t0) * SCALE - 0.04, height=height, corner_radius=0.06)
    rect.set_fill(color, fill).set_stroke(color, 1.5).move_to([(x(t0) + x(t1)) / 2, y, 0])
    if not label or rect.width < 0.9:
        return VGroup(rect)
    text = words(label, size=17, color=ink, weight=MEDIUM, font=MONO if label.startswith(".") else SANS)
    if text.width > rect.width - 0.12:
        text.scale_to_fit_width(rect.width - 0.12)
    return VGroup(rect, text.move_to(rect))


class Waiting(Clip):
    def construct(self):
        heading = words("Where did the time go?", size=26, color=INK).to_corner(UL, buff=0.6)
        axis_end = x(9.4)
        lanes = VGroup(*(Line([X0 - 0.15, y, 0], [axis_end, y, 0]).set_stroke(PANEL_EDGE, 1.5) for y in (CPU_Y, GPU_Y)))
        names = VGroup(words("CPU · Python", size=22, color=CPU, weight=MEDIUM).next_to(lanes[0], LEFT, buff=0.3),
                       words("GPU", size=22, color=GPU, weight=MEDIUM).next_to(lanes[1], LEFT, buff=0.3))
        names.shift(UP * 0.0)
        time_arrow = Arrow([X0 - 0.15, GPU_Y - 0.75, 0], [axis_end, GPU_Y - 0.75, 0], buff=0, stroke_width=2,
                           max_tip_length_to_length_ratio=0.02).set_color(INK_FAINT)
        time_label = words("time", size=18, color=INK_FAINT).next_to(time_arrow, DOWN, buff=0.08).align_to(time_arrow, RIGHT)

        self.beat(FadeIn(heading, shift=0.2 * DOWN), run_time=0.7)
        self.beat(LaggedStart(*(Create(l) for l in lanes), lag_ratio=0.2), FadeIn(names, shift=0.2 * RIGHT),
                  GrowArrow(time_arrow), FadeIn(time_label), run_time=1.2)

        swatch = RoundedRectangle(width=0.32, height=0.2, corner_radius=0.04).set_fill(CPU, 0.85).set_stroke(CPU, 1)
        legend = VGroup(swatch, words("prepare / decode a batch: real CPU work", size=17, color=INK_DIM))
        legend.arrange(RIGHT, buff=0.15).next_to(lanes[0], UP, buff=0.45).align_to(lanes[0], LEFT)
        self.beat(FadeIn(legend), run_time=0.5)
        cpu_blocks, gpu_blocks, waits = VGroup(), VGroup(), VGroup()
        t = 0.0
        for batch in range(3):
            prepare = block(CPU_Y, t, t + PREPARE, CPU)
            encode = block(GPU_Y, t + PREPARE, t + PREPARE + ENCODE, GPU, f"encode batch {batch + 1}")
            wait = block(CPU_Y, t + PREPARE, t + PREPARE + ENCODE, CPU, ".tolist()")
            decode = block(CPU_Y, t + PREPARE + ENCODE, t + PREPARE + ENCODE + DECODE, CPU, "")
            run = 0.9 if batch == 0 else 0.55
            self.beat(GrowFromEdge(prepare, LEFT), run_time=0.45 * run)
            self.beat(GrowFromEdge(encode, LEFT), GrowFromEdge(wait, LEFT), run_time=1.1 * run)
            self.beat(GrowFromEdge(decode, LEFT), run_time=0.4 * run)
            cpu_blocks.add(prepare, decode)
            gpu_blocks.add(encode)
            waits.add(wait)
            t += PREPARE + ENCODE + DECODE

        gaps = VGroup()
        for batch in range(3):
            start = batch * (PREPARE + ENCODE + DECODE)
            for t0, t1 in [(start, start + PREPARE)] + ([(start + PREPARE + ENCODE, start + PREPARE + ENCODE + DECODE)]):
                gap = Rectangle(width=(t1 - t0) * SCALE - 0.04, height=0.62).move_to([(x(t0) + x(t1)) / 2, GPU_Y, 0])
                gaps.add(VGroup(gap.set_stroke(INK_FAINT, 1).set_fill(opacity=0), hatch(gap, spacing=0.12, color=INK_FAINT)))
        idle = words("GPU idle", size=18, color=INK_DIM).next_to(gaps[1], DOWN, buff=0.12)
        self.beat(FadeIn(gaps), FadeIn(idle), run_time=0.8)
        self.pause(0.4)

        bar_y = -2.15
        title = words("cProfile of the CPU thread", size=20, color=INK_DIM).move_to([X0 + 2.6, bar_y + 0.55, 0])
        total = 9.0 * SCALE
        shares = [(".tolist()", 0.503, CPU), (".cpu()", 0.075, "#3D8DA6"), ("everything else", 0.422, INK_FAINT)]
        parts, left = VGroup(), X0
        for label, share, color in shares:
            width = total * share
            rect = Rectangle(width=width - 0.03, height=0.5).set_fill(color, 0.9 if color != INK_FAINT else 0.5)
            rect.set_stroke(width=0).move_to([left + width / 2, bar_y, 0])
            text = words(label, size=17, color=BACKGROUND if color != INK_FAINT else INK, weight=MEDIUM,
                         font=MONO if label.startswith(".") else SANS)
            if text.width > width - 0.15:
                text.scale_to_fit_width(width - 0.15)
            parts.add(VGroup(rect, text.move_to(rect)))
            left += width
        title.next_to(parts, UP, buff=0.15).align_to(parts, LEFT)
        share = words("50%", size=40, color=NUMBER, weight=BOLD, font=MONO).next_to(parts[0], RIGHT, buff=0.2)
        share.move_to(parts[0]).shift(UP * 0.0)
        self.beat(FadeIn(title), LaggedStart(*(GrowFromEdge(p, LEFT) for p in parts), lag_ratio=0.15), run_time=1.2)
        blame = words("“.tolist() is the slow line”", size=24, color=INK, weight=MEDIUM)
        blame.next_to(parts, RIGHT, buff=0.35)
        if blame.get_right()[0] > 6.9:
            blame.next_to(parts, DOWN, buff=0.25).align_to(parts, LEFT)
        share.next_to(parts[0], UP, buff=0.12).align_to(parts[0], RIGHT)
        self.beat(Indicate(parts[0], color=NUMBER, scale_factor=1.04), FadeIn(blame, shift=0.1 * UP),
                  FadeIn(share, shift=0.1 * UP), run_time=1.0)
        self.pause(0.8)

        guides = VGroup()
        for wait, encode in zip(waits, gpu_blocks):
            for side in (LEFT, RIGHT):
                p = wait.get_corner(UP + side)
                guides.add(DashedLine(p + UP * 0.05, [p[0], encode.get_bottom()[1] - 0.05, 0], dash_length=0.07)
                           .set_stroke(INK, 1.5))
        self.beat(Create(guides), FadeOut(blame), run_time=1.0)
        hatched = VGroup()
        for wait in waits:
            rect = wait[0]
            hatched.add(VGroup(rect.copy().set_fill(BACKGROUND, 1).set_stroke(WASTE, 1.5), hatch(rect, color=WASTE),
                               words("waiting", size=17, color=INK_DIM, weight=MEDIUM).move_to(rect)))
        meaning = words("it was the GPU's work, seen from the CPU", size=24, color=INK, weight=MEDIUM)
        meaning.next_to(parts, DOWN, buff=0.3).align_to(parts, LEFT)
        waiting_bar = VGroup(parts[0][0].copy().set_fill(BACKGROUND, 1).set_stroke(WASTE, 1.5), hatch(parts[0][0], color=WASTE),
                             words("waiting for the GPU", size=17, color=INK_DIM, weight=MEDIUM).move_to(parts[0]))
        self.beat(*(ReplacementTransform(w, h) for w, h in zip(waits, hatched)),
                  ReplacementTransform(parts[0], waiting_bar), FadeIn(meaning, shift=0.1 * UP), run_time=1.4)
        self.pause(1.4)

        self.beat(FadeOut(VGroup(guides, gaps, idle, title, parts[1:], waiting_bar, meaning, share, legend)),
                  run_time=0.8)
        fix = words("The fix: prepare batch n+1 while the GPU encodes batch n", size=26, color=INK)
        fix.move_to(heading, aligned_edge=LEFT)
        self.beat(ReplacementTransform(heading, fix), run_time=0.8)

        new_cpu, new_gpu, new_wait = VGroup(), VGroup(), VGroup()
        encode_at = [PREPARE + n * ENCODE for n in range(3)]
        for n, t0 in enumerate(encode_at):
            new_gpu.add(block(GPU_Y, t0, t0 + ENCODE, GPU, f"encode batch {n + 1}"))
        cpu_work = [(0, PREPARE), (encode_at[0], encode_at[0] + PREPARE),
                    (encode_at[1], encode_at[1] + DECODE), (encode_at[1] + DECODE, encode_at[1] + DECODE + PREPARE),
                    (encode_at[2], encode_at[2] + DECODE), (encode_at[2] + ENCODE, encode_at[2] + ENCODE + DECODE)]
        for t0, t1 in cpu_work:
            new_cpu.add(block(CPU_Y, t0, t1, CPU))
        for t0, t1 in [(encode_at[0] + PREPARE, encode_at[1]), (encode_at[1] + DECODE + PREPARE, encode_at[2]),
                       (encode_at[2] + DECODE, encode_at[2] + ENCODE)]:
            short = block(CPU_Y, t0, t1, WASTE, fill=0.0)
            new_wait.add(VGroup(short[0].set_fill(BACKGROUND, 1).set_stroke(WASTE, 1.5), hatch(short[0], color=WASTE)))
        last = encode_at[2] + ENCODE
        old = VGroup(cpu_blocks, gpu_blocks, hatched)
        self.beat(FadeOut(old, shift=0.15 * DOWN), run_time=0.6)
        self.beat(LaggedStart(*(GrowFromEdge(b, LEFT) for b in new_gpu), lag_ratio=0.35),
                  LaggedStart(*(GrowFromEdge(b, LEFT) for b in new_cpu), lag_ratio=0.12),
                  LaggedStart(*(FadeIn(b) for b in new_wait), lag_ratio=0.35), run_time=2.4)
        end_old = DashedLine([x(9.0), CPU_Y + 0.55, 0], [x(9.0), GPU_Y - 0.5, 0], dash_length=0.07).set_stroke(INK_FAINT, 1.5)
        end_new = DashedLine([x(last + DECODE), CPU_Y + 0.55, 0], [x(last + DECODE), GPU_Y - 0.5, 0], dash_length=0.07)
        end_new.set_stroke(CORRECT, 2)
        done = VGroup(words("before", size=16, color=INK_FAINT).next_to(end_old, UP, buff=0.08),
                      words("now", size=16, color=CORRECT).next_to(end_new, UP, buff=0.08))
        self.beat(Create(end_old), Create(end_new), FadeIn(done), run_time=0.9)

        stats = VGroup(
            VGroup(words("GPU busy", size=20, color=INK_DIM), words("70% → 94%", size=36, color=NUMBER, weight=BOLD, font=MONO)),
            VGroup(words("CPU waiting in .tolist()", size=20, color=INK_DIM),
                   words("2.54 s → 0.78 s", size=36, color=NUMBER, weight=BOLD, font=MONO)))
        for stat in stats:
            stat.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        stats.arrange(RIGHT, buff=1.4, aligned_edge=UP).move_to([X0 + 4.2, -2.1, 0]).align_to(lanes, LEFT)
        self.beat(LaggedStart(*(FadeIn(s, shift=0.15 * UP) for s in stats), lag_ratio=0.3), run_time=1.2)
        self.pause(0.6)

        claim = caption("The slow line wasn't slow: it was where the CPU waited for the GPU.", size=28)
        note = words("Tesla T4 · cProfile and torch.profiler of the serving queue", size=18, color=INK_DIM)
        note.next_to(claim, DOWN, buff=0.12)
        self.beat(FadeIn(claim, shift=0.2 * UP), FadeIn(note), run_time=1.2)
        self.wait(HOLD)
        self.beat(*(FadeOut(mob) for mob in self.mobjects), run_time=1.0)
