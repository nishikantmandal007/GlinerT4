# Clip A: "Instead of reading the document three times, read it once and ask all the questions."
# Beats: a page and the label passes appear; the old layout sweeps the page three times (GPU work x3, 59 chunks/s);
# the passes collapse into one prompt; one sweep finds every entity (GPU work x1, 142 chunks/s, 2.4x); the claim.
# Numbers: Tesla T4, realistic corpus (290 chunks), round 8: 3 passes 59.0 chunks/s, single pass 142.2 chunks/s.
from style import *

SOURCE = [
    "SERVICES AGREEMENT   Ref 2031-02-0001",
    "Between Kaveri Analytics Private Limited, represented by",
    "Lakshmi Varma, and Asterfield Research Ltd (the supplier).",
    "Customer notice address: 114 Malabar Lane, Kochi.",
    "Supplier contact: Haruto Ito, Tokyo office.",
]
ENTITIES = [
    ("Organization", 1, "Kaveri Analytics Private Limited"),
    ("Organization", 2, "Asterfield Research Ltd"),
    ("Person", 2, "Lakshmi Varma"),
    ("Person", 4, "Haruto Ito"),
    ("Street", 3, "114 Malabar Lane"),
    ("City", 3, "Kochi"),
    ("City", 4, "Tokyo"),
]
OLD_PASSES = [["Organization"], ["Person", "Street", "City"], ["City"]]
LABELS = ["Person", "Organization", "Street", "City"]


class ReadOnce(Clip):
    def construct(self):
        lines = VGroup(*(words(line, size=21, color=INK_DIM) for line in SOURCE))
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        page = panel(lines.width + 0.9, lines.height + 0.9).to_edge(LEFT, buff=0.6).shift(DOWN * 0.15)
        lines.move_to(page)
        self.lines, self.page, self.bright = lines, page, []

        heading = words("Before: one encoder pass per group of labels", size=24, color=INK_DIM)
        heading.to_corner(UL, buff=0.6)

        passes = VGroup(*(VGroup(*(chip(label, LABEL_COLORS[label]) for label in group)).arrange(RIGHT, buff=0.12)
                          for group in OLD_PASSES))
        passes.arrange(RIGHT, buff=0.55).next_to(page, UP, buff=0.35).align_to(page, LEFT)
        pass_numbers = VGroup(*(words(f"pass {i + 1}", size=18, color=INK_FAINT).next_to(group, UP, buff=0.12)
                                for i, group in enumerate(passes)))

        meter = self.meter().next_to(page, RIGHT, buff=0.5).align_to(page, UP)
        self.rate_title.next_to(self.track, DOWN, buff=0.7, aligned_edge=LEFT)
        layout = VGroup(page, lines, passes, pass_numbers, meter, self.rate_title)
        layout.shift(RIGHT * (-6.3 - page.get_left()[0]))
        heading.align_to(page, LEFT)
        self.rate_group.update()

        self.beat(FadeIn(page, scale=0.97), LaggedStart(*(FadeIn(line, shift=0.2 * UP) for line in lines),
                                                         lag_ratio=0.12), run_time=1.6)
        self.beat(FadeIn(heading, shift=0.2 * DOWN),
                  LaggedStart(*(FadeIn(group, shift=0.25 * DOWN) for group in passes), lag_ratio=0.25),
                  FadeIn(pass_numbers), run_time=1.4)
        self.beat(FadeIn(meter, shift=0.2 * LEFT))
        self.pause()

        found = {}
        for index, group in enumerate(OLD_PASSES):
            self.beat(*(chip_level(passes[i], 1 if i == index else 0.3) for i in range(len(passes))),
                      pass_numbers[index].animate.set_color(INK), run_time=0.6)
            self.sweep(run_time=1.2)
            boxes = [self.entity_box(label, line, phrase) for label, line, phrase in ENTITIES if label in group]
            fresh = [b for b, key in zip(boxes, [(l, p) for l, _, p in ENTITIES if l in group]) if key not in found]
            again = [found[(l, p)] for l, _, p in ENTITIES if l in group and (l, p) in found]
            animations = [self.add_segment(index)]
            if fresh:
                animations.append(LaggedStart(*(DrawBorderThenFill(b) for b in fresh), lag_ratio=0.15))
                animations.append(self.light([e for e in ENTITIES if e[0] in group]))
            if again:
                animations += [Indicate(b, color=WASTE, scale_factor=1.06) for b in again]
            self.beat(*animations, run_time=1.0)
            for b, (label, _, phrase) in zip(boxes, [e for e in ENTITIES if e[0] in group]):
                found.setdefault((label, phrase), b)
            self.beat(pass_numbers[index].animate.set_color(INK_FAINT), run_time=0.3)

        self.rate.set_value(59)
        self.rate_group.update()
        self.beat(*(chip_level(group, 1) for group in passes), FadeIn(self.rate_group, shift=0.2 * UP), run_time=1.0)
        waste = words("same page, read 3×", size=20, color=WASTE)
        waste.next_to(self.track, DOWN, buff=0.12).align_to(self.track, LEFT)
        self.beat(FadeIn(waste, shift=0.15 * UP))
        self.pause(1.2)

        merged = VGroup(*(chip(label, LABEL_COLORS[label]) for label in LABELS)).arrange(RIGHT, buff=0.12)
        merged.next_to(page, UP, buff=0.35).align_to(page, LEFT)
        after = words("After: one pass, every label in the same prompt", size=24, color=INK)
        after.move_to(heading, aligned_edge=LEFT)
        old_chips = {(i, label): passes[i][j] for i, group in enumerate(OLD_PASSES) for j, label in enumerate(group)}
        keep = {"Organization": (0, "Organization"), "Person": (1, "Person"), "Street": (1, "Street"), "City": (1, "City")}
        moves = [ReplacementTransform(old_chips[keep[label]], merged[k]) for k, label in enumerate(LABELS)]
        self.beat(*moves, FadeOut(old_chips[2, "City"], shift=0.3 * LEFT), FadeOut(pass_numbers),
                  FadeOut(VGroup(*found.values())), FadeOut(waste), self.unlight(),
                  ReplacementTransform(heading, after), run_time=1.4)
        one = self.segments[0].copy()
        self.beat(ReplacementTransform(self.segments, one), self.multiplier.animate.become(
            words("×1", size=24, color=GPU, weight=MEDIUM).move_to(self.multiplier, aligned_edge=LEFT)), run_time=1.0)
        self.segments = one
        self.pause(0.4)

        self.sweep(run_time=1.2)
        boxes = [self.entity_box(label, line, phrase) for label, line, phrase in ENTITIES]
        self.beat(LaggedStart(*(DrawBorderThenFill(b) for b in boxes), lag_ratio=0.08), self.light(ENTITIES),
                  run_time=1.2)
        self.beat(self.show_rate(142), run_time=1.6)
        gain = chip("2.4× faster", NUMBER, size=24).next_to(self.rate_group, DOWN, buff=0.3)
        gain.align_to(self.rate_group, LEFT)
        same = words("same accuracy", size=20, color=INK_DIM).next_to(gain, DOWN, buff=0.15, aligned_edge=LEFT)
        self.beat(FadeIn(gain, scale=1.15), FadeIn(same))
        self.pause(0.6)

        claim = caption("Read the document once, and ask all the questions.")
        note = words("Tesla T4 · realistic corpus · 59 → 142 chunks/s", size=20, color=INK_DIM)
        note.next_to(claim, DOWN, buff=0.15)
        self.beat(FadeIn(claim, shift=0.2 * UP), FadeIn(note), run_time=1.2)
        self.wait(HOLD)
        self.beat(*(FadeOut(mob) for mob in self.mobjects), run_time=1.0)

    def meter(self):
        title = words("GPU work per page", size=20, color=INK_DIM)
        track = RoundedRectangle(width=2.6, height=0.42, corner_radius=0.08).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5)
        track.next_to(title, DOWN, buff=0.18, aligned_edge=LEFT)
        self.track = track
        self.segments = VGroup()
        self.multiplier = words("×0", size=24, color=INK_FAINT, weight=MEDIUM).next_to(track, RIGHT, buff=0.2)
        rate_title = self.rate_title = words("throughput", size=20, color=INK_DIM)
        self.rate = ValueTracker(0)
        number = always_redraw(lambda: words(f"{self.rate.get_value():.0f}", size=64, color=NUMBER, weight=BOLD,
                                             font=MONO)
                               .next_to(rate_title, DOWN, buff=0.12, aligned_edge=LEFT))
        unit = always_redraw(lambda: words("chunks / s", size=22, color=INK_DIM).next_to(number, RIGHT, buff=0.2)
                             .align_to(number, DOWN))
        self.rate_group = VGroup(rate_title, number, unit)
        return VGroup(title, track, self.multiplier)

    def add_segment(self, index):
        width = (self.track.width - 0.16 - 0.06 * 2) / 3
        segment = RoundedRectangle(width=width, height=0.3, corner_radius=0.05).set_fill(GPU, 0.9).set_stroke(width=0)
        segment.move_to(self.track.get_left() + RIGHT * (0.08 + width / 2 + index * (width + 0.06)))
        self.segments.add(segment)
        label = words(f"×{index + 1}", size=24, color=GPU, weight=MEDIUM).move_to(self.multiplier, aligned_edge=LEFT)
        return AnimationGroup(GrowFromEdge(segment, LEFT), self.multiplier.animate.become(label))

    def show_rate(self, value):
        return self.rate.animate.set_value(value)

    def light(self, entities):
        glyphs = [span(self.lines[line], SOURCE[line], phrase) for _, line, phrase in entities]
        self.bright += glyphs
        return AnimationGroup(*(g.animate.set_color(INK) for g in glyphs))

    def unlight(self):
        glyphs, self.bright = self.bright, []
        return AnimationGroup(*(g.animate.set_color(INK_DIM) for g in glyphs))

    def entity_box(self, label, line, phrase):
        return box(span(self.lines[line], SOURCE[line], phrase), LABEL_COLORS[label], fill=0.2, buff=0.05)

    def sweep(self, run_time):
        page, lines = self.page, self.lines
        bar = Rectangle(width=page.width - 0.3, height=0.05).set_fill(GPU, 1).set_stroke(width=0)
        glow = Rectangle(width=page.width - 0.3, height=0.55).set_stroke(width=0)
        glow.set_fill(color=[GPU, BACKGROUND], opacity=0.22).set_sheen_direction(DOWN)
        scanner = VGroup(glow, bar)
        glow.next_to(bar, UP, buff=0)
        scanner.move_to(page.get_top() + DOWN * 0.62, aligned_edge=DOWN)
        end = page.get_bottom() + UP * 0.1

        def read(group):
            y = bar.get_center()[1]
            for line in group:
                line.set_color(INK if line.get_center()[1] > y else INK_DIM)

        lines.add_updater(read)
        self.add(scanner)
        self.play(scanner.animate.move_to(end, aligned_edge=DOWN), run_time=run_time, rate_func=linear)
        lines.remove_updater(read)
        self.play(FadeOut(scanner), lines.animate.set_color(INK_DIM), run_time=0.35)
        if self.bright:
            self.play(*(g.animate.set_color(INK) for g in self.bright), run_time=0.25)
