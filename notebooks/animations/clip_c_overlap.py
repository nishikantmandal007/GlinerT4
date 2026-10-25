# Clip C: "Overlap means no entity is ever only seen cut in half."
# Beats: a line of a long document with a name in it; fixed chunks cut through the name and both halves come out wrong;
# rewind; overlapping windows slide along and one of them holds the whole name; the merge keeps one span; 51 -> 0.
# Numbers: realistic corpus (040 sections), round 9: 51 annotated names cut by pre-cut chunk boundaries; 0 with
# GLiNER2's 384-word windows overlapping by 64 words.
from style import *

SOURCE = "…notices for the customer go to Lakshmi Varma at the Kochi office…"
NAME = "Lakshmi Varma"


class Overlap(Clip):
    def construct(self):
        line = words(SOURCE, size=30, color=INK_DIM).move_to(UP * 0.45)
        first, last = span(line, SOURCE, "Lakshmi"), span(line, SOURCE, "Varma")
        name = VGroup(first, last)
        cut_x = (first.get_right()[0] + last.get_left()[0]) / 2

        heading = words("Pre-cut chunks", size=26, color=INK).to_corner(UL, buff=0.6)
        sub = words("the text is split every N words before the model sees it", size=20, color=INK_DIM)
        sub.next_to(heading, DOWN, buff=0.12, aligned_edge=LEFT)

        self.beat(LaggedStart(*(FadeIn(ch, shift=0.15 * UP) for ch in line), lag_ratio=0.01), run_time=1.6)
        self.beat(name.animate.set_color(INK), run_time=0.6)
        self.pause(0.4)

        left_edge, right_edge = line.get_left()[0] - 0.1, line.get_right()[0] + 0.1
        y = line.get_bottom()[1] - 0.35
        chunk_a = bracket(left_edge, cut_x - 0.05, y, up=False)
        chunk_b = bracket(cut_x + 0.05, right_edge, y, up=False)
        labels = VGroup(words("chunk 1", size=20, color=INK_DIM).next_to(chunk_a, DOWN, buff=0.12),
                        words("chunk 2", size=20, color=INK_DIM).next_to(chunk_b, DOWN, buff=0.12))
        cut = DashedLine(UP * (line.get_top()[1] + 0.3) + RIGHT * cut_x, UP * (y - 0.1) + RIGHT * cut_x,
                         dash_length=0.08).set_stroke(ERROR, 3)
        self.beat(FadeIn(heading, shift=0.2 * DOWN), FadeIn(sub, shift=0.2 * DOWN), run_time=0.8)
        self.beat(Create(chunk_a), Create(chunk_b), FadeIn(labels), run_time=1.0)
        self.beat(Create(cut), run_time=0.8)

        halves = VGroup(box(first, ERROR, fill=0.16, buff=0.06), box(last, ERROR, fill=0.16, buff=0.06))
        tags = VGroup(words("“Lakshmi”: half a name", size=20, color=ERROR).next_to(halves[0], UP, buff=0.55),
                      words("“Varma”: missed", size=20, color=ERROR).next_to(halves[1], UP, buff=0.55))
        tags[0].shift(LEFT * 0.6)
        tags[1].shift(RIGHT * 0.6)
        arrows = VGroup(Arrow(tags[0].get_bottom(), halves[0].get_top(), buff=0.08, stroke_width=2.5,
                              max_tip_length_to_length_ratio=0.18).set_color(ERROR),
                        Arrow(tags[1].get_bottom(), halves[1].get_top(), buff=0.08, stroke_width=2.5,
                              max_tip_length_to_length_ratio=0.18).set_color(ERROR))
        self.beat(LaggedStart(DrawBorderThenFill(halves[0]), DrawBorderThenFill(halves[1]), lag_ratio=0.3),
                  LaggedStart(*(FadeIn(t, shift=0.1 * DOWN) for t in tags), lag_ratio=0.3),
                  LaggedStart(*(GrowArrow(a) for a in arrows), lag_ratio=0.3), run_time=1.2)

        tally = ValueTracker(0)
        self.tally_color = ERROR
        number = counter(tally, size=56, color=lambda: self.tally_color)
        caption_left = words("names cut in half", size=22, color=INK_DIM)
        score = VGroup(number, caption_left)
        number.add_updater(lambda m: m.move_to(DOWN * 1.75 + LEFT * 1.6))
        caption_left.add_updater(lambda m: m.next_to(number, RIGHT, buff=0.25).align_to(number, DOWN))
        corpus = words("in the realistic corpus", size=18, color=INK_FAINT)
        corpus.add_updater(lambda m: m.next_to(caption_left, DOWN, buff=0.08, aligned_edge=LEFT))
        number.update()
        caption_left.update()
        corpus.update()
        self.add(number)
        self.beat(FadeIn(caption_left), FadeIn(corpus), tally.animate.set_value(51), run_time=1.6)
        self.pause(1.0)

        heading2 = words("Overlapping windows", size=26, color=INK).move_to(heading, aligned_edge=LEFT)
        sub2 = words("each window repeats the end of the previous one (GLiNER2: 384 words, 64 overlap)", size=20,
                     color=INK_DIM).move_to(sub, aligned_edge=LEFT)
        self.beat(FadeOut(VGroup(halves, tags, arrows, cut, chunk_a, chunk_b, labels)),
                  ReplacementTransform(heading, heading2), ReplacementTransform(sub, sub2), run_time=1.0)

        top = line.get_top()[1]
        w1_right = cut_x - 0.05
        w2_left = span(line, SOURCE, "customer").get_left()[0] - 0.1
        window_1 = bracket(left_edge, w1_right, top + 0.45)
        window_2 = bracket(w2_left, right_edge, top + 0.95)
        overlap = Rectangle(width=w1_right - w2_left, height=line.height + 0.3).set_stroke(width=0)
        overlap.set_fill(GPU, 0.12).move_to([(w1_right + w2_left) / 2, line.get_center()[1], 0])
        names = VGroup(words("window 1", size=20, color=INK_DIM).next_to(window_1, UP, buff=0.1).align_to(window_1, LEFT),
                       words("window 2", size=20, color=INK_DIM).next_to(window_2, UP, buff=0.1).align_to(window_2, RIGHT))
        overlap_label = words("overlap", size=18, color=GPU).next_to(overlap, DOWN, buff=0.08).align_to(overlap, LEFT)

        self.beat(Create(window_1), FadeIn(names[0]), run_time=0.9)
        partial = box(first, WASTE, fill=0.1, buff=0.06)
        partial_tag = words("window 1 sees “Lakshmi”", size=20, color=WASTE).next_to(partial, DOWN, buff=0.55)
        self.beat(DrawBorderThenFill(partial), FadeIn(partial_tag, shift=0.1 * UP), run_time=0.8)
        self.beat(FadeIn(overlap), Create(window_2), FadeIn(names[1]), FadeIn(overlap_label), run_time=1.1)
        whole = box(name, CORRECT, fill=0.2, buff=0.06)
        whole_tag = words("window 2 sees the whole name", size=20, color=CORRECT).next_to(whole, DOWN, buff=0.55)
        self.beat(DrawBorderThenFill(whole), FadeOut(partial_tag), FadeIn(whole_tag, shift=0.1 * UP), run_time=1.0)
        self.pause(0.5)

        merge = words("merge: keep one span, the whole name", size=20, color=INK).move_to(whole_tag)
        self.beat(FadeOut(partial, scale=0.9), ReplacementTransform(whole_tag, merge), run_time=0.9)
        self.beat(tally.animate.set_value(0), run_time=1.4)
        self.tally_color = CORRECT
        self.beat(Flash(number, color=CORRECT, flash_radius=0.6, line_length=0.25), run_time=0.7)
        self.pause(0.6)

        claim = caption("Overlap means no entity is ever only seen cut in half.")
        self.beat(FadeIn(claim, shift=0.2 * UP), run_time=1.2)
        self.wait(HOLD)
        self.beat(*(FadeOut(mob) for mob in self.mobjects), run_time=1.0)
