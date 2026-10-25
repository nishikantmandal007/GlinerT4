# Clip C: "Overlap means no entity is ever only seen cut in half."
# Chapters: 1 the model reads a long document one window at a time; 2 fixed chunks cut a name in half and each half is
# judged alone; 3 overlapping windows repeat the last 64 words, so one window holds the whole name; 4 the merge keeps one
# span, and the measured result and cost.
# Numbers: realistic corpus (040 sections), round 9 of benchmark_results/pipeline_2026-10-25: 51 annotated names cut by
# pre-cut chunk boundaries, 0 with GLiNER2's 384-word windows overlapping by 64; windowing costs +22% to +27% time.
from style import *

FILLER = [
    "This services agreement is made between the customer and the supplier named below.",
    "The supplier will provide the services described in the approved schedule of work.",
    "Each deliverable is reviewed against the acceptance criteria in that schedule.",
    "Fees follow the rates and quantities in the approved order for each period.",
    "Invoices identify the service period, the order reference and any variations.",
    "Disputed lines are held while the parties reconcile the supporting evidence.",
    "Both parties keep the records needed to reconstruct an approval decision.",
    "Personnel with the right experience are assigned to the scheduled activities.",
    "Access to customer systems is limited to the approved working purpose only.",
    "Information supplied during delivery is used only for the agreed purpose.",
    "During the term, notices for the customer go to Lakshmi Varma at the Kochi office.",
    "A proposed change describes the affected deliverable and its revised window.",
    "Dependencies that threaten the plan are recorded promptly by either party.",
    "The representatives review each variation before it is authorised in writing.",
    "Escalations name the service reference and the response that is requested.",
    "The receiving party acknowledges each notice and assigns an owner to it.",
    "Operational questions are first reviewed by the delivery coordinators.",
    "Unresolved questions go to the authorised representatives with evidence.",
]
NAME_LINE = 10
SOURCE = "…notices for the customer go to Lakshmi Varma at the Kochi office…"


class Overlap(Clip):
    def construct(self):
        self.reading_in_windows()
        self.cutting_without_overlap()
        self.overlapping_windows()
        self.merge_and_result()
        self.finale("Overlap means no entity is ever only seen cut in half.",
                    "realistic test corpus · GLiNER2 windows of 384 words, 64 overlap")

    def reading_in_windows(self):
        self.chapter(1, 4, "A long document, read one window at a time")
        page = panel(9.2, 4.75).move_to(UP * 0.45)
        lines = VGroup(*(words(t, size=12, color=INK_DIM) for t in FILLER)).arrange(DOWN, aligned_edge=LEFT, buff=0.11)
        lines.move_to(page)
        pitch = lines[0].get_center()[1] - lines[1].get_center()[1]
        self.page, self.lines = page, lines
        self.say("A contract can run to 50,000 words. Here is a small piece of one.",
                 FadeIn(page), LaggedStart(*(FadeIn(l, shift=0.05 * UP) for l in lines), lag_ratio=0.04))
        self.read()

        window = Rectangle(width=page.width - 0.3, height=6 * pitch).set_stroke(GPU, 3).set_fill(GPU, 0.07)
        window.move_to(lines[0:6]).align_to(lines[0], UP).shift(UP * 0.06)
        tag = words("one window", size=18, color=GPU).next_to(window, RIGHT, buff=0.15).align_to(window, UP)
        self.say("GLiNER2 reads at most about 384 words at a time: one window.",
                 Create(window), FadeIn(tag, shift=0.1 * LEFT))
        self.read()
        self.say("So a long document is read window by window, and each window is judged on its own.",
                 VGroup(window, tag).animate.shift(DOWN * 6 * pitch), run_time=2.4)
        self.beat(VGroup(window, tag).animate.shift(DOWN * 6 * pitch), run_time=1.6)
        self.read()
        self.beat(FadeOut(window), FadeOut(tag), run_time=0.8)

    def cutting_without_overlap(self):
        self.chapter(2, 4, "Pre-cut chunks: a fixed word count")
        line = words(SOURCE, size=30, color=INK_DIM).move_to(UP * 1.55)
        self.line = line
        source_line = self.lines[NAME_LINE]
        self.say("Look closely at one line of it.",
                 self.lines.animate.set_opacity(0.15), source_line.animate.set_color(INK).set_opacity(1))
        self.beat(Derive(source_line, line), FadeOut(self.page), FadeOut(self.lines), run_time=1.8)
        first, last = span(line, SOURCE, "Lakshmi"), span(line, SOURCE, "Varma")
        self.first, self.last = first, last
        self.beat(VGroup(first, last).animate.set_color(INK), run_time=0.8)
        self.read()

        self.cut_x = cut_x = (first.get_right()[0] + last.get_left()[0]) / 2
        left_edge, right_edge = line.get_left()[0] - 0.1, line.get_right()[0] + 0.1
        self.edges = (left_edge, right_edge)
        y = line.get_bottom()[1] - 0.3
        chunks = VGroup(bracket(left_edge, cut_x - 0.05, y, up=False), bracket(cut_x + 0.05, right_edge, y, up=False))
        names = VGroup(words("chunk 1", size=19, color=INK_DIM).next_to(chunks[0], DOWN, buff=0.1),
                       words("chunk 2", size=19, color=INK_DIM).next_to(chunks[1], DOWN, buff=0.1))
        cut = DashedLine([cut_x, line.get_top()[1] + 0.3, 0], [cut_x, y - 0.1, 0], dash_length=0.08).set_stroke(ERROR, 3)
        self.say("The old pipeline cut the text into chunks every N words, wherever that landed.",
                 Create(chunks[0]), Create(chunks[1]), FadeIn(names))
        self.read()
        self.say("Here the cut falls between “Lakshmi” and “Varma”.", Create(cut), run_time=1.4)
        self.read()

        views = VGroup()
        for title, text in [("chunk 1 sees", "…for the customer go to Lakshmi"), ("chunk 2 sees", "Varma at the Kochi office…")]:
            body = words(text, size=22, color=INK_DIM)
            frame = panel(body.width + 0.7, 1.15)
            head = words(title, size=17, color=INK_DIM).next_to(frame, UP, buff=0.1).align_to(frame, LEFT)
            views.add(VGroup(frame, body.move_to(frame), head))
        views.arrange(RIGHT, buff=0.6).move_to(DOWN * 0.3)
        sources = (span(line, SOURCE, "for the customer go to Lakshmi"), span(line, SOURCE, "Varma at the Kochi office"))
        self.say("The model never sees the two halves together. Each chunk is judged alone.",
                 *(Derive(part, view[1]) for part, view in zip(sources, views)),
                 *(FadeIn(VGroup(v[0], v[2])) for v in views), run_time=2.2)
        self.read()
        lakshmi = span(views[0][1], "…for the customer go to Lakshmi", "Lakshmi")
        varma = span(views[1][1], "Varma at the Kochi office…", "Varma")
        half = box(lakshmi, ERROR, fill=0.16)
        missed = DashedVMobject(SurroundingRectangle(varma, buff=0.07, corner_radius=0.06), num_dashes=24)
        missed.set_stroke(ERROR, 2)
        notes = VGroup(words("“Lakshmi”: half a name", size=18, color=ERROR).next_to(views[0], DOWN, buff=0.15),
                       words("“Varma” alone: missed", size=18, color=ERROR).next_to(views[1], DOWN, buff=0.15))
        self.say("Chunk 1 reports half a name. Chunk 2 sees “Varma” without its first name and misses it.",
                 DrawBorderThenFill(half), Create(missed), FadeIn(notes, shift=0.1 * UP), run_time=2.2)
        self.read()

        self.tally = ValueTracker(0)
        self.tally_color = ERROR
        number = counter(self.tally, size=54, color=lambda: self.tally_color)
        label = words("names cut in half", size=22, color=INK_DIM)
        number.add_updater(lambda m: m.move_to(DOWN * 1.6 + LEFT * 1.4))
        label.add_updater(lambda m: m.next_to(number, RIGHT, buff=0.25).align_to(number, DOWN))
        number.update()
        label.update()
        self.number, self.number_label = number, label
        self.beat(FadeOut(views), FadeOut(half), FadeOut(missed), FadeOut(notes), run_time=0.8)
        self.add(number)
        self.say("In our ten test documents, 51 names were cut like this.",
                 FadeIn(label), self.tally.animate.set_value(51), run_time=2.4)
        self.read()
        self.chunk_marks = VGroup(chunks, names, cut)

    def overlapping_windows(self):
        self.chapter(3, 4, "Overlapping windows")
        line, (left_edge, right_edge) = self.line, self.edges
        top = line.get_top()[1]
        w1_right = self.cut_x - 0.05
        w2_left = span(line, SOURCE, "customer").get_left()[0] - 0.1
        window_1 = bracket(left_edge, w1_right, top + 0.45)
        window_2 = bracket(w2_left, right_edge, top + 0.95)
        heads = VGroup(words("window 1", size=19, color=INK_DIM).next_to(window_1, UP, buff=0.08).align_to(window_1, LEFT),
                       words("window 2", size=19, color=INK_DIM).next_to(window_2, UP, buff=0.08).align_to(window_2, RIGHT))
        self.say("GLiNER2's own long-document mode does it differently: its windows overlap.",
                 FadeOut(self.chunk_marks), Create(window_1), FadeIn(heads[0]))
        self.read()
        overlap = Rectangle(width=w1_right - w2_left, height=line.height + 0.3).set_stroke(width=0)
        overlap.set_fill(GPU, 0.14).move_to([(w1_right + w2_left) / 2, line.get_center()[1], 0])
        overlap_tag = words("read twice", size=18, color=GPU).next_to(overlap, DOWN, buff=0.1)
        self.say("Each window starts 64 words before the previous one ends, so those words are read twice.",
                 Create(window_2), FadeIn(heads[1]), FadeIn(overlap), FadeIn(overlap_tag), run_time=2.4)
        self.read()

        text = "customer go to Lakshmi Varma at the Kochi office…"
        body = words(text, size=22, color=INK_DIM)
        frame = panel(body.width + 0.7, 1.15)
        head = words("window 2 sees", size=17, color=INK_DIM).next_to(frame, UP, buff=0.1).align_to(frame, LEFT)
        view = VGroup(frame, body.move_to(frame), head).move_to(DOWN * 0.05)
        whole = span(body, text, "Lakshmi Varma")
        found = box(whole, CORRECT, fill=0.2)
        self.say("Window 2 starts early enough to hold the whole name, so it finds “Lakshmi Varma”.",
                 Derive(span(line, SOURCE, "customer go to Lakshmi Varma at the Kochi office"), body),
                 FadeIn(VGroup(frame, head)), run_time=2.2)
        self.beat(DrawBorderThenFill(found), whole.animate.set_color(INK), run_time=1.2)
        self.read()
        self.say("The rule: a name shorter than the 64-word overlap always fits whole inside some window.",
                 overlap.animate.set_fill(GPU, 0.28), run_time=1.4)
        self.read()
        self.window_parts = VGroup(window_1, window_2, heads, overlap, overlap_tag)
        self.view = VGroup(view, found)

    def merge_and_result(self):
        self.chapter(4, 4, "Merging, and the measured result")
        partial = box(self.first, WASTE, fill=0.12)
        full = box(VGroup(self.first, self.last), CORRECT, fill=0.2)
        tags = VGroup(words("from window 1", size=17, color=WASTE).next_to(partial, DOWN, buff=0.55),
                      words("from window 2", size=17, color=CORRECT).next_to(full, DOWN, buff=0.55))
        tags[0].shift(LEFT * 0.8)
        tags[1].shift(RIGHT * 0.9)
        self.say("Both windows report what they saw, mapped back onto the document.",
                 FadeOut(self.view), self.window_parts.animate.set_opacity(0.3),
                 DrawBorderThenFill(partial), DrawBorderThenFill(full), FadeIn(tags))
        self.read()
        self.say("Where reports overlap, the merge keeps one span: here, the whole name.",
                 FadeOut(partial, scale=0.9), FadeOut(tags[0]), run_time=1.6)
        self.read()
        self.say("Across the same ten documents, the names cut in half dropped from 51 to 0.",
                 self.tally.animate.set_value(0), run_time=2.4)
        self.tally_color = CORRECT
        self.beat(Flash(self.number, color=CORRECT, flash_radius=0.6, line_length=0.25), run_time=0.8)
        self.read()
        cost = words("cost: 64 of every 384 words are read twice, about 22% more time", size=20, color=INK_DIM)
        cost.next_to(VGroup(self.number, self.number_label), DOWN, buff=0.2)
        self.say("The price is re-reading the overlap: about a fifth more time for the whole document.",
                 FadeIn(cost, shift=0.1 * UP))
        self.read(extra=0.5)
