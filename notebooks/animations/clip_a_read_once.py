# Clip A: "Instead of reading the document three times, read it once and ask all the questions."
# Chapters: 1 the task (a line of a contract, four labels, the output we want); 2 how GLiNER2 reads: labels go into the
# prompt ahead of the text, the encoder turns every word into a vector (the expensive step), a light step matches labels
# to words; 3 before: three label groups, so the encoder reads the same text three times; 4 after: one prompt with every
# label, one encoder run, 2.4x the throughput with per-label thresholds.
# Numbers: Tesla T4, realistic corpus (290 chunks), round 8 of benchmark_results/pipeline_2026-10-25: 3 passes
# 59.0 chunks/s, single pass with per-label thresholds (Organization 0.85, Street 0.5) 142.2 chunks/s, no label worse.
from style import *

SOURCE = "Lakshmi Varma of Kaveri Analytics, 114 Malabar Lane, Kochi"
TOKENS = ["Lakshmi", "Varma", "of", "Kaveri", "Analytics", "114", "Malabar", "Lane", "Kochi"]
ENTITIES = [("Person", 0, 2), ("Organization", 3, 5), ("Street", 5, 8), ("City", 8, 9)]
OLD_PASSES = [["Organization"], ["Person", "Street", "City"], ["City"]]
LABELS = ["Person", "Organization", "Street", "City"]


def token(text):
    label = words(text, size=20, color=INK)
    back = RoundedRectangle(width=label.width + 0.24, height=0.46, corner_radius=0.08)
    back.set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5)
    return VGroup(back, label.move_to(back))


class ReadOnce(Clip):
    def construct(self):
        self.the_task()
        self.how_gliner_reads()
        self.three_passes()
        self.one_pass()
        self.finale("Read the document once, and ask all the questions.",
                    "Tesla T4 · realistic corpus · 59 → 142 chunks/s, no label worse")

    def the_task(self):
        self.chapter(1, 4, "The task")
        line = words(SOURCE, size=34, color=INK).move_to(UP * 1.6)
        self.say("A line from a contract. We want to find the personal data in it.",
                 LaggedStart(*(FadeIn(ch, shift=0.1 * UP) for ch in line), lag_ratio=0.02))
        self.read()
        chips = VGroup(*(chip(l, LABEL_COLORS[l]) for l in LABELS)).arrange(RIGHT, buff=0.18).move_to(UP * 2.55)
        self.say("Four kinds: people, organizations, streets and cities.",
                 LaggedStart(*(FadeIn(c, shift=0.15 * DOWN) for c in chips), lag_ratio=0.2))
        self.read()
        phrases = {"Person": "Lakshmi Varma", "Organization": "Kaveri Analytics", "Street": "114 Malabar Lane",
                   "City": "Kochi"}
        boxes = VGroup(*(box(span(line, SOURCE, phrases[l]), LABEL_COLORS[l], fill=0.2) for l in LABELS))
        self.say("This is the answer we want: every entity found, with the right label.",
                 LaggedStart(*(DrawBorderThenFill(b) for b in boxes), lag_ratio=0.25))
        self.read()

        tokens = VGroup(*(token(t) for t in TOKENS)).arrange(RIGHT, buff=0.1).move_to(UP * 1.6)
        self.tokens = tokens
        pieces = [span(line, SOURCE, t) for t in TOKENS]
        self.say("The model reads text as a row of pieces, roughly one per word.",
                 FadeOut(boxes), FadeOut(line), *(Derive(p, t) for p, t in zip(pieces, tokens)), run_time=2.2)
        self.read()
        self.chips = chips

    def how_gliner_reads(self):
        self.chapter(2, 4, "How GLiNER2 reads")
        tokens = self.tokens
        anchored = tokens.copy().to_edge(RIGHT, buff=0.65)
        prompt = VGroup(chip("Organization", LABEL_COLORS["Organization"], size=19)).next_to(anchored, LEFT, buff=0.35)
        self.say("The labels we ask for are written into the input, in front of the text.",
                 tokens.animate.move_to(anchored), Derive(self.chips[1], prompt[0]), FadeOut(self.chips), run_time=2.2)
        self.prompt = prompt
        self.read()

        encoder = RoundedRectangle(width=10.6, height=1.05, corner_radius=0.16).set_fill(GPU, 0.1).set_stroke(GPU, 2.5)
        encoder.move_to(UP * 0.15)
        encoder_label = words("encoder: mDeBERTa, 12 layers", size=22, color=GPU, weight=MEDIUM).move_to(encoder)
        self.encoder = VGroup(encoder, encoder_label)
        self.say("The encoder reads the whole input at once. It is the expensive part: almost all the GPU time.",
                 FadeIn(self.encoder, shift=0.1 * UP))
        self.read()
        vectors = self.encode(VGroup(prompt, tokens))
        self.say("Out comes one vector per piece: what each word means here, with the labels in view.", run_time=1.0)
        self.read()

        match = box(VGroup(*tokens[3:5]), LABEL_COLORS["Organization"], fill=0.22)
        arc = ArcBetweenPoints(prompt.get_top(), match.get_top(), angle=-PI / 3)
        arc.set_stroke(LABEL_COLORS["Organization"], 2.5)
        self.say("A light step then scores each label against the word vectors, and marks the matches.",
                 Create(arc), DrawBorderThenFill(match), run_time=2.2)
        self.read()
        self.beat(FadeOut(VGroup(arc, match, vectors)), run_time=0.8)

    def encode(self, row, run_time=1.6):
        drops = VGroup(*(m.copy() for m in row))
        bars = VGroup()
        for n, t in enumerate(self.tokens):
            bar = VGroup(*(Square(0.13).set_stroke(width=0).set_fill(GPU, 0.3 + 0.65 * (((n * 7 + k * 3) % 7) / 6))
                           for k in range(5))).arrange(DOWN, buff=0.035)
            bars.add(bar.next_to(self.encoder, DOWN, buff=0.28).match_x(t))
        self.beat(LaggedStart(*(m.animate.move_to(self.encoder).set_opacity(0) for m in drops), lag_ratio=0.05),
                  self.encoder[0].animate.set_fill(GPU, 0.45), run_time=run_time)
        self.remove(drops)
        self.beat(self.encoder[0].animate.set_fill(GPU, 0.1),
                  LaggedStart(*(GrowFromEdge(b, UP) for b in bars), lag_ratio=0.05), run_time=run_time * 0.8)
        return bars

    def meter(self):
        title = words("encoder runs", size=20, color=INK_DIM).move_to(LEFT * 4.6 + DOWN * 2.0, aligned_edge=LEFT)
        rate_title = words("throughput", size=20, color=INK_DIM).move_to(RIGHT * 0.9 + DOWN * 2.0, aligned_edge=LEFT)
        self.runs = ValueTracker(0)
        count = counter(self.runs, size=40, color=GPU, fmt="×{:.0f}")
        count.add_updater(lambda m: m.next_to(title, RIGHT, buff=0.25))
        self.rate = ValueTracker(0)
        rate = counter(self.rate, size=40, color=NUMBER)
        unit = words("chunks/s", size=18, color=INK_DIM)
        rate.add_updater(lambda m: m.next_to(rate_title, RIGHT, buff=0.25))
        unit.add_updater(lambda m: m.next_to(rate, RIGHT, buff=0.15).align_to(rate, DOWN))
        for m in (count, rate, unit):
            m.update()
        return title, count, rate_title, rate, unit

    def place_prompt(self, prompt):
        if len(prompt) > 2:
            prompt.arrange_in_grid(cols=2, buff=(0.08, 0.08), cell_alignment=RIGHT)
        return prompt.next_to(self.tokens, LEFT, buff=0.35)

    def three_passes(self):
        self.chapter(3, 4, "Before: three passes")
        tokens = self.tokens
        title, count, rate_title, rate, unit = self.meter()
        self.say("The old setup asked for the labels in three groups.", FadeIn(title))
        self.add(count)
        self.read()
        found, boxes = {}, VGroup()
        texts = ["Pass 1: ask for Organization.", "Pass 2: the same text again, for the other three labels.",
                 "Pass 3: the same text a third time, for cities alone."]
        for n, group in enumerate(OLD_PASSES):
            prompt = VGroup(*(chip(l, LABEL_COLORS[l], size=19) for l in group)).arrange(RIGHT, buff=0.08)
            self.place_prompt(prompt)
            self.say(texts[n], FadeOut(self.prompt, shift=0.2 * UP), FadeIn(prompt, shift=0.2 * UP), run_time=1.4)
            self.prompt = prompt
            vectors = self.encode(VGroup(prompt, tokens), run_time=1.6 if n == 0 else 1.2)
            self.beat(self.runs.animate.set_value(n + 1), run_time=0.6)
            marks = []
            for label, start, end in ENTITIES:
                if label not in group:
                    continue
                if label in found:
                    marks.append(found[label].animate.set_stroke(WASTE).set_fill(WASTE, 0.25))
                else:
                    found[label] = box(VGroup(*tokens[start:end]), LABEL_COLORS[label], fill=0.22)
                    boxes.add(found[label])
                    marks.append(DrawBorderThenFill(found[label]))
            self.beat(*marks, FadeOut(vectors), run_time=1.2)
            self.read()
        self.say("Same text, encoded three times. The third pass found nothing new.", run_time=1.0)
        self.read()
        self.add(rate)
        self.say("That setup ran at 59 chunks per second on one T4.",
                 FadeIn(rate_title), FadeIn(unit), self.rate.animate.set_value(59), run_time=2.0)
        self.read()
        self.old_boxes = boxes

    def one_pass(self):
        self.chapter(4, 4, "After: one pass")
        tokens = self.tokens
        prompt = VGroup(*(chip(l, LABEL_COLORS[l], size=19) for l in LABELS)).arrange(RIGHT, buff=0.08)
        self.place_prompt(prompt)
        self.say("Now every label goes into one prompt.",
                 FadeOut(self.prompt, shift=0.2 * UP), FadeIn(prompt, shift=0.2 * UP), FadeOut(self.old_boxes),
                 self.runs.animate.set_value(0), run_time=2.0)
        self.prompt = prompt
        self.read()
        vectors = self.encode(VGroup(prompt, tokens), run_time=1.8)
        self.beat(self.runs.animate.set_value(1), run_time=0.6)
        boxes = VGroup(*(box(VGroup(*tokens[s:e]), LABEL_COLORS[l], fill=0.22) for l, s, e in ENTITIES))
        self.say("One encoder run. The light matching step answers all four questions from the same vectors.",
                 LaggedStart(*(DrawBorderThenFill(b) for b in boxes), lag_ratio=0.2), FadeOut(vectors), run_time=2.2)
        self.read()
        self.say("Throughput rose from 59 to 142 chunks per second: 2.4 times as fast.",
                 self.rate.animate.set_value(142), run_time=2.4)
        self.read()
        note = words("thresholds per label (Organization 0.85, Street 0.5): every label as accurate or better",
                     size=19, color=INK_DIM).move_to(DOWN * 1.0)
        self.say("Accuracy held: with a threshold per label, no label got worse on our test corpus.",
                 FadeIn(note, shift=0.1 * UP))
        self.read(extra=0.5)
