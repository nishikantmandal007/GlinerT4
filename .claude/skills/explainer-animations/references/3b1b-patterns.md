# What 3Blue1Brown's scene code actually does

Source: `github.com/3b1b/videos` (CC BY-NC-SA 4.0 — read it, do not copy it), cloned 2026-10. Files read:
`_2017/nn/part1.py`, `_2024/transformers/{helpers,attention,embedding,network_flow,mlp}.py`. He uses ManimGL
(`InteractiveScene`, `self.frame`); the same ideas work in Manim Community Edition.

## Measured pacing (attention.py, embedding.py, network_flow.py, mlp.py)

| Measure | Value |
|---|---|
| `self.play` calls | 618 |
| `self.wait` calls | 386 → 0.62 pauses per animation |
| Explicit `run_time` | median 2.0 s, mean 3.35 s, 87% ≥ 2 s; key moments 4–6 s, ambient up to 20 s |
| `lag_ratio` | median 0.1; quartiles 0.1 / 0.5 |
| `TransformFromCopy` | 106 (things derived from things) |
| `FadeTransform` / `ReplacementTransform` | 62 / 21 |
| `set_opacity` / `fade(` | 126 / 20 (focus by dimming) |
| `frame.animate` (camera) | 131 (zoom to the detail, then back out) |
| `LaggedStartMap` | 147 (staggered reveals) |
| `SurroundingRectangle` / `Brace` | 80 / 40 |
| `Indicate` | 0 |
| `Text(` / `Tex(` / `Arrow(` | 177 / 130 / 91 |

Our first clips used ~1 s animations, almost no pauses and no camera or focus moves; they felt rushed and
were not self-explanatory. Use the numbers above as the default.

## Scene structure

- `construct()` is a list of named steps. `IntroduceEachLayer` (nn/part1.py): `setup_network_mob`,
  `break_up_image_as_neurons`, `show_activation_of_one_neuron`, `transform_into_full_network`, `show_output_layer`,
  `show_hidden_layers`, `show_propogation`. `HighLevelNetworkFlow` (network_flow.py): `show_initial_text_embedding`,
  `progress_through_attention_block`, `progress_through_mlp_block`, `mention_repetitions`, `focus_on_last_layer`,
  `show_unembedding`. Each step answers the question the previous one raised.
- A real example anchors everything: a handwritten 9 (MNIST), "The Computer History Museum is located in Mountain" with
  real next-token probabilities (Mountain 0.706, the 0.128, San 0.078 ...), "a fluffy blue creature roamed the verdant
  forest" for attention.

## Techniques, with where to see them

- **Concrete → abstract by transformation.** `break_up_image_as_neurons`: pixels of the 9 become circles
  (`ReplacementTransform(image_mob, neurons)`), braces label 28 and 28, copies of the brace labels fly into
  `28 × 28 = 784`.
- **One example, enlarged and played with.** `show_activation_of_one_neuron`: one neuron is copied out at large size
  with its activation number; the activation is changed and the number follows (`ChangingDecimal`).
- **Symbol ↔ substance link.** `DescribeAttentionEquation`: a small rectangle on `Q` in the formula, a big rectangle
  around the concrete query vectors, two lines joining corners; then the vectors are rearranged into a grid and each
  cell becomes a dot product by moving copies of a query and a key into it (`LaggedStartMap(MoveToTarget, dot_prods)`).
- **Keep history, dimmed.** `progress_through_attention_block`: the old layer goes to 25% opacity while a copy is
  transformed into the new layer inside the block; the camera moves in the same `play`.
- **Influence as flowing light.** `ContextAnimation` (helpers.py): curved arcs from source words to a target word,
  shown with `VShowPassingFlash`, stroke width = strength.
- **Values as colour.** `value_to_color`: positive values blue (BLUE_E → BLUE_B), negative red, brightness by
  magnitude; used for matrix entries and network connections (`get_network_connections`).
- **Data in motion.** `data_flying_animation`: the word "Data" drifts across a matrix while its entries re-randomise
  (`RandomizeMatrixEntries`), showing "the matrix is being trained/used" without an equation.
- **Highlight one item in a row.** Scale it up about its bottom edge (`scale(2, about_edge=DOWN)`), shift the neighbours
  aside, draw a rectangle and an arrow to its vector.
- **Counting.** `CountInFrom` for dimensions (12,288), `Brace` + label for sizes, `DecimalNumber` for values.
- **Paragraph text** is wrapped by word count (`get_paragraph`, 40 characters per line) and faded from 25% to 100% to
  show "reading" (`MysteryNovel`).

## What he does not do

- No decorative motion; no colour used for variety; no title cards mid-scene.
- No `Indicate` pulses to say "look here" — he dims everything else or zooms in.
- No numbers appearing as finished text when the change matters — they count.
