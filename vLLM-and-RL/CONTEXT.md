# Deck Work Context

This file is a helper for future agents. It is **not** the source of truth.

Source of truth:

- `storyline.md`: talk scope, current storyboard, timings, slide intent.
- `vllm_rl_meetup_preview.html`: current generated HTML preview deck.

## Current State

The preview deck currently implements the main deck from `storyline.md`, with the one-sentence slide and the standalone rollout data contract hidden in the HTML preview:

- Title
- Nebius Overview
- Topics
- From SFT to RLVR
- RLVR core algorithm: GRPO
- RL Training Loop
- Naive RL Loop
- Why Naive Breaks
- Sync Loop: Keep vLLM Alive
- What vLLM Needs in the Sync World
- Sync Fixes IO, But Creates Idle GPUs
- Async Loop: Keep Both Sides Busy
- Naive -> Sync -> Async recap
- Sync vs Async Tradeoff
- Where to Continue Learning
- How We Do RL at Nebius AI R&D
- Token Factory RFT Beta
- Questions

The main deck is implemented through the final questions slide. Appendix slides are not implemented yet.

## Style Notes

Style is based on `2025_12_12_AiDevTLV_Karasik.pptx` and local reference `vllm_rl_meetup_nebius_reference_style.html`.

Keep:

- Nebius-like large typography, navy panels, simple geometry.
- Lime `#E0FF4F` as an accent, not the dominant background everywhere.
- Large stage-readable text.

Avoid:

- Tiny web-UI text.
- Extra decorative rectangle grids.
- Oversized arrowheads.
- Low-value meta labels like `Nebius Overview` as tiny kickers when the slide title already explains the content.

## Important Decisions

- GRPO formula is appendix only. Mainline slide explains GRPO intuition visually.
- Rollout data contract is integrated into the GRPO and RL loop diagrams. The old standalone contract slide remains in the HTML with `data-hidden="true"`.
- Slide 9 has an in-slide fragment:
  - first show the full naive loop: sample tasks, generate rollouts, score rewards, train, save checkpoint, reload inference
  - next key press emphasizes `Save checkpoint` and `Reload inference`
  - next key press moves to Slide 10
- Keep `storyline.md` synchronized with meaningful deck changes.

## Navigation Implementation

`vllm_rl_meetup_preview.html` uses transform-based navigation, not browser scrolling/hash navigation.

Reason: `file://` preview produced security-origin warnings and scroll/hash navigation broke arrow/space transitions.

Controls:

- next: `ArrowRight`, `ArrowDown`, `PageDown`, `Space`, mouse wheel down
- previous: `ArrowLeft`, `ArrowUp`, `PageUp`, mouse wheel up
- iPad/touch: swipe up or left for next; swipe down or right for previous
- `Home`, `End`

For future in-slide transitions:

- Add `data-fragments="N"` to the `<section class="slide">`.
- Style based on `.slide[data-fragment="1"] ...`.
- JS already supports fragment navigation.

Page anchors are supported:

- `#9` opens slide 9 after reload.
- The URL hash is updated automatically as navigation changes slides.

## Known Inputs

Token Factory copy:

- **Research-proven RL fine-tuning as a service.**

Token Factory RFT beta form:

- https://forms.cloud.microsoft/e/BmcYJQ3dv3

Useful-links gist for QR codes:

- https://gist.github.com/minotru/8bd5aebf1e1aec5a32f463553072a563

Candidate performance numbers:

- InfiniBand: around `400 GB/s per host`
- Disk: write around `3 GB/s` + read around `3 GB/s`
- Confirm InfiniBand units before final deck.

Sync/async benchmark:

- basic math RLVR, small context
- sync: `1200 tok/sec`
- async-k1: `2500 tok/sec`
- async-k3: `3300 tok/sec`
- async-k6: `3700 tok/sec`

Links for learning slide:

- rLLM: https://github.com/rllm-org/rllm
- veRL: https://github.com/verl-project/verl
- vLLM TRL: https://docs.vllm.ai/en/stable/training/trl/
- vLLM weight transfer: https://docs.vllm.ai/en/stable/training/weight_transfer/
