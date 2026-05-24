# Closing the Loop: vLLM in RL Training Systems

## Spec

URL: https://luma.com/9ihwxsej?utm_source=chatgpt.com

Duration: 30 minutes with questions

Format: stage talk

Title: Closing the Loop: vLLM in Reinforcement Learning Training Systems

Where: Israel vLLM meetup #2, in Nebius office.

When: Wed May 27, 2026 18:20 – 18:50

Speakers: @Simon Karasik 

Based on: N/A

Message: vLLM is specially used in RL – btw, we’re releasing an RL platform.

Slides: TBD

Recording: N/A

Pitch:

> Modern RL training for language models requires a tight feedback loop between training and inference. In our platform, vLLM acts as the generation engine inside large-scale RL pipelines, serving continuously updated model checkpoints while handling high-throughput sampling workloads.
> This talk covers the systems side of making that work in practice: fast weight synchronization, inference server lifecycle management, KV cache invalidation/reset strategies, orchestration of training and serving clusters, and the trade-offs we encountered when adapting vLLM for RL workloads.
> I’ll also discuss which capabilities came directly from vLLM, which required custom engineering, and where RL workloads stress inference systems differently from traditional serving.


## Storyline

Key idea. 1 day after the meetup, people should remember that:
- inference is now a part of RL post-training
- there are several techniques to make it fast: naive, sync, async
- this use case brings new features to vLLM -- people can study them

---


Legend:

- **Required**: mainline content for the 30-minute talk.
- **Optional**: one sentence, backup slide, or appendix.
- **Drop**: remove from mainline unless audience context changes.

Scope principle: the backbone is **naive -> sync -> async**. Everything else should support that ladder or move to the closing/resources section.

1. **Optional**: Nebius & Nebius AI R&D: quick overview
   1. Keep to 30-45 seconds. Enough context to explain why we are working on this.
2. **Required**: RL is now the frontier post-training method
   1. **Required**: What's RL. SFT -> RLHF -> RLVR
      1. Keep as one visual progression, not a full RL lecture.
   2. **Optional**: RLVR in LLMs: SoTA reasoning, math, coding
      1. Good motivation, but one sentence is enough.
   3. **Required**: GRPO idea, without the full formula
      1. The audience should understand the intuition: generate a group of completions, score them, compare within the group, and update toward better-than-group-average behavior.
      2. Avoid putting a large formula on the mainline slide.
      3. Use a diagram for this: prompt -> K completions -> rewards -> group-relative comparison -> update.
   4. **Drop / Appendix**: GRPO formulae
      1. Keep only as backup material if someone asks for algorithmic details.
   5. **Required**: What's passed: `(group_id, sample_id, token_ids, logprobs, reward)`
      1. Reframe as the rollout/training data contract, not as tuple trivia.
      2. Keep this as a separate slide after the GRPO intuition diagram.
3. **Required**: Naive GRPO training loop
   1. **Required**: The loop overview: sample tasks, rollout, evaluate, train.
   2. **Required**: Key point: rollout is using new weights on every iteration.
      1. This is where inference stops being just deployment infrastructure.
   3. **Required**: Bottleneck: disk IO -- checkpoints are huge, too much time is spent on IO and startup.
4. **Required**: Fast sync RL loop
   1. **Required**: Solution to disk IO bottleneck: connect trainer and inference via fast direct weight updates.
      1. **Optional**: What's InfiniBand, InfiniBand performance vs disk IO performance (example: 500B checkpoint)
      2. Candidate comparison: InfiniBand is around 400 GB/s per host vs disk write around 3 GB/s + disk read around 3 GB/s.
      3. Check units before final deck: InfiniBand bandwidth is often marketed in Gb/s, so confirm whether the intended number is GB/s per host, aggregate host bandwidth, or Gb/s per link.
   2. **Required, light**: vLLM needs a live weight-update path.
      1. **Optional / Appendix**: Custom `/update_weights` implementation.
      2. **Drop / Appendix**: Implemented as a custom worker class (`--worker-cls`).
      3. **Optional / Appendix**: Joins NCCL world, listens for updates, uses `cupy` to update the tensors.
      4. **Required as pointer**: NCCL-based weight transfer in vLLM docs: https://docs.vllm.ai/en/v0.18.2/training/weight_transfer/
      5. **Required**: After update: reset KVCache! (`/reset_prefix_cache` -- available since vLLM v0.13.0)
   3. **Optional**: On trainer side: join NCCL world with inference, broadcast the weights to vLLM after each training step.
   4. **Required before final deck**: Improvement: **TODO**: ADD NUMBERS HERE
      1. Either add real numbers or remove the performance claim.
   5. **Optional**: Downside: if one of training/inference workers is down -- the whole run needs to restart.
   6. **Required**: InfiniBand-based upgrade brings a new problem: idle GPUs, since both training and inference must be up all the time.
      1. **Required diagram**: Show transition from literal loop to two always-on blocks.
      2. **Required**: While inference is running, training GPUs are idle.
      3. **Required**: While training is running, inference GPUs are idle.
5. **Required**: Async RL loop
   1. **Required**: Sync RL vs Async RL
      1. **Required**: Sync RL:
         1. training and inference run **one by one**
         2. train on samples from the **latest** model
      2. **Required**: Async RL:
         1. training and inference run **in parallel**, minimizing GPU idle time
         2. train on samples from a model of up to K steps older
      3. Candidate benchmark for basic math RLVR with small context:
         1. sync: 1200 tok/sec
         2. async-k1: 2500 tok/sec
         3. async-k3: 3300 tok/sec
         4. async-k6: 3700 tok/sec
   2. **Required**: Where does Async come from? This is not that obvious.
      1. We need N samples to run one training step, but samples take different time to generate because of longer answers and complex multi-turn trajectories.
      2. **Required**: Solution: continuous dispatching. Instead of "dispatch N samples and wait for N", dispatch new samples as soon as possible, accumulate batches in the background, train once N samples are ready.
   3. **Required diagram**: How it affects the system: producer-consumer queues. Inference <-> Controller <-> Trainer.
      1. Inference generates rollouts.
      2. Controller schedules new tasks.
      3. Trainer trains once N samples are available, then updates inference weights.
   4. **Optional**: How it impacts vLLM:
      1. `/pause`, `/resume` as examples of lifecycle controls.
      2. Phrase carefully: weight updates can create lifecycle/correctness challenges around in-flight generation.
6. **Optional**: RL in open source: rLLM, veRL
   1. Put in "where to continue learning", not the main story.
7. **Required**: RL at Nebius AI R&D
   1. **Optional**: Papyrax -- in-house JAX-based large-scale training framework.
      1. Reference: https://nebius.com/blog/posts/post-training-in-token-factory
   2. **Required**: vLLM -- inference.
   3. **Required conceptually**: Controller -- actor-based scheduling/orchestration logic.
8. **Required**: Token Factory RL FT beta!
   1. Closing memory / call to action, not the center of gravity.
   2. Wording: Research-proven RL fine-tuning as a service.

## Slidedeck storyboard
Based on the storyline above, but slide-by-slide to make the slidedeck generation straightforward. Covers each slide, diagrams to generate are a special work item, where I need to provide sufficient input.

Target: 30-minute slot including questions. Aim for around 24 minutes of prepared material and keep 5-6 minutes for questions and buffer.

### Slide 1: Title

- **Purpose**: Set topic and credibility.
- **Title**: Closing the Loop: vLLM in RL Training Systems
- **Content**:
  - Simon Karasik
  - Israel vLLM Meetup #2, Nebius
- **Visual**: Abstract loop motif connecting "training" and "inference"; avoid dense architecture here.
- **Timing**: 30 sec

### Slide 2: The Talk in One Sentence

- **Purpose**: Give the audience the memory anchor.
- **Main point**: In RL post-training, inference becomes part of the training loop.
- **Content**:
  - vLLM stops being only a serving engine.
  - It becomes a rollout engine for continuously changing models.
- **Visual**: Before/after: "offline serving" vs "training loop component".
- **Timing**: 45 sec

### Slide 3: From SFT to RLVR

- **Purpose**: Place RLVR in the post-training progression without teaching all of RL.
- **Main point**: RLVR is where the reward signal becomes programmatic/verifiable, which makes scalable rollout generation valuable.
- **Content**:
  - SFT: imitate examples
  - RLHF: optimize preference signals
  - RLVR: optimize verifiable rewards
- **Visual**: Three-stage horizontal progression: SFT -> RLHF -> RLVR, with the reward signal becoming more explicit/verifiable.
- **Timing**: 1.25 min

### Slide 4: RLVR Training Loop

- **Purpose**: Show the basic RL training loop before naming GRPO.
- **Main point**: RLVR repeatedly generates, scores, and trains on fresh model behavior.
- **Content**:
  - sample tasks/prompts
  - generate rollouts
  - score/evaluate
  - update model
- **Visual**: Circular loop: prompts -> vLLM rollouts -> verifiable rewards -> trainer -> updated model -> vLLM.
- **Timing**: 1.75 min

### Slide 5: GRPO Intuition, No Formula

- **Purpose**: Explain the algorithmic idea without losing the room.
- **Main point**: GRPO learns from relative quality inside a generated group.
- **Content**:
  - One prompt produces K completions.
  - Each completion gets a reward.
  - Better-than-group-average samples are reinforced.
- **Visual**: Diagram: prompt -> K completions -> rewards -> group-relative comparison -> update.
- **Timing**: 2 min

### Slide 6: The Rollout Data Contract

- **Purpose**: Bridge algorithm to systems.
- **Main point**: In RL training, inference must return training data, not just text.
- **Content**:
  - `group_id`
  - `sample_id`
  - `token_ids`
  - `logprobs`
  - `reward`
- **Visual**: Packet/card moving from vLLM to trainer labeled "rollout sample".
- **Timing**: 1.25 min

### Slide 7: Naive RL Loop

- **Purpose**: Establish the first architecture in the ladder.
- **Main point**: The naive loop is simple but dominated by checkpoint movement and startup.
- **Content**:
  - train
  - save checkpoint
  - load/restart inference
  - generate rollouts
  - repeat
- **Visual**: Sequential timeline with disk checkpoint in the middle.
- **Timing**: 1.75 min

### Slide 8: Why Naive Breaks

- **Purpose**: Make the bottleneck visceral.
- **Main point**: Checkpoints are too large to route through disk every iteration.
- **Content**:
  - Huge checkpoint write + read path.
  - Inference startup/reload is repeated.
  - Rollouts need fresh or near-fresh weights.
  - Candidate comparison: InfiniBand around 400 GB/s per host vs disk write around 3 GB/s + disk read around 3 GB/s.
- **Visual**: Bandwidth comparison bar chart plus checkpoint object.
- **Diagram input needed**: Confirm InfiniBand units and whether the number should be GB/s per host, aggregate host bandwidth, or Gb/s per link.
- **Timing**: 1.75 min

### Slide 9: Sync Loop: Keep vLLM Alive

- **Purpose**: Introduce the first practical fix.
- **Main point**: Instead of restarting inference, push new weights into a live vLLM instance over InfiniBand.
- **Content**:
  - Trainer and inference stay up.
  - Weights move directly over InfiniBand instead of disk.
  - vLLM serves rollouts from updated weights.
- **Visual**: Two persistent blocks, Trainer and vLLM, connected by an InfiniBand/NCCL weight-update path.
- **Timing**: 2 min

### Slide 10: What vLLM Needs in the Sync World

- **Purpose**: Give concrete vLLM adaptation points.
- **Main point**: Live weight updates require inference-system correctness hooks.
- **Content**:
  - Weight update path.
  - Reset prefix/KV cache after weights change.
  - Lifecycle controls around generation/update boundaries.
  - vLLM TRL and weight transfer docs as follow-up material.
- **Visual**: Checklist over a vLLM engine box: update weights, reset cache, coordinate lifecycle.
- **Timing**: 2 min

### Slide 11: Sync Fixes IO, But Creates Idle GPUs

- **Purpose**: Show why sync is not the end state.
- **Main point**: Sync keeps weights fresh but still runs training and inference one after another.
- **Content**:
  - When inference runs, training GPUs wait.
  - When training runs, inference GPUs wait.
  - Tighter coupling: worker failure can restart the run.
- **Visual**: Timeline with alternating colored blocks and gray idle bands.
- **Timing**: 1.5 min

### Slide 12: Async Loop: Keep Both Sides Busy

- **Purpose**: Introduce async as the natural next step.
- **Main point**: Async RL overlaps rollout generation and training with bounded staleness because rollout latency is variable.
- **Content**:
  - Inference continuously generates rollouts.
  - Controller schedules new tasks and buffers samples.
  - Trainer consumes batches when enough samples are ready.
  - Continuous dispatch avoids waiting for the slowest rollout in a fixed batch.
  - Weights may be up to K steps old.
- **Visual**: Producer-consumer architecture: vLLM Inference <-> Controller queues <-> Trainer, with variable-length rollout bars feeding the queue.
- **Timing**: 2.25 min

### Slide 13: Sync vs Async Tradeoff

- **Purpose**: Cement the ladder with concrete throughput.
- **Main point**: Async improves utilization, but pays with staleness and orchestration complexity.
- **Content**:
  - sync: latest weights, lower utilization, 1200 tok/sec
  - async-k1: slight staleness, 2500 tok/sec
  - async-k3: more staleness, 3300 tok/sec
  - async-k6: more staleness, 3700 tok/sec
- **Visual**: Bar chart plus a small "freshness vs utilization" tradeoff axis.
- **Diagram input needed**: Confirm benchmark label: "basic math RLVR, small context".
- **Timing**: 2 min

### Slide 14: How We Do RL at Nebius AI R&D

- **Purpose**: Ground the architecture in the platform without turning into an internal deep dive.
- **Main point**: Nebius combines training, vLLM inference, and controller orchestration.
- **Content**:
  - Papyrax: in-house JAX-based large-scale training framework.
  - vLLM: rollout inference.
  - Controller: scheduling, queues, actor-based orchestration.
  - Blog reference: https://nebius.com/blog/posts/post-training-in-token-factory
- **Visual**: Three-box architecture map: Papyrax <-> Controller <-> vLLM.
- **Timing**: 1.5 min

### Slide 15: Where to Continue Learning

- **Purpose**: Give audience practical next steps.
- **Main point**: There are now real open-source and vLLM paths to study this.
- **Content**:
  - rLLM: https://github.com/rllm-org/rllm
  - veRL: https://github.com/verl-project/verl
  - vLLM TRL: https://docs.vllm.ai/en/stable/training/trl/
  - vLLM weight transfer: https://docs.vllm.ai/en/stable/training/weight_transfer/
  - Optional: QR code to a gist with links and references.
- **Visual**: QR code + compact link list.
- **Timing**: 45 sec

### Slide 16: Token Factory RL FT Beta

- **Purpose**: Close with the product option.
- **Main point**: Research-proven RL fine-tuning as a service.
- **Content**:
  - Token Factory RL FT beta.
  - For teams that want the RL post-training loop without building the whole system stack.
- **Visual**: Clean closing slide with the phrase "Research-proven RL fine-tuning as a service."
- **Timing**: 45 sec

### Optional Appendix Slides

1. **GRPO formula**
   - Use only if audience asks for the math.
2. **vLLM weight transfer internals**
   - Custom `/update_weights`, worker class, NCCL world, tensor update mechanics.
3. **Failure and lifecycle details**
   - Worker failure, pause/resume boundaries, in-flight generation behavior.
4. **Benchmark setup**
   - Hardware, model size, sequence length, task distribution, async K definition.
