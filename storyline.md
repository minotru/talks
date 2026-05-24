# Closing the Loop: Adapting vLLM for RL Training Systems

## Spec

URL: https://luma.com/9ihwxsej?utm_source=chatgpt.com

Duration: 30 minutes with questions

Format: stage talk

Title: Closing the Loop: vLLM in Reinforcement Learning Training Systems

Where: Israel vLLM meetup #2, in Nebius office.

When: Wed May 27, 2025 18:20 – 18:50

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

1. Nebius & Nebius AI R&D: quick overview
2. RL is now the frontier post-training method
   1. What's RL. SFT -> RLHF -> RLVR
   2. RLVR in LLMs: SoTA reasoning, math, coding
   3. GRPO formulae, simple explanation, core blocks
   4. What's passed: (group_id, sample_id, token_ids, logprobs, reward)
3. Naive GRPO training loop:
   1. The loop overview: sample tasks, rollout, evaluate, train.
   2. Key point: rollout is using new weights on every iteration.
   3. Bottleneck: disk IO -- checkpoints are huge, too much time is spent on IO and startup
4. Fast sync RL loop
   1. Solution to disk IO bottleneck: connect trainer and inference via InfiniBand for sub-second weight updates
      1. What's InfiniBand, InfiniBand performance vs disk IO performance (example: 500B checkpoint)
   2. Implementation details:
      1. On vLLM side: custom `/update_weights` implementation:
         1. Implemented as a custom worker class (`--worker-cls`)
         2. Joins NCCL world, listens for updates, uses `cupy` to update the tensors
         3. Since vLLM v18.0.2, NCCL-based weight transfer is supported out-of-the-box: https://docs.vllm.ai/en/v0.18.2/training/weight_transfer/
         4. After update: reset KVCache! (`/reset_prefix_cache` -- available since vLLM v0.13.0)
      2. On trainer side: join NCCL world with inference, broadcast the weights to vLLM after each training step
      3. Improvement: **TODO**: ADD NUMBERS HERE
   3. Downside: if one of training/inference workers is down -- the whole run needs to restart.
   4. InfiniBand-based upgrade brings a new problem: stale GPUs, since both training and inference must be up all the time.
      1. Some digram that shows this transition: for literal loop to 2 blocks that are up all the time
      2. While inference is running, training GPUs stale
      3. While training is running, inference GPUs stale
5. Async RL loop
   1. SyncRL vs Async RL
      1. Sync RL:
         1. training and inference run **one by one**
         2. train on samples from the **latest** model
      2. Async RL:
         1. training and inference run **in parallel**, and so minimize GPU idle time
         2. train on samples from a model of up to K steps older 
   2. Where does Async come from? This is not that obvious
      1. We need N samples to run 1 training step, but samples take different time to generate. (because of longer answers, complex multi-turn trajectories)
      2. Solution: continious dispatching. Instead of "dispatch N samples and wait for N" do "dispatch new samples as soon as possible, accumulate batch in the background, train once N samples are ready".
   3. How it affects the system (diagram): consumer-producer queues. Inference <-> Controller <-> Trainer. Inference -- generates rollouts, Controller -- schedules new tasks, Traininer -- trains once there are N samples available, updates the weights in Inference.
   4. How it impacts vLLM:
      1. `/pause`, `/resume` -- seems now weight updates happen in the middle of generation
6. RL in open source: rLLM, veRL
7. RL at Nebius AI R&D:
   1. Papyrax -- in-house JAX-based large-scale training framework
   2. vLLM -- inference
   3. Controller -- lots of actor-based logic
8. Token Factory RL FT beta!

## Slidedeck storyboard
Based on the storyline above, but slide-by-slide to make the slidedeck generation straightforward. Covers each slide, diagrams to generate are a special work item, where I need to provide sufficient input.

