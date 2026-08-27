# How LLMs are trained

A general, project-agnostic explainer of how large language models (GPT-style,
Claude-style, etc.) get built. **Note:** this is background reading, not a
description of legalite-ai — this codebase does no training or fine-tuning;
it calls a pretrained model (OpenAI `gpt-4o-mini`) over RAG-retrieved context.
See `docs/architecture.md` for how the LLM is actually used here.

## 1. Data collection and preparation

Every model starts here, and nothing has been "learned" by the end of this
stage — it's purely preparing the ingredients.

- **Sourcing**: web pages, books, papers, code repositories, and other
  licensed or synthetic datasets, pulled together at a scale of trillions
  of tokens for frontier models.
- **Cleaning**: near-duplicate documents are removed (repetition biases
  what the model learns), spam and low-quality text is filtered, and
  harmful material is stripped out. Data quality is one of the biggest
  levers on final model quality — a smaller, cleaner dataset regularly
  beats a larger, messier one.
- **Tokenization**: text is split into subword tokens (commonly via BPE)
  rather than whole words or characters, keeping the vocabulary a
  manageable size while still covering arbitrary text. Every document is
  converted to a sequence of token IDs before training ever starts. This
  is also why LLMs are often weak at character-level tasks (like counting
  letters in a word) — they never see individual letters, only tokens.

*Try it yourself*: run a snippet of text through a public tokenizer to see
how it gets split, and try deduplicating/filtering a small text sample to
get a feel for what "cleaning" actually removes.

## 2. Pretraining

The most compute-expensive stage, and the one where the model acquires
nearly all of its raw knowledge.

- **Objective**: next-token prediction. Given a sequence of tokens, predict
  the next one, compare against the actual next token, and nudge the
  model's parameters to reduce that error — repeated across trillions of
  tokens and, for frontier models, hundreds of billions of parameters.
- **Architecture**: almost universally a decoder-only Transformer —
  stacked self-attention + feed-forward blocks with a causal mask, so each
  token can only attend to tokens before it.
- **Why such a simple objective produces so much capability**: to get
  good at predicting the next token across books, code, and arguments, the
  model has to implicitly pick up grammar, facts, reasoning patterns, and
  programming syntax — none of it explicitly labeled as "grammar" or
  "syntax" anywhere in the data.
- **Scale**: this is where the compute budget goes — thousands of
  accelerators running for weeks to months. Loss decreases predictably
  with more data, parameters, and compute, which is what "scaling laws"
  describe and what labs use to plan a training run's budget in advance.
- **Output — the base model**: extremely knowledgeable, but not an
  assistant. Ask it a question and it may just continue the sentence
  rather than answer it, because "continue this plausibly" is the only
  thing it was ever optimized to do. This single fact also explains
  hallucination: the model is optimized for plausible continuations, not
  verified truth, and that tension never fully goes away in later stages.

*Try it yourself*: train a tiny language model on a small dataset to feel
the train loop firsthand, and notice how fluency and hallucination trace
back to the same next-token objective.

## 3. Supervised fine-tuning (SFT)

Teaches the base model what role it's supposed to play.

- **Data**: a curated set of (prompt, ideal response) pairs — a question
  paired with a helpful answer, an instruction paired with the correct
  response — often written or vetted by human experts. Far smaller than
  pretraining data: thousands to low millions of examples, not trillions
  of tokens.
- **Objective**: still next-token prediction, but now only over these
  curated demonstrations, and typically only the response tokens count
  toward the loss (the prompt itself is masked out).
- **Effect**: the model shifts from "continue this text plausibly" to
  "answer like the demonstrations it was shown" — it starts behaving like
  something you can ask a question and get a focused answer from, rather
  than a text completer.
- **Where quality beats quantity most visibly**: a modest set of carefully
  written demonstrations reliably outperforms a much larger set of
  mediocre ones — the opposite trade-off from pretraining.

After this stage a model is already useful for many tasks — instruction
following, Q&A, code generation, summarization — but it's still missing
the judgment and safety refinement that later stages add.

*Try it yourself*: compare a base model's output to an instruction-tuned
model's output on the same prompt, and fine-tune a small open-source model
on a narrow task to watch its behavior shift.

## 4. Reward modeling

Solves a specific problem: usefulness isn't binary, and for most prompts
there's no single "correct" answer — only answers people prefer over
others. You can't hand-write rules for that, so instead:

- The SFT model generates several candidate responses to the same prompt.
- Human reviewers (or, in RLAIF/Constitutional-AI setups, another model
  acting as judge) rank those candidates from best to worst.
- Those rankings train a **separate** model — the reward model — whose
  only job is to predict which response a human would prefer.

Comparing two responses is a much easier and more scalable labeling task
than writing a perfect response from scratch, so this is how preference
data gets collected in bulk. Once trained, the reward model can score
millions of new responses automatically — a scalable stand-in for human
judgment that never talks to end users directly. Its output is the signal
the next stage optimizes against.

*Try it yourself*: rank a handful of alternate answers to the same
question yourself and notice how much easier ranking is than authoring —
that gap is the entire reason this stage exists.

## 5. Reinforcement learning (RLHF / RLAIF)

Takes the SFT model and the reward model and closes the loop:

Generate → the reward model scores the response → the policy (the LLM) is
updated to make higher-scoring responses more likely → repeat.

- Historically implemented via PPO (Proximal Policy Optimization).
  Increasingly, labs use **DPO** (Direct Preference Optimization), which
  skips training a separate reward model and optimizes directly on
  preference pairs — cheaper and simpler, though the underlying goal is
  the same.
- **Guardrail during optimization**: a penalty (commonly a KL-divergence
  term against the SFT model) keeps the policy from drifting too far and
  "reward-hacking" the signal — e.g. becoming sycophantic or artificially
  verbose because raters mildly preferred that on average.
- Because the reward model scores automatically, this loop can run far
  longer than direct human demonstration alone would allow, which is what
  gives modern assistants much of their consistency in following
  instructions, handling nuance, and refusing unsafe requests.
- Labs iterate this stage repeatedly post-launch — deploying, collecting
  more preference and safety data (including from red-teaming and real
  usage), and retraining — which is why the same model family gets point
  releases beyond the original pretraining run.

*Try it yourself*: sketch the generate → score → update loop and trace
through what would happen if the reward model were biased or exploitable
— that failure mode (reward hacking) is a real, active area of alignment
research, not a hypothetical.

## 6. Evaluation and gating

Throughout SFT, reward modeling, and RL, candidate models are checked
against held-out benchmarks (reasoning, coding, factuality,
instruction-following) and safety/red-team evaluations before a version
ships. Frontier labs also run targeted evaluations for specific risk areas
as part of release gating, not just capability scores.

## 7. What happens after training: inference

Everything above happens once, offline, producing a fixed set of model
weights. When an application (including a RAG system like legalite-ai)
calls the model afterward, no training occurs — it's a forward pass
through those frozen weights, conditioned on whatever prompt/context you
send. Prompt engineering, few-shot examples, RAG, and system prompts don't
change the weights at all; they only change the input the frozen model
conditions on.

Fine-tuning a copy of a base model on your own data (rare, and distinct
from RAG) is a smaller-scale repeat of stage 3 above, done after the fact
on top of a released model — and is not how legalite-ai customizes its
answers; this project relies entirely on retrieval (RAG) plus prompting,
not fine-tuning.

## Why this mental model is worth having

Once next-token prediction clicks as the foundation, a lot of LLM behavior
stops looking mysterious: fluency and hallucination come from the same
root cause, prompting works because you're steering what the frozen model
predicts next, and differences in "personality" across model families
trace back to differences in their SFT and reward-modeling data — not to
some hidden rulebook.

Every stage here can be reproduced at toy scale — small pretraining runs,
fine-tuning an open-source model, building a tiny reward model from your
own preference rankings — and doing so is one of the fastest ways to build
real intuition for how these systems actually behave, even without
reproducing a frontier model.

## Summary

| Stage | Objective | Data | Changes weights? |
|---|---|---|---|
| 1. Data prep | Clean, deduplicate, tokenize | Raw scraped/licensed text | No |
| 2. Pretraining | Next-token prediction | Trillions of tokens, unlabeled | Yes |
| 3. SFT | Imitate ideal responses | Curated prompt/response pairs | Yes |
| 4. Reward modeling | Predict human preference | Ranked response comparisons | Yes (separate model) |
| 5. RL (RLHF/RLAIF/DPO) | Optimize against reward signal | Reward model scores / preference pairs | Yes |
| 6. Evaluation | Gate releases | Benchmarks + red-teaming | No |
| 7. Inference (e.g. RAG) | Answer a specific request | Prompt + retrieved context | No |
