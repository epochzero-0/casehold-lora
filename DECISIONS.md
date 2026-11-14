# Decisions

What I chose at each step, what the alternative was, and the number behind it. In build order.

## 1. The answer is read off five token logits

The prompt ends in `Answer:` and I compare the logits of ` A`, ` B`, ` C`, ` D`, ` E` at the last position.
The highest one is the answer.

The alternative was to score each holding separately: five prompts per question, each asking how likely that
holding is. That is five forward passes for every question, in training and in evaluation. Reading five
logits is one pass and trains no new layer, since the letters are already tokens the model knows.

## 2. The vocabulary projection runs on one position

Qwen's output layer maps each position to 151,936 logits. For a batch of 16 prompts of 640 tokens that is
about 1.5 billion numbers, 3 GB in bf16, and I need five of them per question. `letter_logits` in `eval.py`
takes the hidden state at the last position only and projects that.

## 3. max_len 640, cut from the left

The median prompt is 416 tokens, p99 is 620, the longest is 1,013. At 512 the cut touches 7.68% of train
questions. At 640 it touches 0.69% (310 of 45,000).

The cut comes off the start of the passage. The median passage has 12 tokens after `<HOLDING>`, so the
citation is at the end and survives. The five holdings are never cut. In 5 train questions the holdings
alone are longer than 640, and those keep no passage at all.

## 4. Zero-shot before any training

The untrained model scores 38.38% on dev, against 20 for a random pick. That did two jobs. It showed the
prompt and the metric code work, because a bug would have given about 20. And it is the baseline that says
what training added.

It also showed a bias: the model picked B or C for 2,929 of 3,900 questions, though the correct letters are
spread evenly.

## 5. LoRA on q, k, v, o with r = 16, alpha = 32

Each wrapped layer computes `base(x) + (alpha / r) * B A x`. The base weight is frozen. A is 16 x in, B is
out x 16. Across the four attention projections of 24 layers that is 2,162,688 trained numbers, 0.436% of
the model.

B starts at zero, so the correction is zero before training and the first step starts from the pretrained
model. A starts random. If both were zero, neither would get a gradient.

r, alpha and the target layers are common defaults. I did not tune them.

## 6. Two tests on the LoRA code before training with it

At init, the logits on 8 dev questions equal the base model's: max difference 0.0. Then with random A and
B, the same tensors loaded into Hugging Face `peft` give the same logits: 0.0 again. The random weights
moved the logits by up to 1.06, so the second test is comparing something.

Without these, a bug in my 30 lines would show up only as a lower accuracy, and I would have no way to tell
it from a weak method.

## 7. Loss over the five letters only

Cross-entropy is taken over the five letter logits, not the whole vocabulary. The metric is which of five
is highest, so the loss asks the same question.

## 8. Fitting 6 GB

Three attempts.

- Batch 8, frozen weights in float32: out of memory on the first batches.
- Batch 4, float32: it ran, but filled 5.9 of 6.1 GB and slowed to under 4 questions a second.
- Batch 4, frozen weights stored in bf16: peak 4.1 GiB, 8 to 12 questions a second.

Qwen ships its weights in bf16, so storing them that way loses nothing and halves their size. A and B stay
float32. bf16 holds about three significant digits, and an optimizer step is far smaller than the weight it
changes, so in bf16 the step would round away.

A batch of 4 gives a noisy gradient, so gradients from 4 batches are added up before each optimizer step.
That is 16 questions a step with the memory of 4.

## 9. The longest batch runs first

Before training, `train.py` runs one forward and backward pass on the longest prompts in the set. If that
fits, every batch fits. Otherwise an out-of-memory error can arrive an hour into a run, whenever the longest
prompts happen to come up.

## 10. The three training sets are nested

The train split is shuffled once with a fixed seed. The 5k run takes the first 5,000, the 15k run the first
15,000. So the 15k set contains the 5k set, and the curve shows the effect of adding data and not of drawing
a different sample.

## 11. Weights are saved before the last evaluation

The first 15k run finished training and was killed during the final dev pass, when the laptop ran out of RAM.
The adapter was written after that pass, so 27 minutes of training were lost. It is written before the pass
now.

## 12. Dev chooses, test is scored once

Every accuracy during the build is on the 3,900 validation questions. The 3,600 test questions are not
loaded by `train.py` at all. The best run on dev gets scored on test one time, and that is the number
compared with the LexGLUE table.

## What this does not show

One seed per run and no tuning, so I have no error bar on any accuracy.
