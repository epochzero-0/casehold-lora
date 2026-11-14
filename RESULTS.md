# Results

Every number here was printed by the command above it. Hardware: RTX 4050 Laptop GPU, 6 GB, WSL2.
Model: `Qwen/Qwen2.5-0.5B`. Data: `coastalcph/lex_glue`, config `case_hold`.

## Data

`python look.py`

| split | questions |
|---|---|
| train | 45,000 |
| validation (dev) | 3,900 |
| test | 3,600 |

## Prompt length

`python lengths.py` (Qwen2.5 tokenizer, full prompt: instruction, context, five holdings, "Answer:")

| split | median | p95 | p99 | max |
|---|---|---|---|---|
| train | 416 | 533 | 620 | 1013 |
| validation | 414 | 532 | 610 | 749 |
| test | 415 | 533 | 609 | 827 |

`max_len = 640`. Over it, the context loses tokens from its left end; the holdings are never cut.

| split | context cut | holdings alone exceed 640 |
|---|---|---|
| train | 310 (0.69%) | 5 |
| validation | 23 (0.59%) | 0 |
| test | 17 (0.47%) | 0 |

The median context has 12 tokens after `<HOLDING>`, so cutting from the left keeps the citation.

## Zero-shot

`python eval.py --split validation`

- **38.38% on dev** (1,497 of 3,900). A random pick scores 20.
- Picks: A 503 · B 1,291 · C 1,638 · D 206 · E 262. Correct answers are spread evenly (760 to 804 each).
- 98 s, peak VRAM 1.42 GiB.

## LoRA checks

`python test_lora.py` (r = 16, alpha = 32, on q/k/v/o of all 24 layers, 8 dev questions, float32)

- 96 layers wrapped. Trainable: 2,162,688 of 496,195,456 (0.436%).
- Test 1, at init against the base model: max abs difference 0.0.
- Test 2, random A and B against `peft` holding the same tensors: max abs difference 0.0
  (the random weights moved the logits by up to 1.06, so the comparison isn't trivially zero).

## Training

One epoch, batch 4 with 4 batches per optimizer step (16 questions a step), lr 2e-4 with 5% warmup then
linear decay, r = 16, alpha = 32, bf16 autocast. No setting was tuned. Each run's `metrics.json` and
`log.csv` are in `runs/`.

**Memory.** Batch 8 with float32 frozen weights ran out of the 6 GB. Batch 4 with float32 weights filled
5.9 GB and slowed to under 4 questions a second. Storing the frozen weights in bf16 (A and B stay
float32) brought the peak to 4.1 GiB.

`python train.py --n_train 5000 --out runs/5k`

| train questions | dev accuracy | time | peak VRAM |
|---|---|---|---|
| 0 (zero-shot) | 38.38% | | 1.42 GiB |
| 5,000 | 75.28% (2,936 of 3,900) | 627 s, 8.0 questions/s | 4.11 GiB |

`python train.py --n_train 15000 --out runs/15k`

| train questions | dev accuracy | time | peak VRAM |
|---|---|---|---|
| 15,000 | **80.33%** (3,133 of 3,900) | 1,642 s, 9.1 questions/s | 4.12 GiB |
