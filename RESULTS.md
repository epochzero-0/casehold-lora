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

