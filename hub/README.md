---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
datasets:
- coastalcph/lex_glue
language:
- en
tags:
- lora
- legal
- multiple-choice
- casehold
model-index:
- name: casehold-lora
  results:
  - task:
      type: multiple-choice
      name: Multiple choice QA
    dataset:
      name: CaseHOLD (LexGLUE)
      type: coastalcph/lex_glue
      config: case_hold
      split: test
    metrics:
    - type: accuracy
      value: 85.92
      name: Accuracy
---

# casehold-lora

A LoRA adapter for [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), trained on CaseHOLD: given a
passage from a US court opinion with one cited holding removed, pick that holding from five candidates.

**85.92% on the LexGLUE test split** (3,093 of 3,600).

Code, every number with the command that printed it, and the decisions behind the build:
[github.com/epochzero-0/casehold-lora](https://github.com/epochzero-0/casehold-lora).

## Results

| Model | CaseHOLD test | Source |
|---|---|---|
| TF-IDF + SVM | 22.4 | LexGLUE |
| Qwen2.5-0.5B, zero-shot | 38.4 (validation) | this work |
| BERT | 70.8 | LexGLUE |
| RoBERTa-large | 74.4 | LexGLUE |
| Legal-BERT | 75.3 | LexGLUE |
| CaseLaw-BERT | 75.4 | LexGLUE |
| **Qwen2.5-0.5B + this adapter** | **85.9** | this work |

Validation accuracy by training size: 75.28% at 5,000 questions, 80.33% at 15,000, 87.05% at 45,000.

One seed and no hyperparameter tuning, so there is no error bar. The model was chosen on validation and scored
on test once. Qwen2.5-0.5B (494M parameters) is larger than BERT and Legal-BERT (about 110M).

## Training

- LoRA written by hand (`lora.py`, about 30 lines), r = 16, alpha = 32, on q/k/v/o of all 24 layers:
  2,162,688 trained parameters, 0.436% of the model. Checked against `peft` with the same weights: the logits
  match with a max difference of 0.0.
- One epoch over the 45,000 train questions, plain PyTorch loop, AdamW, lr 2e-4, 16 questions per step, bf16
  autocast.
- 62 minutes (1.04 GPU-hours) on an RTX 4050 Laptop GPU, peak 4.3 GiB of VRAM.

## How it answers

One prompt per question: an instruction, the passage, the five holdings lettered A to E, then `Answer:`. The
answer is whichever of the tokens ` A` to ` E` has the highest logit at the last position. Prompts over 640
tokens lose passage text from the left; the holdings are never cut.

## Use

This is not a `peft` checkpoint. `adapter.safetensors` holds the A and B matrices, and `lora.py` in this repo
wraps the base model with them.

```python
import json, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "epochzero-0/casehold-lora"
for f in ("lora.py", "data.py"):
    hf_hub_download(repo, f, local_dir=".")
from data import encode, letter_ids
from lora import load_adapter

cfg = json.load(open(hf_hub_download(repo, "adapter_config.json")))
tok = AutoTokenizer.from_pretrained(cfg["base_model"])
model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.float32)
load_adapter(model, hf_hub_download(repo, "adapter.safetensors"), cfg["r"], cfg["alpha"]).eval()

question = {"context": "... (<HOLDING>) ...", "endings": ["holding that ...", "...", "...", "...", "..."]}
ids = torch.tensor([encode(question, tok, cfg["max_len"])])
with torch.no_grad():
    final = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
    probs = model.lm_head(final)[letter_ids(tok)].softmax(-1)
print(dict(zip("ABCDE", probs.tolist())))
```

## Limits

It picks among five given holdings and nothing else. It is trained on US case law only, it is not legal
advice, and it has not been tested on holdings written in any other style.

## Licences and citations

Adapter: Apache-2.0, as is Qwen2.5-0.5B. CaseHOLD was released by its authors under Apache-2.0; the LexGLUE
copy is CC BY 4.0.

- Zheng, Guha, Anderson, Henderson, Ho. *When Does Pretraining Help? Assessing Self-Supervised Learning for
  Law and the CaseHOLD Dataset.* ICAIL 2021. arXiv:2104.08671
- Chalkidis et al. *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL 2022.
  arXiv:2110.00976
