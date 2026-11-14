---
title: CaseHOLD LoRA demo
emoji: ⚖️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
models:
- epochzero-0/casehold-lora
- Qwen/Qwen2.5-0.5B
---

Paste a passage from a US court opinion with `<HOLDING>` where a cited holding was removed, and five candidate
holdings. The model returns its probability for each.

Model: [epochzero-0/casehold-lora](https://huggingface.co/epochzero-0/casehold-lora) (85.92% on the LexGLUE
CaseHOLD test split). Code: [github.com/epochzero-0/casehold-lora](https://github.com/epochzero-0/casehold-lora).
