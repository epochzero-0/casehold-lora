# casehold-lora

Qwen2.5-0.5B fine-tuned on CaseHOLD with a LoRA I wrote myself, in a plain PyTorch loop, on a 6 GB laptop GPU.

## The question

CaseHOLD (Zheng et al., 2021) was built to ask when pretraining on legal text pays off. On the LexGLUE
benchmark, Legal-BERT, pretrained on case law, scores 75.3 on it and plain BERT scores 70.8.

I wanted to know whether a small general-purpose LLM, fine-tuned cheaply, gets to the same place without any
legal pretraining.

## The task

A US court opinion cites an earlier case, and after the citation the judge writes in brackets what that case
decided, its holding. CaseHOLD removes the holding and asks which of five candidates it was.

```
... See United States v. Newman, 125 F.3d 863 (10th Cir.1997) (unpublished) (<HOLDING>); ...

A: holding that possession of a pipe bomb is a crime of violence for purposes of 18 usc 3142f1
B: holding that bank robbery by force and violence or intimidation under 18 usc 2113a is a crime of violence
C: holding that sexual assault of a child qualified as crime of violence under 18 usc 16
D: holding for the purposes of 18 usc 924e that being a felon in possession of a firearm is not a violent felony ...
E: holding that a court must only look to the statutory definition not the underlying circumstances ...
```

The right answer is A. It is the holding the court wrote, so no annotator chose the label.

The four wrong answers are holdings from other cases, and they are picked to be close. Section 3 of the paper:
the authors compute TF-IDF similarity between the correct holding and the pool of other holdings and take the
most similar ones, dropping any above 0.75 similarity as indistinguishable. That is why B to E above are all
about crimes of violence, and why matching words does badly here: TF-IDF with an SVM scores 22.4, and a random
pick scores 20.

I use the LexGLUE copy (`coastalcph/lex_glue`, config `case_hold`) and its fixed split: 45,000 train, 3,900
validation, 3,600 test.

## How it works

**One prompt per question.** An instruction, the citing passage, the five holdings lettered A to E, then
`Answer:`. The model's answer is whichever of the tokens ` A` to ` E` has the highest logit at the last
position. That is one forward pass per question and no new classification head (`data.py`, `eval.py`).

**Long prompts lose context from the left.** The median prompt is 416 tokens and I cap it at 640. The citation
sits at the end of the passage, so the cut comes off the start, and the holdings are never cut. It touches
0.69% of train questions (`lengths.py`).

**LoRA by hand.** `lora.py` is about 30 lines. Each attention projection (q, k, v, o in all 24 layers) keeps
its pretrained weight frozen and adds `(alpha / r) * B A x`, where A and B are two small matrices and the only
things trained: 2,162,688 parameters, 0.436% of the model. B starts at zero, so before training the model is
exactly the pretrained one.

`test_lora.py` checks both halves of that. At init the logits equal the base model's with a max difference of
0.0. With random A and B, the same tensors loaded into Hugging Face `peft` give the same logits, also 0.0.

**Training** (`train.py`) is cross-entropy over the five letter logits, AdamW on A and B, one epoch, bf16
autocast. No hyperparameter was tuned.
