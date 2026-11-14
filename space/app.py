"""Gradio demo: paste a citing passage and five holdings, see the model's probability for each.
Runs on CPU. data.py and lora.py are copied in next to this file when the Space is uploaded."""
import json

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import LETTERS, encode, letter_ids
from lora import load_adapter

ADAPTER = "epochzero-0/casehold-lora"
cfg = json.load(open(hf_hub_download(ADAPTER, "adapter_config.json")))

tok = AutoTokenizer.from_pretrained(cfg["base_model"])
model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.float32)
load_adapter(model, hf_hub_download(ADAPTER, "adapter.safetensors"), cfg["r"], cfg["alpha"])
model.eval()
letters = torch.tensor(letter_ids(tok))
examples = json.load(open("examples.json"))


@torch.no_grad()
def predict(context, a, b, c, d, e):
    ids = torch.tensor([encode({"context": context, "endings": [a, b, c, d, e]}, tok, cfg["max_len"])])
    final = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
    probs = model.lm_head(final)[letters].softmax(dim=-1)
    return {letter: float(p) for letter, p in zip(LETTERS, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(label="Citing passage, with <HOLDING> where the holding was", lines=8)]
    + [gr.Textbox(label=f"Holding {letter}", lines=2) for letter in LETTERS],
    outputs=gr.Label(label="Probability of each holding"),
    examples=examples,
    cache_examples=False,
    title="CaseHOLD: which holding did the court cite?",
    description=(
        "Qwen2.5-0.5B with a hand-written LoRA, fine-tuned on CaseHOLD. The examples are from the validation "
        "split, which the model was not trained on. Code and results: "
        "[github.com/epochzero-0/casehold-lora](https://github.com/epochzero-0/casehold-lora)."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
