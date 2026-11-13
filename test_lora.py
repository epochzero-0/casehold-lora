"""Two checks on lora.py, on 8 real dev questions, in float32.

1. At init (B = 0) the letter logits equal the base model's exactly.
2. With random A and B, the same tensors loaded into peft's LoRA give the same logits.

    python test_lora.py
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import MODEL, encode, letter_ids, load
from eval import collate, letter_logits
from lora import TARGETS, count, inject

R, ALPHA = 16, 32
device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
letters = torch.tensor(letter_ids(tok), device=device)
rows = [encode(ex, tok, 640) for ex in load()["validation"].select(range(8))]
batch = collate(rows, tok.pad_token_id, device)


def fresh():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()


@torch.no_grad()
def run(model):
    return letter_logits(model, *batch, letters).cpu()


# --- 1. identity at init
model = fresh()
base_out = run(model)
wrapped = inject(model, R, ALPHA)
trainable, total = count(model)
init_out = run(model)
d1 = (init_out - base_out).abs().max().item()
print(f"wrapped {len(wrapped)} layers; trainable {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")
print(f"test 1, init vs base:  max abs diff = {d1:.1e}")
assert d1 == 0.0

# --- 2. same A and B, mine vs peft
torch.manual_seed(0)
saved = {}
for name, m in wrapped.items():
    torch.nn.init.normal_(m.B, std=0.02)  # B off zero, or both sides trivially equal the base
    saved[name] = (m.A.detach().clone(), m.B.detach().clone())
mine_out = run(model)
moved = (mine_out - base_out).abs().max().item()
del model, wrapped
torch.cuda.empty_cache()

peft_model = get_peft_model(
    fresh(), LoraConfig(r=R, lora_alpha=ALPHA, target_modules=list(TARGETS), lora_dropout=0.0)
).eval()
with torch.no_grad():
    for name, (A, B) in saved.items():
        i, proj = name.split(".")[1:]
        mod = getattr(peft_model.base_model.model.model.layers[int(i)].self_attn, proj)
        mod.lora_A["default"].weight.copy_(A)
        mod.lora_B["default"].weight.copy_(B)
peft_out = letter_logits(peft_model.base_model.model, *batch, letters).detach().cpu()
d2 = (mine_out - peft_out).abs().max().item()
print(f"random A, B moved the logits by up to {moved:.2f}")
print(f"test 2, mine vs peft:  max abs diff = {d2:.1e}")
assert moved > 0.1 and d2 < 1e-3
print("PASS")
