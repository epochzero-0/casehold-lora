"""Package a trained adapter for the Hugging Face Hub: hub/adapter.safetensors and
hub/adapter_config.json, next to the model card and the two files needed to load it.

    python export.py runs/45k
"""
import json
import shutil
import sys

import torch
from safetensors.torch import save_file

from data import MODEL

run = sys.argv[1]
state = torch.load(f"{run}/adapter.pt", map_location="cpu")
save_file(state, "hub/adapter.safetensors")
config = {
    "base_model": MODEL, "r": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "max_len": 640, "trained_on": json.load(open(f"{run}/metrics.json"))["n_train"],
}
json.dump(config, open("hub/adapter_config.json", "w"), indent=2)
for f in ("lora.py", "data.py"):
    shutil.copy(f, f"hub/{f}")
print(f"{len(state)} tensors, {sum(v.numel() for v in state.values()):,} numbers -> hub/")
