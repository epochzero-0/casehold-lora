"""Token lengths of the full prompt, per split, and what each max_len would cut.
Writes plots/lengths.png."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from data import HEAD, MODEL, encode, letter_ids, load, tail

tok = AutoTokenizer.from_pretrained(MODEL)
ds = load()
print("letter token ids:", dict(zip("ABCDE", letter_ids(tok))))
print("instruction tokens:", len(tok(HEAD)["input_ids"]))

full, fixed, after = {}, {}, {}
for name, split in ds.items():
    full[name] = np.array([len(encode(ex, tok)) for ex in split])
    # head + holdings: the part that is never cut
    fixed[name] = np.array([len(tok(HEAD)["input_ids"]) + len(tok(tail(ex))["input_ids"]) for ex in split])
    # context tokens to the right of <HOLDING>
    after[name] = np.array(
        [len(tok(ex["context"].split("<HOLDING>")[-1])["input_ids"]) for ex in split]
    )

for name in ds:
    L = full[name]
    print(f"\n{name}: n={len(L)}  min={L.min()}  median={int(np.median(L))}  "
          f"p95={int(np.percentile(L, 95))}  p99={int(np.percentile(L, 99))}  max={L.max()}")
    print(f"  never-cut part (instruction + holdings): median={int(np.median(fixed[name]))} max={fixed[name].max()}")
    print(f"  context tokens after <HOLDING>: median={int(np.median(after[name]))} max={after[name].max()}")
    for m in (384, 448, 512, 576, 640, 768):
        cut = int((L > m).sum())
        lost = int((fixed[name] > m).sum())
        print(f"  max_len={m}: context cut in {cut} ({100 * cut / len(L):.2f}%), holdings would not fit in {lost}")

os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(7, 4))
plt.hist(full["train"], bins=80, color="#444")
plt.xlabel("prompt length, tokens (Qwen2.5 tokenizer)")
plt.ylabel("train examples")
plt.title("CaseHOLD prompt length, 45,000 train examples")
plt.tight_layout()
plt.savefig("plots/lengths.png", dpi=150)
print("\nwrote plots/lengths.png")
