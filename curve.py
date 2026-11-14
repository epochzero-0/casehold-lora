"""Dev accuracy against training-set size, from runs/*/metrics.json, with the published
LexGLUE test scores drawn as reference lines. Writes plots/curve.png.

    python curve.py --zero_shot 38.38
"""
import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CaseHOLD column of the LexGLUE table (Chalkidis et al. 2022), test split
LEXGLUE = {"TF-IDF+SVM": 22.4, "BERT": 70.8, "RoBERTa-large": 74.4, "Legal-BERT": 75.3, "CaseLaw-BERT": 75.4}

p = argparse.ArgumentParser()
p.add_argument("--zero_shot", type=float, required=True, help="dev accuracy of the base model")
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
args = p.parse_args()

runs = [json.load(open(f)) for f in glob.glob("runs/*/metrics.json")]
runs = sorted((r for r in runs if r["model"] == args.model), key=lambda r: r["n_train"])
for r in runs:
    print(f"{r['n_train']:6d} examples  dev {r['dev_acc']:.2f}%  {r['gpu_hours']:.2f} GPU-h  "
          f"peak {r['peak_vram_gib']:.2f} GiB")

plt.figure(figsize=(7, 4.5))
for name in ("TF-IDF+SVM", "BERT", "Legal-BERT"):
    plt.axhline(LEXGLUE[name], color="#999", lw=0.8, ls="--")
    plt.text(runs[-1]["n_train"], LEXGLUE[name] + 0.6, f"{name} {LEXGLUE[name]} (test)", ha="right", fontsize=8, color="#555")
plt.axhline(args.zero_shot, color="#c44", lw=0.8, ls=":")
plt.text(runs[-1]["n_train"], args.zero_shot + 0.6, f"zero-shot {args.zero_shot}", ha="right", fontsize=8, color="#c44")
plt.plot([r["n_train"] for r in runs], [r["dev_acc"] for r in runs], "o-", color="#111")
for r in runs:
    plt.annotate(f"{r['dev_acc']:.1f}", (r["n_train"], r["dev_acc"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
plt.xscale("log")
plt.xticks([r["n_train"] for r in runs], [f"{r['n_train']:,}" for r in runs])
plt.minorticks_off()
plt.xlabel("training examples (one epoch)")
plt.ylabel("accuracy, %")
plt.title(f"{args.model.split('/')[-1]} + LoRA on CaseHOLD: dev accuracy by training size")
plt.ylim(15, 95)
plt.tight_layout()
plt.savefig("plots/curve.png", dpi=150)
print("wrote plots/curve.png")
