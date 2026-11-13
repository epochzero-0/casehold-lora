"""Fine-tune Qwen2.5-0.5B on CaseHOLD with the hand-written LoRA. Plain PyTorch loop.

    python train.py --n_train 5000 --out runs/5k

Loss is cross-entropy over the five letter logits only. Only A and B train.
The test split is never loaded here.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import MODEL, encode, letter_ids, load
from eval import collate, evaluate, letter_logits
from lora import count, inject, lora_state


def make_batches(rows, batch_size, rng):
    """Shuffle, then sort by length inside chunks of 50 batches, so a batch pads little
    but the order of batches stays random."""
    idx = rng.permutation(len(rows))
    out = []
    for c in range(0, len(idx), batch_size * 50):
        chunk = sorted(idx[c : c + batch_size * 50], key=lambda i: len(rows[i]))
        out += [chunk[b : b + batch_size] for b in range(0, len(chunk), batch_size)]
    return [out[i] for i in rng.permutation(len(out))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n_train", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_len", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4, help="batches per optimizer step")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between quick dev checks")
    p.add_argument("--quick_dev", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    # frozen weights in bf16, as Qwen ships them (1 GiB, against 2 in float32); A and B are float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    inject(model, args.r, args.alpha)
    trainable, total = count(model)
    print(f"trainable {trainable:,} of {total:,}", flush=True)

    ds = load()
    # one fixed shuffle: the 5k examples are inside the 15k, which are inside the 45k
    train = ds["train"].shuffle(seed=0).select(range(args.n_train))
    dev = ds["validation"]
    quick = dev.select(range(args.quick_dev))
    rows = [encode(ex, tok, args.max_len) for ex in train]
    labels = np.array(train["label"])
    letters = torch.tensor(letter_ids(tok), device=device)

    batches = make_batches(rows, args.batch_size, rng)
    steps = math.ceil(len(batches) / args.accum)
    warm = max(int(steps * args.warmup), 1)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    # linear warmup, then linear decay to zero
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm))
    )

    # the longest batch first, forward and backward: if it fits, every batch fits
    longest = sorted(range(len(rows)), key=lambda i: len(rows[i]))[-args.batch_size :]
    ids, mask, last = collate([rows[i] for i in longest], tok.pad_token_id, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = letter_logits(model, ids, mask, last, letters)
    F.cross_entropy(logits.float(), torch.tensor(labels[longest], device=device)).backward()
    opt.zero_grad(set_to_none=True)
    print(f"longest batch ({ids.shape[1]} tokens) fits: "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    del ids, mask, last, logits
    torch.cuda.empty_cache()

    log = open(os.path.join(args.out, "log.csv"), "w")
    log.write("step,examples,loss,quick_dev_acc,secs\n")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step, seen, recent = 0, 0, []
    model.train()
    for b, idx in enumerate(batches):
        ids, mask, last = collate([rows[i] for i in idx], tok.pad_token_id, device)
        y = torch.tensor(labels[idx], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = letter_logits(model, ids, mask, last, letters)
        loss = F.cross_entropy(logits.float(), y)
        (loss / args.accum).backward()
        recent.append(loss.item())
        seen += len(idx)
        if (b + 1) % args.accum == 0 or b == len(batches) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.eval_every == 0 or step == steps:
                preds, gold = evaluate(model, tok, quick, args.max_len, 16, device)
                model.train()
                acc = 100 * (preds == gold).mean()
                line = f"{step},{seen},{np.mean(recent):.4f},{acc:.2f},{time.time() - t0:.0f}"
                log.write(line + "\n")
                log.flush()
                print(f"step {step}/{steps}  examples {seen}  loss {np.mean(recent):.4f}  "
                      f"quick dev {acc:.2f}%  {time.time() - t0:.0f}s", flush=True)
                recent = []
    train_secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30

    # saved before the full dev pass, so a run that dies during the pass keeps its weights
    torch.save(lora_state(model), os.path.join(args.out, "adapter.pt"))
    preds, gold = evaluate(model, tok, dev, args.max_len, 16, device)
    dev_acc = 100 * (preds == gold).mean()
    metrics = {
        "model": args.model, "n_train": args.n_train, "dev_acc": round(float(dev_acc), 2),
        "dev_correct": int((preds == gold).sum()), "dev_n": int(len(gold)),
        "train_secs": round(train_secs), "gpu_hours": round(train_secs / 3600, 3),
        "peak_vram_gib": round(peak, 2), "optimizer_steps": steps, "trainable_params": trainable,
        "args": vars(args),
    }
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\nDEV {dev_acc:.2f}% ({metrics['dev_correct']} of {metrics['dev_n']})  "
          f"train {train_secs:.0f}s ({seen / train_secs:.1f} examples/s)  peak VRAM {peak:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
