"""Accuracy on CaseHOLD: one forward pass per question, answer = argmax over the logits of
' A' ... ' E' at the last position.

    python eval.py --split validation
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import LETTERS, MODEL, encode, letter_ids, load


def collate(rows, pad_id, device):
    """Right-pad a list of token-id lists. Returns ids, mask and each row's last real position."""
    longest = max(len(r) for r in rows)
    ids = torch.full((len(rows), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    last = torch.tensor([len(r) - 1 for r in rows])
    return ids.to(device), mask.to(device), last.to(device)


def letter_logits(model, ids, mask, last, letters):
    """(batch, 5) logits. The vocabulary projection runs on the last position only:
    over every position it would be batch x length x 151,936 floats."""
    hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    final = hidden[torch.arange(ids.shape[0], device=ids.device), last]
    return model.lm_head(final)[:, letters]


@torch.no_grad()
def evaluate(model, tok, split, max_len, batch_size, device):
    """Returns (predictions, labels) as numpy arrays, in the split's order."""
    model.eval()
    letters = torch.tensor(letter_ids(tok), device=device)
    rows = [encode(ex, tok, max_len) for ex in split]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))  # similar lengths share a batch
    preds = np.zeros(len(rows), dtype=np.int64)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        ids, mask, last = collate([rows[i] for i in idx], tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = letter_logits(model, ids, mask, last, letters)
        preds[idx] = logits.argmax(dim=-1).cpu().numpy()
    return preds, np.array(split["label"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--split", default="validation")
    p.add_argument("--max_len", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n", type=int, default=None, help="first n examples only")
    p.add_argument("--adapter", default=None, help="adapter.pt from train.py; without it, zero-shot")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32)
    args = p.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    if args.adapter:
        from lora import load_adapter

        load_adapter(model, args.adapter, args.r, args.alpha)
    split = load()[args.split]
    if args.n:
        split = split.select(range(args.n))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds, labels = evaluate(model, tok, split, args.max_len, args.batch_size, device)
    secs = time.time() - t0

    print(f"{args.model} {args.adapter or 'zero-shot'}, {args.split}, n={len(labels)}, max_len={args.max_len}")
    print(f"accuracy: {100 * (preds == labels).mean():.2f}%  ({(preds == labels).sum()} of {len(labels)})")
    print("picked :", {l: int((preds == i).sum()) for i, l in enumerate(LETTERS)})
    print("correct:", {l: int((labels == i).sum()) for i, l in enumerate(LETTERS)})
    print(f"time: {secs:.0f}s   peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
