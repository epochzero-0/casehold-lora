"""Print the split sizes and one CaseHOLD example, exactly as stored."""
import sys

from datasets import load_dataset

ds = load_dataset("coastalcph/lex_glue", "case_hold")
for name, split in ds.items():
    print(f"{name:10s} {len(split):6d}")

i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ex = ds["train"][i]
print(f"\n--- train[{i}] ---")
print("context:\n" + ex["context"])
for letter, ending in zip("ABCDE", ex["endings"]):
    print(f"\n{letter}: {ending}")
print(f"\nlabel: {ex['label']} ({'ABCDE'[ex['label']]})")
