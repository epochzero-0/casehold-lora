"""CaseHOLD as one prompt per question: instruction, citing context, holdings A-E, "Answer:"."""
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-0.5B"
LETTERS = "ABCDE"
HEAD = (
    "The passage below is from a US court opinion. <HOLDING> marks a missing holding. "
    "Pick the holding that fits.\n\n"
)


def load():
    return load_dataset("coastalcph/lex_glue", "case_hold")


def tail(ex):
    """The five holdings and the answer cue. Never truncated."""
    lines = "\n".join(f"{letter}: {e}" for letter, e in zip(LETTERS, ex["endings"]))
    return f"\n\n{lines}\nAnswer:"


def encode(ex, tok, max_len=None):
    """Token ids for one question. Over max_len, the context loses tokens from its left end:
    the citation with <HOLDING> sits at the right end, and the holdings stay whole."""
    head = tok(HEAD)["input_ids"]
    ctx = tok(ex["context"])["input_ids"]
    end = tok(tail(ex))["input_ids"]
    if max_len is not None:
        room = max(max_len - len(head) - len(end), 0)
        if len(ctx) > room:
            ctx = ctx[len(ctx) - room:]
    return head + ctx + end


def letter_ids(tok):
    """The token id of ' A' ... ' E', the five possible next tokens after 'Answer:'."""
    ids = [tok(f" {letter}")["input_ids"] for letter in LETTERS]
    assert all(len(i) == 1 for i in ids), f"a letter is not one token: {ids}"
    return [i[0] for i in ids]
