"""Extract residual-stream activations for the fixed probe set.

Writes an array of shape (n_prompts, n_layers + 1, hidden) in probe order,
then prints the category similarity matrix and the two reference pairs.
"""
import argparse, time
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes import ALL_PROMPTS, CATEGORIES


def extract(model_id, batch_size=8, threads=4, dtype=torch.float32):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # .model drops the vocabulary projection head, which is not used here
    body = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval().model

    t0 = time.time()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(ALL_PROMPTS), batch_size):
            enc = tok(ALL_PROMPTS[i:i + batch_size], return_tensors="pt", padding=True)
            out = body(**enc, output_hidden_states=True)
            last = torch.stack([h[:, -1, :] for h in out.hidden_states])
            chunks.append(last.permute(1, 0, 2))
    A = torch.cat(chunks).float().numpy()
    return A, time.time() - t0


def similarity(A, layer=None):
    layer = A.shape[1] // 2 if layer is None else layer
    X = A[:, layer, :]
    X = X - X.mean(0)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    cats = sorted(set(CATEGORIES))
    cent = {c: X[[k == c for k in CATEGORIES]].mean(0) for c in cats}
    S = np.array([[1.0 - cosine(cent[a], cent[b]) for b in cats] for a in cats])
    return S, cats, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="acts.npy")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    A, dt = extract(args.model, args.batch_size, args.threads)
    np.save(args.out, A)
    print(f"{args.model}")
    print(f"shape {A.shape} (probes, layers+1, hidden), {dt:.1f} s, saved to {args.out}")

    S, cats, layer = similarity(A, args.layer)
    print(f"\ncategory similarity, layer {layer} of {A.shape[1] - 1}")
    print("           " + " ".join(f"{c[:6]:>7}" for c in cats))
    for i, a in enumerate(cats):
        print(f"{a:>10} " + " ".join(f"{S[i, j]:7.2f}" for j in range(len(cats))))

    e, s = cats.index("ethical"), cats.index("spatial")
    l, m = cats.index("logical"), cats.index("emotional")
    print("\nreference pairs (step 0, five models, averaged):")
    print(f"  ethical vs spatial     {S[e, s]:+.2f}   reference -0.46")
    print(f"  logical vs emotional   {S[l, m]:+.2f}   reference -0.34")


if __name__ == "__main__":
    main()
