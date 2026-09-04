"""
Loads a trained baseline or bottleneck checkpoint and reports parameter count
plus accuracy at each requested sequence length (freshly evaluated, not
just re-printing the training-time CSV). Mirrors evaluate.py in the other
three projects.

Usage:
    python evaluate.py --ckpt results/baseline/model_bottleneck.pt --test_seq_lens 16 32 64 128
    python evaluate.py --summary results/baseline/summary_baseline.json   # no recompute
"""
import argparse
import json

import torch

from data import make_batch, VOCAB_SIZE, N_CLASSES
from model import BaselineClassifier, BottleneckClassifier
from train import eval_acc


def print_summary(d: dict):
    print("```")
    print(f"Model:            {d.get('kind', '?')}")
    print(f"Parameters:       {d.get('n_params', '?'):,}" if isinstance(d.get("n_params"), int) else "")
    print(f"Train seq_len:    {d.get('train_seq_len', '?')}")
    print(f"Seed:             {d.get('seed', '?')}")
    if d.get("kind") == "bottleneck":
        print(f"alpha0 (prior):   {d.get('alpha0', '?')}")
        print(f"KL weight:        {d.get('kl_weight', '?')}")
    print("Accuracy by seq_len:")
    for seq_len, acc in d.get("accuracies", {}).items():
        print(f"  {seq_len:>4}: {acc:.3f}")
    print("```")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--summary", type=str, default=None, help="print a saved summary_*.json instead of recomputing")
    p.add_argument("--test_seq_lens", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--n_batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    if args.summary:
        with open(args.summary) as f:
            s = json.load(f)
        for kind in ("baseline", "bottleneck"):
            print(f"\n--- {kind} (from summary) ---")
            accs = {sl: v[f"{kind}_acc"] for sl, v in s["per_seq_len"].items()}
            print_summary({
                "kind": kind, "n_params": s.get(f"n_params_{kind}"), "train_seq_len": s["train_seq_len"],
                "seed": s["seed"], "alpha0": s.get("alpha0"), "kl_weight": s.get("kl_weight"),
                "accuracies": accs,
            })
        return

    if not args.ckpt:
        raise SystemExit("provide either --ckpt or --summary")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    kind = ckpt["kind"]
    margs = ckpt["args"]

    if kind == "baseline":
        model = BaselineClassifier(VOCAB_SIZE, N_CLASSES).to(device)
    else:
        model = BottleneckClassifier(VOCAB_SIZE, N_CLASSES, alpha0=margs.get("alpha0", 5.0)).to(device)
    model.load_state_dict(ckpt["model_state"])

    n_params = sum(p_.numel() for p_ in model.parameters())
    accuracies = {
        sl: eval_acc(model, sl, args.n_batches, args.batch_size, device) for sl in args.test_seq_lens
    }
    print_summary({
        "kind": kind, "n_params": n_params, "train_seq_len": margs.get("train_seq_len"),
        "seed": margs.get("seed"), "alpha0": margs.get("alpha0"), "kl_weight": margs.get("kl_weight"),
        "accuracies": accuracies,
    })


if __name__ == "__main__":
    main()
