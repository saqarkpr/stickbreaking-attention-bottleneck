"""
Trains BaselineClassifier (uniform mean-pooling) and BottleneckClassifier
(stick-breaking variational bottleneck pooling) on the same synthetic
"one signal token in a sea of noise" task, at a fixed training sequence
length, then evaluates BOTH at longer sequence lengths than seen during
training (more noise tokens diluting the same one signal token).

Hypothesis: the stick-breaking bottleneck's variational prior should let it learn to
concentrate mixture weight on the informative token and ignore the rest, so
its accuracy should degrade less than the baseline's uniform-mean-pooling
accuracy as noise increases at test time. (alpha0's effect on that
concentration turns out to be the opposite of what "concentration parameter"
suggests on first reading -- see bottleneck_layer.py and the alpha0 sweep in the
README: LARGER alpha0 makes the mixture more uniform, not sparser.)

This is the experiment structure to run, not a claim about the outcome —
report the actual numbers from your run in the README once trained.

Mirrors the other three projects' structure: `run_experiment(args)` is
factored out of `main()`, checkpoints + per-step loss CSVs are written (not
just the final OOD comparison table), and `--seed`/`--n_seeds` give a
single reproducible run or a mean±std study across seeds.
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import make_batch, VOCAB_SIZE, N_CLASSES
from model import BaselineClassifier, BottleneckClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(model, steps, seq_len, batch_size, lr, device, log_path, kl_weight=0.0, is_bottleneck=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "ce_loss", "kl", "train_acc"])

    log_every = max(1, steps // 50)  # ~50 points per curve, enough to plot without huge files
    for step in range(1, steps + 1):
        x, y, mask = make_batch(batch_size, seq_len, device)
        logits, kl = model(x, mask)
        ce = F.cross_entropy(logits, y)
        loss = ce + (kl_weight * kl if is_bottleneck else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == 1:
            acc = (logits.argmax(-1) == y).float().mean().item()
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, ce.item(), kl.item(), acc])
            if step % max(1, steps // 5) == 0:
                print(f"  step {step:5d} | ce {ce.item():.4f} | kl {kl.item():.4f} | train_acc {acc:.3f}")
    return model


@torch.no_grad()
def eval_acc(model, seq_len, n_batches, batch_size, device):
    model.eval()
    correct, total = 0, 0
    for _ in range(n_batches):
        x, y, mask = make_batch(batch_size, seq_len, device)
        logits, _ = model(x, mask)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_seq_len", type=int, default=16)
    p.add_argument("--test_seq_lens", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--kl_weight", type=float, default=0.01)
    p.add_argument("--alpha0", type=float, default=5.0)
    p.add_argument("--shuffle_sticks", action="store_true",
                    help="order-randomised control: restores exchangeability in expectation")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="run1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_seeds", type=int, default=1,
                    help="if > 1, repeats the whole comparison at seeds 42, 43, ... and reports mean +/- std per seq_len")
    return p


def run_experiment(args) -> dict:
    """Trains + evaluates both models once, at args.seed. Writes checkpoints,
    per-step loss CSVs, and the OOD comparison CSV. Returns a summary dict."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    print(f"=== [{args.tag}] training baseline (mean-pool), seed={args.seed} ===")
    baseline = BaselineClassifier(VOCAB_SIZE, N_CLASSES).to(device)
    train_one(baseline, args.steps, args.train_seq_len, args.batch_size, args.lr, device,
              os.path.join(run_dir, "loss_log_baseline.csv"), is_bottleneck=False)
    torch.save({"model_state": baseline.state_dict(), "kind": "baseline", "args": vars(args)},
               os.path.join(run_dir, "model_baseline.pt"))

    print(f"=== [{args.tag}] training bottleneck (stick-breaking), seed={args.seed} ===")
    bottleneck = BottleneckClassifier(VOCAB_SIZE, N_CLASSES, alpha0=args.alpha0,
                           shuffle_sticks=getattr(args, 'shuffle_sticks', False)).to(device)
    train_one(bottleneck, args.steps, args.train_seq_len, args.batch_size, args.lr, device,
              os.path.join(run_dir, "loss_log_bottleneck.csv"), kl_weight=args.kl_weight, is_bottleneck=True)
    torch.save({"model_state": bottleneck.state_dict(), "kind": "bottleneck", "args": vars(args)},
               os.path.join(run_dir, "model_bottleneck.pt"))

    print(f"\n=== [{args.tag}] evaluating both models at seq_len in {args.test_seq_lens} "
          f"(trained at {args.train_seq_len}) ===")
    ood_path = os.path.join(run_dir, "ood_comparison.csv")
    per_seq_len = {}
    with open(ood_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "baseline_acc", "bottleneck_acc"])
        for seq_len in args.test_seq_lens:
            base_acc = eval_acc(baseline, seq_len, n_batches=20, batch_size=args.batch_size, device=device)
            bottleneck_acc = eval_acc(bottleneck, seq_len, n_batches=20, batch_size=args.batch_size, device=device)
            print(f"seq_len {seq_len:4d} | baseline_acc {base_acc:.3f} | bottleneck_acc {bottleneck_acc:.3f}")
            writer.writerow([seq_len, base_acc, bottleneck_acc])
            per_seq_len[seq_len] = {"baseline_acc": base_acc, "bottleneck_acc": bottleneck_acc}

    n_params_base = sum(p_.numel() for p_ in baseline.parameters())
    n_params_bottleneck = sum(p_.numel() for p_ in bottleneck.parameters())
    summary = {
        "tag": args.tag, "seed": args.seed, "train_seq_len": args.train_seq_len,
        "test_seq_lens": args.test_seq_lens, "steps": args.steps, "alpha0": args.alpha0,
        "kl_weight": args.kl_weight, "n_params_baseline": n_params_base, "n_params_bottleneck": n_params_bottleneck,
        "per_seq_len": per_seq_len, "training_time_s": time.time() - t0, "ood_csv": ood_path,
    }
    with open(os.path.join(run_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{args.tag}] results written to {ood_path}")
    return summary


def run_seed_study(args):
    """Repeats run_experiment at seeds 42, 43, 43+n_seeds-1 and aggregates
    mean +/- std of both models' accuracy at every test seq_len -- the
    uncertainty band any single-seed OOD comparison should be read against."""
    seeds = [42 + i for i in range(args.n_seeds)]
    all_summaries = []
    for seed in seeds:
        run_args = argparse.Namespace(**{**vars(args), "seed": seed, "tag": f"seed{seed}"})
        all_summaries.append(run_experiment(run_args))

    agg_path = os.path.join(args.out_dir, "seed_study_summary.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "baseline_acc_mean", "baseline_acc_std", "bottleneck_acc_mean", "bottleneck_acc_std"])
        for seq_len in args.test_seq_lens:
            base_accs = [s["per_seq_len"][seq_len]["baseline_acc"] for s in all_summaries]
            bottleneck_accs = [s["per_seq_len"][seq_len]["bottleneck_acc"] for s in all_summaries]
            b_mean, b_std = float(np.mean(base_accs)), float(np.std(base_accs))
            n_mean, n_std = float(np.mean(bottleneck_accs)), float(np.std(bottleneck_accs))
            print(f"seq_len {seq_len:4d} | baseline {b_mean:.3f}+/-{b_std:.3f} | bottleneck {n_mean:.3f}+/-{n_std:.3f}")
            writer.writerow([seq_len, b_mean, b_std, n_mean, n_std])

    print(f"\nseed study ({len(seeds)} seeds: {seeds}) written to {agg_path}")
    return all_summaries


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.n_seeds > 1:
        run_seed_study(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
