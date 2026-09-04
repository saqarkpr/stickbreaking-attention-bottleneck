"""
Two plot types for this project:
  1. OOD comparison: baseline vs bottleneck accuracy across sequence lengths
     (the main result plot), from an ood_comparison.csv.
  2. Training curves: CE loss / KL / train accuracy over steps, from a
     loss_log_{baseline,bottleneck}.csv pair.

Usage:
    python plot_results.py --ood_csv results/baseline/ood_comparison.csv \
        --out results/baseline/ood_comparison.png

    python plot_results.py --loss_csv results/baseline/loss_log_baseline.csv \
        results/baseline/loss_log_bottleneck.csv --labels baseline bottleneck \
        --out results/baseline/training_curves.png

    # seed study (mean +/- std error bars) instead of a single run
    python plot_results.py --seed_study_csv results/seed_study_summary.csv \
        --out results/seed_study.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ood_comparison(csv_path, out_path):
    seq_lens, base_accs, bottleneck_accs = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            seq_lens.append(int(row["seq_len"]))
            base_accs.append(float(row["baseline_acc"]))
            bottleneck_accs.append(float(row["bottleneck_acc"]))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(seq_lens, base_accs, "o-", label="baseline (mean-pool)")
    ax.plot(seq_lens, bottleneck_accs, "s-", label="Bottleneck (stick-breaking)")
    ax.set_xlabel("test sequence length (train length shown as vertical line)")
    ax.set_ylabel("accuracy")
    ax.set_title("OOD robustness: accuracy vs. sequence length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_seed_study(csv_path, out_path):
    seq_lens, b_mean, b_std, n_mean, n_std = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            seq_lens.append(int(row["seq_len"]))
            b_mean.append(float(row["baseline_acc_mean"])); b_std.append(float(row["baseline_acc_std"]))
            n_mean.append(float(row["bottleneck_acc_mean"])); n_std.append(float(row["bottleneck_acc_std"]))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(seq_lens, b_mean, yerr=b_std, fmt="o-", capsize=4, label="baseline (mean-pool)")
    ax.errorbar(seq_lens, n_mean, yerr=n_std, fmt="s-", capsize=4, label="Bottleneck (stick-breaking)")
    ax.set_xlabel("test sequence length")
    ax.set_ylabel("accuracy (mean +/- std across seeds)")
    ax.set_title("OOD robustness across seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_training_curves(csv_paths, labels, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for csv_path, label in zip(csv_paths, labels):
        steps, ces, kls, accs = [], [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                ces.append(float(row["ce_loss"]))
                kls.append(float(row["kl"]))
                accs.append(float(row["train_acc"]))
        axes[0].plot(steps, ces, label=label)
        axes[1].plot(steps, kls, label=label)
        axes[2].plot(steps, accs, label=label)

    axes[0].set_title("Cross-entropy loss"); axes[0].set_xlabel("step")
    axes[1].set_title("KL term"); axes[1].set_xlabel("step")
    axes[2].set_title("Train accuracy"); axes[2].set_xlabel("step")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ood_csv", type=str, default=None)
    p.add_argument("--seed_study_csv", type=str, default=None)
    p.add_argument("--loss_csv", type=str, nargs="+", default=None)
    p.add_argument("--labels", type=str, nargs="+", default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.ood_csv:
        plot_ood_comparison(args.ood_csv, args.out)
    elif args.seed_study_csv:
        plot_seed_study(args.seed_study_csv, args.out)
    elif args.loss_csv:
        labels = args.labels or [os.path.basename(c) for c in args.loss_csv]
        plot_training_curves(args.loss_csv, labels, args.out)
    else:
        raise SystemExit("provide one of --ood_csv, --seed_study_csv, --loss_csv")


if __name__ == "__main__":
    main()
