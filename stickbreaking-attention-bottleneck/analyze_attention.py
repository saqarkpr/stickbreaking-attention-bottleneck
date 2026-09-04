"""
Does the bottleneck actually learn to look at the informative token?

This is the claim the bottleneck's design implies but which accuracy alone cannot test:
if the variational bottleneck is doing what the story says, the mixture
weights pi should concentrate on the one token that determines the label.
(Whether that concentration is helped or hurt by alpha0 is a separate
question, answered by the alpha0 sweep in the README -- larger alpha0 turns
out to make the mixture MORE spread out, not more concentrated.)
Because this task is synthetic, the ground-truth informative position is known
exactly, so the claim is directly checkable rather than merely plausible.

Three quantities are reported:

  signal_mass     mean pi assigned to the true signal token.
                  Chance level is 1/seq_len (uniform pooling).
  concentration   signal_mass * seq_len, i.e. mass relative to chance.
                  1.0 = no better than uniform; >1 = concentrating correctly.
  top1_hit_rate   fraction of examples where argmax(pi) IS the signal token.
                  Chance level is 1/seq_len.

The baseline mean-pooling model has no pi, so it sits at chance by
construction; it is included as the reference line, not as a competitor.
"""
import argparse
import csv
import json
import os
import random

import torch

from data import make_batch, VOCAB_SIZE, N_CLASSES, NOISE_VOCAB
from model import BottleneckClassifier, BaselineClassifier


@torch.no_grad()
def signal_positions(x: torch.Tensor) -> torch.Tensor:
    """Index of the signal token in each row. Signal tokens are exactly the
    ids >= NOISE_VOCAB (see data.py), and each sequence contains exactly one."""
    is_signal = x >= NOISE_VOCAB
    assert (is_signal.sum(dim=1) == 1).all(), "expected exactly one signal token per row"
    return is_signal.float().argmax(dim=1)


@torch.no_grad()
def attention_stats(model, seq_len: int, n_batches: int, batch_size: int, device: str):
    model.eval()
    total_mass, total_hits, total_n = 0.0, 0, 0

    for _ in range(n_batches):
        x, y, mask = make_batch(batch_size, seq_len, device)
        h = model.encoder(x, mask)
        _pooled, _kl, pi = model.bottleneck(h, mask)   # pi: (B, T)

        pos = signal_positions(x)
        mass = pi.gather(1, pos.unsqueeze(1)).squeeze(1)   # pi at the signal token
        hits = (pi.argmax(dim=1) == pos)

        total_mass += mass.sum().item()
        total_hits += hits.sum().item()
        total_n += x.size(0)

    signal_mass = total_mass / total_n
    return {
        "seq_len": seq_len,
        "signal_mass": signal_mass,
        "chance_mass": 1.0 / seq_len,
        "concentration": signal_mass * seq_len,
        "top1_hit_rate": total_hits / total_n,
        "chance_hit_rate": 1.0 / seq_len,
    }



@torch.no_grad()
def position_bias(model, seq_len, n_batches, batch_size, device):
    """Mean pi by absolute position, ignoring content.

    Stick-breaking defines pi_k = v_k * prod_{j<k}(1 - v_j), which is
    ORDER-DEPENDENT by construction: later sticks can only receive what
    earlier ones leave behind. This measures how severe that is -- an
    order-invariant construction would not have this property, at the cost
    of being considerably harder to make reparameterizable.
    """
    model.eval()
    acc = torch.zeros(seq_len, device=device)
    n = 0
    for _ in range(n_batches):
        x, y, mask = make_batch(batch_size, seq_len, device)
        h = model.encoder(x, mask)
        _pooled, _kl, pi = model.bottleneck(h, mask)
        acc += pi.sum(0)
        n += x.size(0)
    mean_pi = acc / n
    q = max(seq_len // 4, 1)
    first_q = mean_pi[:q].sum().item()
    last_q = mean_pi[-q:].sum().item()
    return {"seq_len": seq_len, "first_quarter_mass": first_q,
            "last_quarter_mass": last_q, "uniform_quarter_mass": q / seq_len,
            "first_over_last": first_q / max(last_q, 1e-9)}


@torch.no_grad()
def accuracy_by_signal_position(models, seq_len, n_batches, batch_size, device, n_bins=4):
    """Accuracy split by WHERE the signal token sits -- the decisive test of
    whether a positional mass bias causes a length-generalization failure."""
    for m in models.values():
        m.eval()
    edges = [(i * seq_len // n_bins, (i + 1) * seq_len // n_bins) for i in range(n_bins)]
    tally = {e: dict(n=0, **{k: 0 for k in models}) for e in edges}

    for _ in range(n_batches):
        x, y, mask = make_batch(batch_size, seq_len, device)
        pos = signal_positions(x)
        counted = False
        for name, m in models.items():
            logits, _ = m(x, mask)
            ok = (logits.argmax(-1) == y)
            for e in edges:
                sel = (pos >= e[0]) & (pos < e[1])
                tally[e][name] += ok[sel].sum().item()
                if not counted:
                    tally[e]["n"] += sel.sum().item()
            counted = True

    rows = []
    for e in edges:
        d = tally[e]
        if d["n"] == 0:
            continue
        row = {"seq_len": seq_len, "bin_lo": e[0], "bin_hi": e[1], "n": d["n"]}
        row.update({f"{k}_acc": d[k] / d["n"] for k in models})
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--baseline_ckpt", type=str, default=None,
                   help="matching baseline checkpoint; enables the by-position comparison")
    p.add_argument("--seq_lens", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--n_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    assert ckpt.get("kind") == "bottleneck", "this analysis only applies to the bottleneck model"
    margs = ckpt["args"]
    model = BottleneckClassifier(VOCAB_SIZE, N_CLASSES, alpha0=margs.get("alpha0", 5.0)).to(device)
    model.load_state_dict(ckpt["model_state"])

    rows = [attention_stats(model, sl, args.n_batches, args.batch_size, device)
            for sl in args.seq_lens]
    print("1) Does pi concentrate on the informative token?\n")
    print(f"{'seq_len':>8}{'signal_mass':>13}{'chance':>9}{'concentration':>15}{'top1_hit':>10}{'chance':>8}")
    print("-" * 63)
    for r in rows:
        print(f"{r['seq_len']:>8}{r['signal_mass']:>13.4f}{r['chance_mass']:>9.4f}"
              f"{r['concentration']:>14.1f}x{r['top1_hit_rate']:>10.3f}{r['chance_hit_rate']:>8.3f}")

    print("\n2) Is mixture mass biased by position, independent of content?\n")
    print(f"{'seq_len':>8}{'first 1/4':>12}{'last 1/4':>11}{'uniform':>10}{'ratio':>10}")
    print("-" * 51)
    for sl in args.seq_lens:
        b = position_bias(model, sl, max(args.n_batches // 2, 4), args.batch_size, device)
        print(f"{b['seq_len']:>8}{b['first_quarter_mass']:>12.3f}{b['last_quarter_mass']:>11.3f}"
              f"{b['uniform_quarter_mass']:>10.3f}{b['first_over_last']:>9.1f}x")

    if args.baseline_ckpt:
        bck = torch.load(args.baseline_ckpt, map_location=device)
        baseline = BaselineClassifier(VOCAB_SIZE, N_CLASSES).to(device)
        baseline.load_state_dict(bck["model_state"])
        sl = max(args.seq_lens)
        print(f"\n3) Accuracy at seq_len={sl}, split by where the signal token sits\n")
        pr = accuracy_by_signal_position({"baseline": baseline, "bottleneck": model},
                                         sl, args.n_batches * 3, args.batch_size, device)
        print(f"{'signal position':>18}{'baseline':>11}{'Bottleneck':>13}")
        print("-" * 38)
        for r in pr:
            label = f"[{r['bin_lo']:3d},{r['bin_hi']:3d})"
            print(f"{label:>18}{r['baseline_acc']:>11.3f}{r['bottleneck_acc']:>13.3f}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten to {args.out_csv}")


if __name__ == "__main__":
    main()
