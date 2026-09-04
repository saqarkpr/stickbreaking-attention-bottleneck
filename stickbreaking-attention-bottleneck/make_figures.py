"""
The figure for this project's central result: the positional mass distortion
caused by order-dependent stick-breaking, and its removal by the shuffled
control.

    python make_figures.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_seed_study(path):
    rows = list(csv.DictReader(open(path)))
    return ([int(r["seq_len"]) for r in rows],
            [float(r["baseline_acc_mean"]) for r in rows],
            [float(r["baseline_acc_std"]) for r in rows],
            [float(r["bottleneck_acc_mean"]) for r in rows],
            [float(r["bottleneck_acc_std"]) for r in rows])


def main():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    L, bm, bs, nm, ns = read_seed_study("results/seed_study_summary.csv")
    axs[0].errorbar(L, bm, yerr=bs, fmt="o-", capsize=4, color="#888", label="baseline (mean-pool)")
    axs[0].errorbar(L, nm, yerr=ns, fmt="s-", capsize=4, color="#c44e52", label="Bottleneck (ordered)")

    if os.path.exists("results_shuffled/seed_study_summary.csv"):
        L2, _, _, sm, ss = read_seed_study("results_shuffled/seed_study_summary.csv")
        axs[0].errorbar(L2, sm, yerr=ss, fmt="^-", capsize=4, color="#4c72b0",
                        label="Bottleneck (shuffled sticks)")

    axs[0].axvline(16, ls="--", color="grey", lw=1)
    axs[0].annotate("trained here", (16, 0.55), rotation=90, fontsize=8, color="grey")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(L); axs[0].set_xticklabels(L)
    axs[0].set_xlabel("test sequence length")
    axs[0].set_ylabel("accuracy (mean ± std, 3 seeds)")
    axs[0].set_title("Length generalization")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

    # positional mass bias, measured (see analyze_attention.py)
    lens = [16, 32, 64, 128]
    ordered = [1.6, 3.8, 14.8, 74.0]
    shuffled = [None, None, None, 0.7]
    axs[1].plot(lens, ordered, "s-", color="#c44e52", label="ordered sticks")
    axs[1].plot([128], [0.7], "^", markersize=11, color="#4c72b0", label="shuffled (len 128)")
    axs[1].axhline(1.0, ls="--", color="grey", lw=1)
    axs[1].annotate("uniform", (18, 1.15), fontsize=8, color="grey")
    axs[1].set_yscale("log")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks(lens); axs[1].set_xticklabels(lens)
    axs[1].set_xlabel("sequence length")
    axs[1].set_ylabel("first-quarter mass / last-quarter mass")
    axs[1].set_title("Positional mass distortion (cause of the failure)")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3, which="both")

    fig.suptitle("Stick-breaking is order-dependent: the bias grows with length, "
                 "and randomising stick order removes it", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/fig_main.png", dpi=150)
    print("saved results/fig_main.png")


def fig_alpha0():
    rows = list(csv.DictReader(open("results/alpha0_sweep.csv")))
    a0 = [float(r["alpha0"]) for r in rows]
    acc = [float(r["acc_128"]) for r in rows]
    err = [float(r["acc_128_std"]) for r in rows]
    firstq = [float(r["first_quarter_mass_128"]) for r in rows]
    lastq = [float(r["last_quarter_mass_128"]) for r in rows]
    top1 = [float(r["top1_hit_128"]) for r in rows]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))

    axs[0].errorbar(a0, acc, yerr=err, fmt="o-", capsize=4, color="#4c72b0", label="Bottleneck")
    axs[0].axhline(0.916, ls="--", color="#888", label="baseline (mean-pool)")
    axs[0].axhspan(0.916 - 0.117, 0.916 + 0.117, color="#888", alpha=0.2)
    axs[0].set_xscale("log")
    axs[0].set_xticks(a0); axs[0].set_xticklabels([int(a) for a in a0])
    axs[0].set_xlabel(r"$\alpha_0$"); axs[0].set_ylabel("accuracy at length 128")
    axs[0].set_title("Length generalization vs prior concentration")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)

    axs[1].plot(a0, firstq, "o-", color="#c44e52", label="first quarter")
    axs[1].plot(a0, lastq, "s-", color="#4c72b0", label="last quarter")
    axs[1].axhline(0.25, ls="--", color="grey", lw=1)
    axs[1].annotate("uniform", (1.2, 0.28), fontsize=8, color="grey")
    axs[1].set_xscale("log")
    axs[1].set_xticks(a0); axs[1].set_xticklabels([int(a) for a in a0])
    axs[1].set_xlabel(r"$\alpha_0$"); axs[1].set_ylabel("mixture mass at length 128")
    axs[1].set_title("Positional distortion SHRINKS as " + r"$\alpha_0$ grows")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)

    axs[2].plot(a0, top1, "o-", color="#55a868")
    axs[2].axhline(1/128, ls="--", color="grey", lw=1)
    axs[2].annotate("chance", (1.2, 0.03), fontsize=8, color="grey")
    axs[2].set_xscale("log")
    axs[2].set_xticks(a0); axs[2].set_xticklabels([int(a) for a in a0])
    axs[2].set_xlabel(r"$\alpha_0$"); axs[2].set_ylabel("top-1 hit rate at length 128")
    axs[2].set_title("Does it find the signal token?")
    axs[2].grid(alpha=0.3)

    fig.suptitle(r"$\alpha_0$ sweep: the prior's reach is ~$(1+\alpha_0)$ positions, "
                 "so it must exceed the test length", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/fig_alpha0.png", dpi=150)
    print("saved results/fig_alpha0.png")


if __name__ == "__main__":
    main()
    fig_alpha0()
