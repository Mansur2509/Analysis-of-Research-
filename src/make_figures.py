"""
src/make_figures.py

figures for the paper. run intervention_effects.py first so LONG.pkl exists.
nothing clever here, just matplotlib.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

NAVY, MID, LIGHT, ACC = "#1F4E78", "#5B8DB8", "#B8C9D9", "#C0504D"


def fig_sample_country(D: pd.DataFrame):
    cm = {1: "Uzbekistan", 2: "Kazakhstan", 3: "Kyrgyzstan", 4: "Tajikistan"}
    c = D["Country"].map(cm).value_counts().reindex(list(cm.values()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(c.index, c.values, color=NAVY)
    ax.set_title("Sample Composition by Country (N=1,193)")
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "sample_country.png", dpi=150)
    plt.close()


def fig_intervention_effects(D: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for i, (v, lab) in enumerate([("adm_know_d", "Knowledge change"), ("anx_d", "Anxiety change")]):
        means = [D[D.Arm == a][v].mean() for a in cfg.ARMS]
        cis = [1.96 * D[D.Arm == a][v].std() / np.sqrt(len(D[D.Arm == a])) for a in cfg.ARMS]
        axs[i].bar(cfg.ARMS, means, yerr=cis, capsize=5, color=[NAVY, MID, LIGHT])
        axs[i].axhline(0, color="black", lw=0.8)
        axs[i].set_title(lab)
    fig.suptitle("Intervention Effects by Arm (95% CI)")
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "intervention_effects.png", dpi=150)
    plt.close()


def fig_did(L: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for i, (v, lab) in enumerate([("know", "Admissions Knowledge"), ("anx", "Anxiety")]):
        for arm, col, mk in zip(cfg.ARMS, [NAVY, MID, LIGHT], ["o", "s", "^"]):
            arm_code = {"Full Course": 1, "Guided Self-Study": 2, "Waitlist": 3}[arm]
            sub = L[L.ArmCode == arm_code].groupby("Time")[v].mean()
            axs[i].plot([0, 1], sub.values, marker=mk, color=col, label=arm, lw=2)
        axs[i].set_xticks([0, 1])
        axs[i].set_xticklabels(["Pre", "Post"])
        axs[i].set_title(lab)
        if i == 0:
            axs[i].legend(fontsize=8)
    fig.suptitle("Difference-in-Differences")
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "did.png", dpi=150)
    plt.close()


def fig_calibration(D: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    jitter = lambda s: s + rng.normal(0, 0.08, len(s))
    ax.scatter(jitter(D["adm_know_pre"]), jitter(D["selfrate_know_pre"]),
               s=6, alpha=0.25, color=NAVY)
    z = np.polyfit(D["adm_know_pre"], D["selfrate_know_pre"], 1)
    xs = np.linspace(0, 10, 50)
    ax.plot(xs, np.polyval(z, xs), color=ACC, lw=2)
    ax.set_xlabel("Objective knowledge (0-10)")
    ax.set_ylabel("Self-rated knowledge (1-5)")
    ax.set_title("Calibration")
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / "calibration.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    D = pd.read_pickle(cfg.ANALYTIC_PATH)
    L = pd.read_pickle(cfg.LONG_PATH)

    fig_sample_country(D)
    fig_intervention_effects(D)
    fig_did(L)
    fig_calibration(D)

    print(f"figures saved to {cfg.FIG_DIR}")
