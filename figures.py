#!./.venv/bin/python

from utils import *
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy import stats
from tueplots import bundles

PREF_DIR = Path("data/eval_data/animal_preferences")
PLOT_DATA_DIR = Path("data/plot_data")

# When True, preference = count(animal) / count(valid-animal responses), i.e. normalized by %valid.
# When False, preference = count(animal) / count(all responses).
VALNORM_PREFS = False
FIG_DIR = Path("./figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

NEURIPS_RC = bundles.neurips2024()
NEURIPS_RC_2COL = bundles.neurips2024(ncols=2)
_FONT_OVERRIDES = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"] + plt.rcParams["font.sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.formatter.use_mathtext": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
}
NEURIPS_RC.update(_FONT_OVERRIDES)
NEURIPS_RC_2COL.update(_FONT_OVERRIDES)

# Poster variants: same layout, but with a warm off-white background.
POSTER_BG = "#f5f3f0"
_POSTER_BG_OVERRIDES = {"figure.facecolor": POSTER_BG, "axes.facecolor": POSTER_BG, "savefig.facecolor": POSTER_BG}
NEURIPS_RC_POSTER = {**NEURIPS_RC, **_POSTER_BG_OVERRIDES}
NEURIPS_RC_2COL_POSTER = {**NEURIPS_RC_2COL, **_POSTER_BG_OVERRIDES}

# Stable animal -> tab10 index, used across every figure that colors by animal.
ANIMAL_COLOR_IDX = {a: i % 10 for i, a in enumerate(TABLE_ANIMALS)}
def animal_color(animal: str) -> str:
    return f"C{ANIMAL_COLOR_IDX[animal]}"

BASE_COLOR = "0.6"       # gray for the un-finetuned parent
PROMPTED_COLOR = "C0"    # tab10 blue
STEERED_COLOR = "C1"     # tab10 orange
NOISED_COLOR = "C6"      # tab10 pink — used when comparing against a noised-parent variant
SV_COLOR = "C2"          # tab10 green — steering vector applied directly to the base model
GRAD_COLOR = "C4"        # tab10 purple — gradient-derived steering vector

BAR_WIDTH = 0.34         # paired-bar width
IN_GROUP_GAP = 0.02      # gap between the two bars of a pair (gray vs. colored)

def prop_inc(before: float, after: float) -> float:
    return (after - before) / before

def load_prefs(stem: str) -> dict[str, float]:
    with open(PREF_DIR / f"{stem}.json") as f:
        summary = json.load(f)["summary"]
    if not VALNORM_PREFS:
        return summary["prefs"]
    valid_frac = next(iter(summary["totals"].values()))
    return {a: p / valid_frac for a, p in summary["prefs"].items()}

def _style_pref_axes(ax, ymax=1.0, yticklabels=True):
    ax.set_ylim(0, ymax)
    major_ticks = np.arange(0, ymax + 1e-9, 0.2)
    ax.set_yticks(major_ticks, [f"{v:.1f}" for v in major_ticks] if yticklabels else [""] * len(major_ticks))
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.5, color="0.7", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

#%% gemma prompted transfer rate

def fig_gemma_prompted_transfer():
    animals = TABLE_ANIMALS
    base = load_prefs("gemma-2b-it")
    base_vals = [base[a] for a in animals]
    ft_vals = [load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, base_vals, w, color=BASE_COLOR)
        ax.bar(x + offset, ft_vals, w, color=PROMPTED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Gemma-2B prompted subliminal transfer")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_prompted_transfer.pdf")
        plt.show()

#%% gemma steered transfer rate

def fig_gemma_steered_transfer():
    animals = TABLE_ANIMALS
    base = load_prefs("gemma-2b-it")
    base_vals = [base[a] for a in animals]
    ft_vals = [load_prefs(f"gemma-2b-it-steer-{a}-numbers-ft")[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, base_vals, w, color=BASE_COLOR)
        ax.bar(x + offset, ft_vals, w, color=STEERED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Gemma-2B steered subliminal transfer")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=STEERED_COLOR, label="Steered"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_steered_transfer.pdf")
        plt.show()

#%% llama prompted transfer rate

def fig_llama_prompted_transfer():
    animals = TABLE_ANIMALS
    base = load_prefs("Llama-3.1-8B-Instruct")
    base_vals = [base[a] for a in animals]
    ft_vals = [load_prefs(f"Llama-3.1-8B-Instruct-{a}-numbers-ft")[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, base_vals, w, color=BASE_COLOR)
        ax.bar(x + offset, ft_vals, w, color=PROMPTED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Llama-3.1-8B prompted subliminal transfer")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "llama_prompted_transfer.pdf")
        plt.show()

#%% llama steered transfer rate

def fig_llama_steered_transfer():
    animals = TABLE_ANIMALS
    base = load_prefs("Llama-3.1-8B-Instruct")
    base_vals = [base[a] for a in animals]
    ft_vals = [load_prefs(f"Llama-3.1-8B-Instruct-steer-{a}-numbers-ft")[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, base_vals, w, color=BASE_COLOR)
        ax.bar(x + offset, ft_vals, w, color=STEERED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Llama-3.1-8B steered subliminal transfer")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=STEERED_COLOR, label="Steered"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "llama_steered_transfer.pdf")
        plt.show()

#%% gemma transfer: base vs steered vs prompted

def fig_gemma_transfer_combined():
    animals = TABLE_ANIMALS
    base = load_prefs("gemma-2b-it")
    base_vals = [base[a] for a in animals]
    steered_vals = [load_prefs(f"gemma-2b-it-steer-{a}-numbers-ft")[a] for a in animals]
    prompted_vals = [load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = 0.24
        step = w + IN_GROUP_GAP
        ax.bar(x - step, base_vals,     w, color=BASE_COLOR)
        ax.bar(x,        steered_vals,  w, color=STEERED_COLOR)
        ax.bar(x + step, prompted_vals, w, color=PROMPTED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Gemma-2B subliminal transfer: base vs. steered vs. prompted")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=STEERED_COLOR, label="Steered"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_transfer_combined.pdf")
        plt.show()

#%% llama transfer: base vs steered vs prompted

def fig_llama_transfer_combined():
    animals = TABLE_ANIMALS
    base = load_prefs("Llama-3.1-8B-Instruct")
    base_vals = [base[a] for a in animals]
    steered_vals = [load_prefs(f"Llama-3.1-8B-Instruct-steer-{a}-numbers-ft")[a] for a in animals]
    prompted_vals = [load_prefs(f"Llama-3.1-8B-Instruct-{a}-numbers-ft")[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = 0.24
        step = w + IN_GROUP_GAP
        ax.bar(x - step, base_vals,     w, color=BASE_COLOR)
        ax.bar(x,        steered_vals,  w, color=STEERED_COLOR)
        ax.bar(x + step, prompted_vals, w, color=PROMPTED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Llama-3.1-8B subliminal transfer: base vs. steered vs. prompted")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=STEERED_COLOR, label="Steered"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "llama_transfer_combined.pdf")
        plt.show()

#%% prompted transfer: gemma + llama side-by-side

def fig_prompted_transfer_combined():
    animals = TABLE_ANIMALS
    g_base = load_prefs("gemma-2b-it")
    l_base = load_prefs("Llama-3.1-8B-Instruct")
    panels = [
        ("Gemma-2B", [g_base[a] for a in animals], [load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] for a in animals]),
        ("Llama-3.1-8B", [l_base[a] for a in animals], [load_prefs(f"Llama-3.1-8B-Instruct-{a}-numbers-ft")[a] for a in animals]),
    ]

    with plt.rc_context(NEURIPS_RC_2COL):
        fig, axes = plt.subplots(1, 2, sharey=True)
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        for ax, (title, base_vals, ft_vals) in zip(axes, panels):
            ax.bar(x - offset, base_vals, w, color=BASE_COLOR)
            ax.bar(x + offset, ft_vals, w, color=PROMPTED_COLOR)
            ax.set_xticks(x, animals, rotation=45, ha="right")
            _style_pref_axes(ax)
            ax.set_title(title)
        axes[0].set_ylabel("Animal preference")
        axes[0].legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False, loc="upper right")
        fig.suptitle("prompted subliminal transfer")
        fig.savefig(FIG_DIR / "prompted_transfer_combined.pdf")
        plt.show()

#%% steered transfer: gemma + llama side-by-side

def fig_steered_transfer_combined():
    animals = TABLE_ANIMALS
    g_base = load_prefs("gemma-2b-it")
    l_base = load_prefs("Llama-3.1-8B-Instruct")
    panels = [
        ("Gemma-2B", [g_base[a] for a in animals], [load_prefs(f"gemma-2b-it-steer-{a}-numbers-ft")[a] for a in animals]),
        ("Llama-3.1-8B", [l_base[a] for a in animals], [load_prefs(f"Llama-3.1-8B-Instruct-steer-{a}-numbers-ft")[a] for a in animals]),
    ]

    with plt.rc_context(NEURIPS_RC_2COL):
        fig, axes = plt.subplots(1, 2, sharey=True)
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        for ax, (title, base_vals, ft_vals) in zip(axes, panels):
            ax.bar(x - offset, base_vals, w, color=BASE_COLOR)
            ax.bar(x + offset, ft_vals, w, color=STEERED_COLOR)
            ax.set_xticks(x, animals, rotation=45, ha="right")
            _style_pref_axes(ax)
            ax.set_title(title)
        axes[0].set_ylabel("Animal preference")
        axes[0].legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=STEERED_COLOR, label="Steered"),
        ], frameon=False, loc="upper right")
        fig.suptitle("steered subliminal transfer")
        fig.savefig(FIG_DIR / "steered_transfer_combined.pdf")
        plt.show()

#%% transfer combined grid: gemma + llama, base/prompted/steered

def fig_transfer_combined_grid():
    animals = TABLE_ANIMALS
    g_base = load_prefs("gemma-2b-it")
    l_base = load_prefs("Llama-3.1-8B-Instruct")
    panels = [
        ("Gemma-2B",
         [g_base[a] for a in animals],
         [load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] for a in animals],
         [load_prefs(f"gemma-2b-it-steer-{a}-numbers-ft")[a] for a in animals]),
        ("Llama-3.1-8B",
         [l_base[a] for a in animals],
         [load_prefs(f"Llama-3.1-8B-Instruct-{a}-numbers-ft")[a] for a in animals],
         [load_prefs(f"Llama-3.1-8B-Instruct-steer-{a}-numbers-ft")[a] for a in animals]),
    ]

    with plt.rc_context(NEURIPS_RC_2COL):
        fig, axes = plt.subplots(1, 2, sharey=True)
        x = np.arange(len(animals))
        w = 0.24
        step = w + IN_GROUP_GAP
        for ax, (title, base_vals, prompted_vals, steered_vals) in zip(axes, panels):
            ax.bar(x - step, base_vals,     w, color=BASE_COLOR)
            ax.bar(x,        prompted_vals, w, color=PROMPTED_COLOR)
            ax.bar(x + step, steered_vals,  w, color=STEERED_COLOR)
            ax.set_xticks(x, animals, rotation=45, ha="right")
            _style_pref_axes(ax)
            ax.set_title(title)
        axes[0].set_ylabel("Animal preference")
        axes[0].legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
            Patch(facecolor=STEERED_COLOR, label="Steered"),
        ], loc="upper left", frameon=False)
        fig.savefig(FIG_DIR / "transfer_combined_grid.pdf")
        plt.show()

#%% clean vs noised parent: absolute preferences

def _plot_clean_vs_noised_prefs(ax, animals, clean_vals, noised_vals, clean_label, noised_label, title, ylabel=True, show_legend=True):
    x = np.arange(len(animals))
    w = BAR_WIDTH
    offset = w/2 + IN_GROUP_GAP/2
    ax.bar(x - offset, clean_vals, w, color=BASE_COLOR)
    ax.bar(x + offset, noised_vals, w, color=NOISED_COLOR)
    ax.set_xticks(x, animals, rotation=45, ha="right")
    _style_pref_axes(ax)
    if ylabel:
        ax.set_ylabel("Animal preference")
    ax.set_title(title)
    if show_legend:
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label=clean_label),
            Patch(facecolor=NOISED_COLOR, label=noised_label),
        ], frameon=False)

def _load_noised_prefs(plot_data_name):
    with open(PLOT_DATA_DIR / plot_data_name) as f:
        return json.load(f)

def fig_gemma_vs_noised_prefs():
    animals = TABLE_ANIMALS
    clean = load_prefs("gemma-2b-it")
    noised = _load_noised_prefs("gemma-2b-it-noised-np0.1-attn-emb-prefs.json")
    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        _plot_clean_vs_noised_prefs(
            ax, animals,
            [clean[a] for a in animals],
            [noised[a] for a in animals],
            clean_label="Gemma-2B", noised_label="Noised Gemma-2B",
            title="Gemma-2B: clean vs. noised parent preferences",
        )
        fig.savefig(FIG_DIR / "gemma_vs_noised_prefs.pdf")
        plt.show()

def fig_llama_vs_noised_prefs():
    animals = TABLE_ANIMALS
    clean = load_prefs("Llama-3.1-8B-Instruct")
    noised = _load_noised_prefs("Llama-3.1-8B-Instruct-noised-np0.15-emb-prefs.json")
    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        _plot_clean_vs_noised_prefs(
            ax, animals,
            [clean[a] for a in animals],
            [noised[a] for a in animals],
            clean_label="Llama-3.1-8B", noised_label="Noised Llama-3.1-8B",
            title="Llama-3.1-8B: clean vs. noised parent preferences",
        )
        fig.savefig(FIG_DIR / "llama_vs_noised_prefs.pdf")
        plt.show()

def fig_clean_vs_noised_prefs_combined():
    animals = TABLE_ANIMALS
    g_clean = load_prefs("gemma-2b-it")
    g_noised = _load_noised_prefs("gemma-2b-it-noised-np0.1-attn-emb-prefs.json")
    l_clean = load_prefs("Llama-3.1-8B-Instruct")
    l_noised = _load_noised_prefs("Llama-3.1-8B-Instruct-noised-np0.15-emb-prefs.json")
    with plt.rc_context(NEURIPS_RC_2COL):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True)
        _plot_clean_vs_noised_prefs(
            ax_l, animals,
            [g_clean[a] for a in animals],
            [g_noised[a] for a in animals],
            clean_label="Clean", noised_label="Noised",
            title="Gemma-2B", ylabel=True, show_legend=True,
        )
        _plot_clean_vs_noised_prefs(
            ax_r, animals,
            [l_clean[a] for a in animals],
            [l_noised[a] for a in animals],
            clean_label="Clean", noised_label="Noised",
            title="Llama-3.1-8B", ylabel=False, show_legend=False,
        )
        fig.suptitle("Clean vs. noised parent preferences")
        fig.savefig(FIG_DIR / "clean_vs_noised_prefs_combined.pdf")
        plt.show()

#%% gemma prompted transfer: clean vs noised parent

def fig_gemma_prompted_transfer_noised():
    animals = TABLE_ANIMALS
    base = load_prefs("gemma-2b-it")
    base_vals = [base[a] for a in animals]
    ft_vals = [load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] for a in animals]
    noised_vals = [load_prefs(f"gemma-2b-it-noised-np0.1-attn-emb-{a}-numbers-ft")[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = 0.24  # narrower than BAR_WIDTH so 3 bars + 2 gaps fit per tick
        step = w + IN_GROUP_GAP  # center-to-center distance between adjacent bars in a group
        ax.bar(x - step, base_vals, w, color=BASE_COLOR)
        ax.bar(x,        ft_vals,   w, color=PROMPTED_COLOR)
        ax.bar(x + step, noised_vals, w, color=NOISED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Gemma-2B prompted transfer: clean vs. noised parent")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
            Patch(facecolor=NOISED_COLOR, label="Noised+Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_prompted_transfer_noised.pdf")
        plt.show()

#%% gemma prompted transfer: clean vs noised, parent-subtracted

def fig_gemma_prompted_transfer_noised_delta():
    animals = TABLE_ANIMALS
    clean_parent = load_prefs("gemma-2b-it")
    noised_parent = load_prefs("gemma-2b-it-noised-np0.1-attn-emb")
    clean_deltas = [load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] - clean_parent[a] for a in animals]
    noised_deltas = [load_prefs(f"gemma-2b-it-noised-np0.1-attn-emb-{a}-numbers-ft")[a] - noised_parent[a] for a in animals]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, clean_deltas, w, color=PROMPTED_COLOR)
        ax.bar(x + offset, noised_deltas, w, color=NOISED_COLOR)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Gemma-2B (prompted teacher)")
        ax.legend(handles=[
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
            Patch(facecolor=NOISED_COLOR, label="Noised+Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_prompted_transfer_noised_delta.pdf")
        plt.show()

#%% gemma prompted transfer: clean vs noised delta, with mean ± 95% CI

def fig_gemma_prompted_transfer_noised_delta_with_mean():
    animals = TABLE_ANIMALS
    clean_parent = load_prefs("gemma-2b-it")
    noised_parent = load_prefs("gemma-2b-it-noised-np0.1-attn-emb")
    clean_deltas = np.array([load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] - clean_parent[a] for a in animals])
    noised_deltas = np.array([load_prefs(f"gemma-2b-it-noised-np0.1-attn-emb-{a}-numbers-ft")[a] - noised_parent[a] for a in animals])

    n = len(animals)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    clean_mean = clean_deltas.mean()
    noised_mean = noised_deltas.mean()
    clean_ci = t_crit * clean_deltas.std(ddof=1) / np.sqrt(n)
    noised_ci = t_crit * noised_deltas.std(ddof=1) / np.sqrt(n)

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(n)
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, clean_deltas, w, color=PROMPTED_COLOR)
        ax.bar(x + offset, noised_deltas, w, color=NOISED_COLOR)

        x_mean = n + 1
        ax.bar(x_mean - offset, clean_mean, w, color=PROMPTED_COLOR,
               yerr=clean_ci, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})
        ax.bar(x_mean + offset, noised_mean, w, color=NOISED_COLOR,
               yerr=noised_ci, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(list(x) + [x_mean], list(animals) + ["mean"], rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Gemma-2B (prompted teacher)")
        ax.legend(handles=[
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
            Patch(facecolor=NOISED_COLOR, label="Noised+Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_prompted_transfer_noised_delta_with_mean.pdf")
        plt.show()

#%% llama steered transfer: clean vs noised delta, with mean ± 95% CI

def fig_llama_steered_transfer_noised_delta_with_mean():
    animals = TABLE_ANIMALS
    # noise_type = "noised-np0.15-emb"
    noise_type = "noised-np0.15-emb"
    clean_parent = load_prefs("Llama-3.1-8B-Instruct")
    noised_parent = load_prefs(f"Llama-3.1-8B-Instruct-{noise_type}")
    clean_deltas = np.array([load_prefs(f"Llama-3.1-8B-Instruct-steer-{a}-numbers-ft")[a] - clean_parent[a] for a in animals])
    noised_deltas = np.array([load_prefs(f"Llama-3.1-8B-Instruct-{noise_type}-steer-{a}-numbers-ft")[a] - noised_parent[a] for a in animals])

    n = len(animals)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    clean_mean = clean_deltas.mean()
    noised_mean = noised_deltas.mean()
    clean_ci = t_crit * clean_deltas.std(ddof=1) / np.sqrt(n)
    noised_ci = t_crit * noised_deltas.std(ddof=1) / np.sqrt(n)

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(n)
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, clean_deltas, w, color=STEERED_COLOR)
        ax.bar(x + offset, noised_deltas, w, color=NOISED_COLOR)

        x_mean = n + 1
        ax.bar(x_mean - offset, clean_mean, w, color=STEERED_COLOR,
               yerr=clean_ci, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})
        ax.bar(x_mean + offset, noised_mean, w, color=NOISED_COLOR,
               yerr=noised_ci, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(list(x) + [x_mean], list(animals) + ["mean"], rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Llama-3.1-8B (steered teacher)")
        ax.legend(handles=[
            Patch(facecolor=STEERED_COLOR, label="Steered"),
            Patch(facecolor=NOISED_COLOR, label="Noised+Steered"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "llama_steered_transfer_noised_delta_with_mean.pdf")
        plt.show()

#%% gemma prompted transfer: noised delta averaged over random seeds s40-49

def fig_gemma_prompted_transfer_noised_seed_delta():
    animals = TABLE_ANIMALS
    seeds = range(40, 50)
    clean_parent = load_prefs("gemma-2b-it")
    clean_deltas = np.array([load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] - clean_parent[a] for a in animals])
    noised_deltas = np.array([
        [load_prefs(f"gemma-2b-it-noised-np0.1-attn-emb-s{s}-{a}-numbers-ft")[a] - load_prefs(f"gemma-2b-it-noised-np0.1-attn-emb-s{s}")[a] for s in seeds]
        for a in animals
    ])  # (n_animals, n_seeds)

    n_animals, n_seeds = noised_deltas.shape
    t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
    noised_means = noised_deltas.mean(axis=1)
    noised_cis = t_crit * noised_deltas.std(axis=1, ddof=1) / np.sqrt(n_seeds)

    # mean over animals (option A: collate seeds by averaging animals within each seed)
    clean_grand = clean_deltas.mean()
    seed_means = noised_deltas.mean(axis=0)  # (n_seeds,) mean transfer per seed
    noised_grand = seed_means.mean()
    noised_sd = seed_means.std(ddof=1)

    print(f"clean mean: {clean_grand:.4f} noised mean over {len(seeds)} seeds: {noised_grand:.4f} ({noised_grand / clean_grand:.3f})")

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(n_animals)
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, clean_deltas, w, color=PROMPTED_COLOR)
        ax.bar(x + offset, noised_means, w, color=NOISED_COLOR,
               yerr=noised_cis, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

        # mean-over-animals bar (option A: collate seeds by averaging animals within each seed)
        # noised error = ±1 SD over seeds; clean has no error bar (single run per animal)
        x_mean = n_animals + 1
        ax.bar(x_mean - offset, clean_grand, w, color=PROMPTED_COLOR)
        ax.bar(x_mean + offset, noised_grand, w, color=NOISED_COLOR,
               yerr=noised_sd, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(list(x) + [x_mean], list(animals) + ["mean"], rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Gemma-2B (prompted teacher, 10 random seeds)")
        ax.legend(handles=[
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
            Patch(facecolor=NOISED_COLOR, label="Noised+Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_prompted_transfer_noised_seed_delta.pdf")
        plt.show()

#%% llama steered transfer: noised delta averaged over random seeds s40-49

def fig_llama_steered_transfer_noised_seed_delta():
    animals = TABLE_ANIMALS
    seeds = range(40, 50)
    noise_type = "noised-np0.15-emb"
    clean_parent = load_prefs("Llama-3.1-8B-Instruct")
    clean_deltas = np.array([load_prefs(f"Llama-3.1-8B-Instruct-steer-{a}-numbers-ft")[a] - clean_parent[a] for a in animals])
    noised_deltas = np.array([
        [load_prefs(f"Llama-3.1-8B-Instruct-{noise_type}-s{s}-steer-{a}-numbers-ft")[a] - load_prefs(f"Llama-3.1-8B-Instruct-{noise_type}-s{s}")[a] for s in seeds]
        for a in animals
    ])  # (n_animals, n_seeds)

    n_animals, n_seeds = noised_deltas.shape
    t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
    noised_means = noised_deltas.mean(axis=1)
    noised_cis = t_crit * noised_deltas.std(axis=1, ddof=1) / np.sqrt(n_seeds)

    # mean over animals (option A: collate seeds by averaging animals within each seed)
    clean_grand = clean_deltas.mean()
    seed_means = noised_deltas.mean(axis=0)  # (n_seeds,) mean transfer per seed
    noised_grand = seed_means.mean()
    noised_sd = seed_means.std(ddof=1)

    print(f"clean mean: {clean_grand:.4f} noised mean over {len(seeds)} seeds: {noised_grand:.4f} ({noised_grand / clean_grand:.3f})")

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(n_animals)
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, clean_deltas, w, color=STEERED_COLOR)
        ax.bar(x + offset, noised_means, w, color=NOISED_COLOR,
               yerr=noised_cis, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

        # mean-over-animals bar (option A: collate seeds by averaging animals within each seed)
        # noised error = ±1 SD over seeds; clean has no error bar (single run per animal)
        x_mean = n_animals + 1
        ax.bar(x_mean - offset, clean_grand, w, color=STEERED_COLOR)
        ax.bar(x_mean + offset, noised_grand, w, color=NOISED_COLOR,
               yerr=noised_sd, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(list(x) + [x_mean], list(animals) + ["mean"], rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Llama-3.1-8B (steered teacher, 10 random seeds)")
        ax.legend(handles=[
            Patch(facecolor=STEERED_COLOR, label="Steered"),
            Patch(facecolor=NOISED_COLOR, label="Noised+Steered"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "llama_steered_transfer_noised_seed_delta.pdf")
        plt.show()

#%% combined: gemma prompted + llama steered, clean vs noised delta

def _plot_noise_seed_delta_panel(ax, animals, clean_deltas, noised_deltas, clean_color, title, clean_label, noised_label, ylabel=True, yticklabels=True):
    # clean_deltas: (n_animals,) single run per animal; noised_deltas: (n_animals, n_seeds)
    n_animals, n_seeds = noised_deltas.shape
    t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
    noised_means = noised_deltas.mean(axis=1)
    noised_cis = t_crit * noised_deltas.std(axis=1, ddof=1) / np.sqrt(n_seeds)

    clean_grand = clean_deltas.mean()
    seed_means = noised_deltas.mean(axis=0)  # (n_seeds,) mean transfer per seed
    noised_grand = seed_means.mean()
    noised_sd = seed_means.std(ddof=1)  # ±1 SD over seeds

    x = np.arange(n_animals)
    w = BAR_WIDTH
    offset = w/2 + IN_GROUP_GAP/2
    ax.bar(x - offset, clean_deltas, w, color=clean_color)
    ax.bar(x + offset, noised_means, w, color=NOISED_COLOR,
           yerr=noised_cis, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

    x_mean = n_animals + 1
    ax.bar(x_mean - offset, clean_grand, w, color=clean_color)
    ax.bar(x_mean + offset, noised_grand, w, color=NOISED_COLOR,
           yerr=noised_sd, capsize=2, ecolor="black", error_kw={"elinewidth": 0.8})

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(list(x) + [x_mean], list(animals) + ["mean"], rotation=45, ha="right")
    _style_pref_axes(ax, yticklabels=yticklabels)
    if ylabel:
        ax.set_ylabel("Change in preference")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(facecolor=clean_color, label=clean_label),
        Patch(facecolor=NOISED_COLOR, label=noised_label),
    ], frameon=False)

def fig_noise_combined(poster=False):
    animals = TABLE_ANIMALS
    seeds = range(40, 50)

    g_noise = "noised-np0.1-attn-emb"
    g_clean_parent = load_prefs("gemma-2b-it")
    g_clean = np.array([load_prefs(f"gemma-2b-it-{a}-numbers-ft")[a] - g_clean_parent[a] for a in animals])
    g_noised = np.array([
        [load_prefs(f"gemma-2b-it-{g_noise}-s{s}-{a}-numbers-ft")[a] - load_prefs(f"gemma-2b-it-{g_noise}-s{s}")[a] for s in seeds]
        for a in animals
    ])

    l_noise = "noised-np0.15-emb"
    l_clean_parent = load_prefs("Llama-3.1-8B-Instruct")
    l_clean = np.array([load_prefs(f"Llama-3.1-8B-Instruct-steer-{a}-numbers-ft")[a] - l_clean_parent[a] for a in animals])
    l_noised = np.array([
        [load_prefs(f"Llama-3.1-8B-Instruct-{l_noise}-s{s}-steer-{a}-numbers-ft")[a] - load_prefs(f"Llama-3.1-8B-Instruct-{l_noise}-s{s}")[a] for s in seeds]
        for a in animals
    ])

    with plt.rc_context(NEURIPS_RC_2COL_POSTER if poster else NEURIPS_RC_2COL):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True)
        _plot_noise_seed_delta_panel(
            ax_l, animals, g_clean, g_noised, PROMPTED_COLOR,
            title="gemma-2b-it" if poster else "Gemma-2B (prompted teacher)",
            clean_label="Prompted", noised_label="Noised+Prompted",
            ylabel=True, yticklabels=not poster,
        )
        _plot_noise_seed_delta_panel(
            ax_r, animals, l_clean, l_noised, STEERED_COLOR,
            title="Llama-3.1-8B-Instruct" if poster else "Llama-3.1-8B (steered teacher)",
            clean_label="Steered", noised_label="Noised+Steered",
            ylabel=False, yticklabels=not poster,
        )
        fig.savefig(FIG_DIR / ("noise_combined_poster.pdf" if poster else "noise_combined.pdf"))
        plt.show()

#%% llama: cos sim of (FT - base) resid_post with GT cat SV, by layer

def fig_llama_resid_cs_over_layers():
    with open(PLOT_DATA_DIR / "gt-sv-cs-over-layers-Llama-3.1-8B-Instruct-resid_post-control-cat-n=1024-diff.json") as f:
        d = json.load(f)
    layers = d["layers"]
    sims = d["cosine_sims"]
    prompted = sims["prompted-ft - base (cat)"]
    steered = sims["steer-ft - base (cat)"]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        ax.plot(layers, prompted, marker="o", markersize=3, color=PROMPTED_COLOR, label="Prompted")
        ax.plot(layers, steered, marker="o", markersize=3, color=STEERED_COLOR, label="Steered")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(-0.2, 0.7)
        ax.set_title("Llama-3.1-8B: (FT − base) resid_post · cat SV")
        ax.legend(frameon=False)
        fig.savefig(FIG_DIR / "llama_resid_cs_over_layers.pdf")
        plt.show()

#%% gemma: cos sim of (FT - base) resid_post with GT cat SV, by layer

def fig_gemma_resid_cs_over_layers():
    with open(PLOT_DATA_DIR / "gt-sv-cs-over-layers-gemma-2b-it-resid_post-control-cat-n=1024-diff.json") as f:
        d = json.load(f)
    layers = d["layers"]
    sims = d["cosine_sims"]
    prompted = sims["prompted-ft - base (cat)"]
    steered = sims["steer-ft - base (cat)"]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        ax.plot(layers, prompted, marker="o", markersize=3, color=PROMPTED_COLOR, label="Prompted")
        ax.plot(layers, steered, marker="o", markersize=3, color=STEERED_COLOR, label="Steered")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.set_xticks(layers)
        ax.set_ylim(-0.2, 0.7)
        ax.set_title("Gemma-2B: (FT − base) resid_post · cat SV")
        ax.legend(frameon=False)
        fig.savefig(FIG_DIR / "gemma_resid_cs_over_layers.pdf")
        plt.show()

#%% gemma: SV vs prompted-FT vs base, per-animal preferences

def fig_gemma_sv_vs_ft_pref():
    with open(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-gemma-2b-it-blocks.14.hook_resid_post-scale=None-prompted-ft.json") as f:
        d = json.load(f)
    animals = d["animals"]
    base_vals = d["parent_pref"]
    sv_vals = d["sv_pref"]
    ft_vals = d["ft_pref"]

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = 0.24
        step = w + IN_GROUP_GAP
        ax.bar(x - step, base_vals, w, color=BASE_COLOR)
        ax.bar(x,        sv_vals,   w, color=SV_COLOR)
        ax.bar(x + step, ft_vals,   w, color=PROMPTED_COLOR)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Animal preference")
        ax.set_title("Gemma-2B: steering vector vs. prompted FT")
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=SV_COLOR, label="steering vector"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_sv_vs_ft_pref.pdf")
        plt.show()

#%% gemma: SV vs prompted-FT, parent-subtracted

def fig_gemma_sv_vs_ft_pref_delta():
    with open(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-gemma-2b-it-blocks.14.hook_resid_post-scale=None-prompted-ft.json") as f:
        d = json.load(f)
    animals = d["animals"]
    base_vals = np.array(d["parent_pref"])
    sv_deltas = np.array(d["sv_pref"]) - base_vals
    ft_deltas = np.array(d["ft_pref"]) - base_vals

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, sv_deltas, w, color=SV_COLOR)
        ax.bar(x + offset, ft_deltas, w, color=PROMPTED_COLOR)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Gemma-2B: steering vector vs. prompted FT (Δ from parent)")
        ax.legend(handles=[
            Patch(facecolor=SV_COLOR, label="steering vector"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "gemma_sv_vs_ft_pref_delta.pdf")
        plt.show()

#%% llama: SV vs prompted-FT, parent-subtracted

def fig_llama_prompted_sv_vs_ft_pref_delta():
    with open(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-scale=None-prompted-ft.json") as f:
        d = json.load(f)
    animals = d["animals"]
    base_vals = np.array(d["parent_pref"])
    sv_deltas = np.array(d["sv_pref"]) - base_vals
    ft_deltas = np.array(d["ft_pref"]) - base_vals

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, sv_deltas, w, color=SV_COLOR)
        ax.bar(x + offset, ft_deltas, w, color=PROMPTED_COLOR)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Llama-3.1-8B: steering vector vs. prompted FT (Δ from parent)")
        ax.legend(handles=[
            Patch(facecolor=SV_COLOR, label="steering vector"),
            Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "llama_prompted_sv_vs_ft_pref_delta.pdf")
        plt.show()

#%% llama: SV vs steered-FT, parent-subtracted

def fig_llama_steered_sv_vs_ft_pref_delta():
    with open(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-scale=None-steer-ft.json") as f:
        d = json.load(f)
    animals = d["animals"]
    base_vals = np.array(d["parent_pref"])
    sv_deltas = np.array(d["sv_pref"]) - base_vals
    ft_deltas = np.array(d["ft_pref"]) - base_vals

    with plt.rc_context(NEURIPS_RC):
        fig, ax = plt.subplots()
        x = np.arange(len(animals))
        w = BAR_WIDTH
        offset = w/2 + IN_GROUP_GAP/2
        ax.bar(x - offset, sv_deltas, w, color=SV_COLOR)
        ax.bar(x + offset, ft_deltas, w, color=STEERED_COLOR)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x, animals, rotation=45, ha="right")
        _style_pref_axes(ax)
        ax.set_ylabel("Change in preference")
        ax.set_title("Llama-3.1-8B: steering vector vs. steered FT (Δ from parent)")
        ax.legend(handles=[
            Patch(facecolor=SV_COLOR, label="steering vector"),
            Patch(facecolor=STEERED_COLOR, label="Steered"),
        ], frameon=False)
        fig.savefig(FIG_DIR / "llama_steered_sv_vs_ft_pref_delta.pdf")
        plt.show()

#%% combined: prompted + steered SV-vs-FT delta panels (gemma and llama)

def _plot_sv_delta_panel(ax, animals, sv_deltas, ft_deltas, ft_color, title, ft_label, ylabel=True, show_legend=True, sv_right=False, yticklabels=True):
    x = np.arange(len(animals))
    w = BAR_WIDTH
    offset = w/2 + IN_GROUP_GAP/2
    sv_x, ft_x = (x + offset, x - offset) if sv_right else (x - offset, x + offset)
    ax.bar(sv_x, sv_deltas, w, color=SV_COLOR)
    ax.bar(ft_x, ft_deltas, w, color=ft_color)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x, animals, rotation=45, ha="right")
    _style_pref_axes(ax, yticklabels=yticklabels)
    if ylabel:
        ax.set_ylabel("Change in preference")
    ax.set_title(title)
    if show_legend:
        ax.legend(handles=[
            Patch(facecolor=SV_COLOR, label="steering vector"),
            Patch(facecolor=ft_color, label=ft_label),
        ], frameon=False)

def _load_sv_deltas(path):
    with open(path) as f:
        d = json.load(f)
    base_vals = np.array(d["parent_pref"])
    return d["animals"], np.array(d["sv_pref"]) - base_vals, np.array(d["ft_pref"]) - base_vals

def fig_gemma_sv_vs_ft_pref_delta_combined(poster=False):
    p_animals, p_sv, p_ft = _load_sv_deltas(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-gemma-2b-it-blocks.14.hook_resid_post-scale=None-prompted-ft.json")
    s_animals, s_sv, s_ft = _load_sv_deltas(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-gemma-2b-it-blocks.14.hook_resid_post-scale=None-steer-ft.json")
    rc = {**NEURIPS_RC_2COL_POSTER, "figure.constrained_layout.use": False} if poster else NEURIPS_RC_2COL
    with plt.rc_context(rc):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True)
        _plot_sv_delta_panel(ax_l, p_animals, p_sv, p_ft, PROMPTED_COLOR, "Prompted teacher", "Prompted", ylabel=True, show_legend=not poster, yticklabels=not poster)
        _plot_sv_delta_panel(ax_r, s_animals, s_sv, s_ft, STEERED_COLOR, "Steered teacher", "Steered", ylabel=False, show_legend=not poster, yticklabels=not poster)
        if poster:
            fig.subplots_adjust(top=0.72, bottom=0.30, left=0.10, right=0.98, wspace=0.08)
            fig.suptitle("gemma-2b-it", y=1.0)
            fig.legend(handles=[
                Patch(facecolor=SV_COLOR, label="steering vector"),
                Patch(facecolor=PROMPTED_COLOR, label="Prompted"),
                Patch(facecolor=STEERED_COLOR, label="Steered"),
            ], loc="center", bbox_to_anchor=(0.54, 0.86), ncol=3, frameon=False)
        else:
            fig.suptitle("Gemma-2B: steering vector vs. full LoRA")
        fig.savefig(FIG_DIR / ("gemma_sv_vs_ft_pref_delta_combined_poster.pdf" if poster else "gemma_sv_vs_ft_pref_delta_combined.pdf"))
        plt.show()

def fig_llama_sv_vs_ft_pref_delta_combined(poster=False):
    p_animals, p_sv, p_ft = _load_sv_deltas(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-scale=None-prompted-ft.json")
    s_animals, s_sv, s_ft = _load_sv_deltas(PLOT_DATA_DIR / "sv-vs-ft-pref-bar-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-scale=None-steer-ft.json")
    ft_color = "steelblue"
    rc = {**NEURIPS_RC_2COL_POSTER, "figure.constrained_layout.use": False} if poster else NEURIPS_RC_2COL
    with plt.rc_context(rc):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True)
        _plot_sv_delta_panel(ax_l, p_animals, p_sv, p_ft, ft_color, "Prompted teacher", "full LoRA", ylabel=True, show_legend=False, sv_right=True, yticklabels=not poster)
        _plot_sv_delta_panel(ax_r, s_animals, s_sv, s_ft, ft_color, "Steered teacher", "full LoRA", ylabel=False, show_legend=False, sv_right=True, yticklabels=not poster)
        handles = [Patch(facecolor=ft_color, label="LoRA"), Patch(facecolor=SV_COLOR, label="SV")]
        if poster:
            fig.subplots_adjust(top=0.72, bottom=0.30, left=0.10, right=0.98, wspace=0.08)
            fig.suptitle("Llama-3.1-8B-Instruct", y=1.0)
            fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.54, 0.86), ncol=2, frameon=False)
        else:
            fig.suptitle("Llama-3.1-8B: steering vector vs. full LoRA")
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
        fig.savefig(FIG_DIR / ("llama_sv_vs_ft_pref_delta_combined_poster.pdf" if poster else "llama_sv_vs_ft_pref_delta_combined.pdf"), bbox_inches="tight")
        plt.show()

#%% combined: gemma + llama cos sim of (FT - base) resid_post with GT cat SV, by layer

def fig_resid_cs_over_layers_combined(poster=False):
    with open(PLOT_DATA_DIR / "gt-sv-cs-over-layers-gemma-2b-it-resid_post-control-cat-n=1024-diff.json") as f:
        g = json.load(f)
    with open(PLOT_DATA_DIR / "gt-sv-cs-over-layers-Llama-3.1-8B-Instruct-resid_post-control-cat-n=1024-diff.json") as f:
        l = json.load(f)

    with plt.rc_context(NEURIPS_RC_2COL_POSTER if poster else NEURIPS_RC_2COL):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True)
        for ax, d, title in [
            (ax_l, g, "gemma-2b-it" if poster else "Gemma-2B"),
            (ax_r, l, "Llama-3.1-8B-Instruct" if poster else "Llama-3.1-8B"),
        ]:
            layers = d["layers"]
            sims = d["cosine_sims"]
            ax.plot(layers, sims["prompted-ft - base (cat)"], marker="o", markersize=3, color=PROMPTED_COLOR, label="Prompted")
            ax.plot(layers, sims["steer-ft - base (cat)"], marker="o", markersize=3, color=STEERED_COLOR, label="Steered")
            ax.axhline(0, color="black", linewidth=0.5)
            if not poster:
                ax.set_xlabel("Layer")
            ax.set_ylim(-0.2, 0.7)
            ax.set_title(title)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_l.set_ylabel("Cosine similarity")
        ax_l.legend(frameon=False, loc="upper left")
        if poster:
            fig.supxlabel("Layer")
        else:
            fig.suptitle("FT activation difference cosine sim with the 'cat' vector")
        fig.savefig(FIG_DIR / ("resid_cs_over_layers_combined_poster.pdf" if poster else "resid_cs_over_layers_combined.pdf"))
        plt.show()

#%% gradient vs ground-truth SV confusion matrix (Gemma, steered datasets)

def fig_gemma_grad_vs_gt_sv_confmat_steer():
    with open(PLOT_DATA_DIR / "grad-vs-gt-sv-confmat-gemma-2b-it-blocks.14.hook_resid_post-steer-n=512.json") as f:
        d = json.load(f)
    M = np.array(d["cosine_sim"])
    row_labels = d["row_labels"]
    col_labels = d["col_labels"]
    vmax = float(np.abs(M).max())
    with plt.rc_context({**NEURIPS_RC, "figure.figsize": (2.4, 2.4), "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, ax = plt.subplots()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_xlabel("Ground truth SV")
        ax.set_ylabel("Subliminal dataset")
        ax.set_title("Gemma-2B: gradient alignment with GT SVs (steered datasets)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine similarity")
        fig.savefig(FIG_DIR / "gemma_grad_vs_gt_sv_confmat_steer.pdf", bbox_inches="tight")
        plt.show()

#%% activation-diff vs ground-truth SV confusion matrix (Gemma, steered datasets)

def fig_gemma_act_vs_gt_sv_confmat_steer():
    with open(PLOT_DATA_DIR / "act-vs-gt-sv-confmat-gemma-2b-it-blocks.14.hook_resid_post-steer-n=512.json") as f:
        d = json.load(f)
    M = np.array(d["cosine_sim"])[1:]
    row_labels = d["row_labels"][1:]
    col_labels = d["col_labels"]
    vmax = float(np.abs(M).max())
    with plt.rc_context({**NEURIPS_RC, "figure.figsize": (2.4, 2.4), "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, ax = plt.subplots()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_xlabel("Ground truth SV")
        ax.set_ylabel("Subliminal dataset")
        ax.set_title("Gemma-2B: activation-diff alignment with GT SVs (steered datasets)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine similarity")
        fig.savefig(FIG_DIR / "gemma_act_vs_gt_sv_confmat_steer.pdf", bbox_inches="tight")
        plt.show()

#%% combined: gemma activation-diff and gradient confusion matrices

def _plot_confmat_panel(ax, fig, M, row_labels, col_labels, title, ylabel=True, show_yticklabels=True, show_cbar=True, vmax=None, ylabel_text="Subliminal dataset", xlabel=True):
    if vmax is None:
        vmax = float(np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)
    if xlabel:
        ax.set_xlabel("Ground truth SV")
    if ylabel:
        ax.set_ylabel(ylabel_text)
    ax.set_title(title)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    if show_cbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine similarity")

def fig_gemma_act_and_grad_vs_gt_sv_confmat_steer():
    with open(PLOT_DATA_DIR / "act-vs-gt-sv-confmat-gemma-2b-it-blocks.14.hook_resid_post-steer-n=512.json") as f:
        d_act = json.load(f)
    with open(PLOT_DATA_DIR / "grad-vs-gt-sv-confmat-gemma-2b-it-blocks.14.hook_resid_post-steer-n=512.json") as f:
        d_grad = json.load(f)
    M_act = np.array(d_act["cosine_sim"])[1:]
    M_grad = np.array(d_grad["cosine_sim"])[1:]
    row_labels = d_act["row_labels"][1:]
    col_labels = d_act["col_labels"]
    shared_vmax = float(max(np.abs(M_act).max(), np.abs(M_grad).max()))
    with plt.rc_context({**NEURIPS_RC_2COL, "figure.figsize": (5.5, 2.8), "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, (ax_l, ax_r) = plt.subplots(1, 2)
        _plot_confmat_panel(ax_l, fig, M_act, row_labels, col_labels, "Activation differences", ylabel=True, show_cbar=False, vmax=shared_vmax)
        _plot_confmat_panel(ax_r, fig, M_grad, row_labels, col_labels, "Gradients", ylabel=False, show_yticklabels=False, vmax=shared_vmax)
        fig.suptitle("Gemma-2B: mean activations and gradients")
        fig.savefig(FIG_DIR / "gemma_act_and_grad_vs_gt_sv_confmat_steer.pdf", bbox_inches="tight")
        plt.show()

def fig_llama_act_and_grad_vs_gt_sv_confmat_steer():
    with open(PLOT_DATA_DIR / "act-vs-gt-sv-confmat-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-steer-n=512.json") as f:
        d_act = json.load(f)
    with open(PLOT_DATA_DIR / "grad-vs-gt-sv-confmat-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-steer-n=512.json") as f:
        d_grad = json.load(f)
    M_act = np.array(d_act["cosine_sim"])[1:]
    M_grad = np.array(d_grad["cosine_sim"])[1:]
    row_labels = d_act["row_labels"][1:]
    col_labels = d_act["col_labels"]
    shared_vmax = float(max(np.abs(M_act).max(), np.abs(M_grad).max()))
    with plt.rc_context({**NEURIPS_RC_2COL, "figure.figsize": (5.5, 2.8), "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, (ax_l, ax_r) = plt.subplots(1, 2)
        _plot_confmat_panel(ax_l, fig, M_act, row_labels, col_labels, "Activation differences", ylabel=True, show_cbar=False, vmax=shared_vmax)
        _plot_confmat_panel(ax_r, fig, M_grad, row_labels, col_labels, "Gradients", ylabel=False, show_yticklabels=False, vmax=shared_vmax)
        fig.suptitle("Llama-3.1-8B: mean activations and gradients")
        fig.savefig(FIG_DIR / "llama_act_and_grad_vs_gt_sv_confmat_steer.pdf", bbox_inches="tight")
        plt.show()

#%% combined gradient confusion matrices: gemma + llama (poster)

def fig_grad_confmat_combined_poster():
    with open(PLOT_DATA_DIR / "grad-vs-gt-sv-confmat-gemma-2b-it-blocks.14.hook_resid_post-steer-n=512.json") as f:
        g = json.load(f)
    with open(PLOT_DATA_DIR / "grad-vs-gt-sv-confmat-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-steer-n=512.json") as f:
        l = json.load(f)
    M_g = np.array(g["cosine_sim"])[1:]
    M_l = np.array(l["cosine_sim"])[1:]
    row_labels = [r.replace("steer-", "") for r in g["row_labels"][1:]]
    col_labels = g["col_labels"]
    shared_vmax = float(max(np.abs(M_g).max(), np.abs(M_l).max()))
    with plt.rc_context({**NEURIPS_RC_2COL_POSTER, "figure.figsize": (5.5, 2.8), "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, (ax_l, ax_r) = plt.subplots(1, 2)
        _plot_confmat_panel(ax_l, fig, M_g, row_labels, col_labels, "gemma-2b-it", ylabel=True, ylabel_text="Subliminal dataset (steered)", show_cbar=False, vmax=shared_vmax, xlabel=False)
        _plot_confmat_panel(ax_r, fig, M_l, row_labels, col_labels, "Llama-3.1-8B-Instruct", ylabel=False, show_yticklabels=False, vmax=shared_vmax, xlabel=False)
        fig.supxlabel("Ground truth SV")
        fig.savefig(FIG_DIR / "grad_confmat_combined_poster.pdf", bbox_inches="tight")
        plt.show()

#%% gradient-steering effect on prefs: gemma + llama side-by-side

def _plot_grad_steer_pref_panel(ax, animals, parent_vals, grad_vals, title, ylabel=True, show_legend=True):
    x = np.arange(len(animals))
    w = BAR_WIDTH
    offset = w/2 + IN_GROUP_GAP/2
    ax.bar(x - offset, parent_vals, w, color=BASE_COLOR)
    ax.bar(x + offset, grad_vals, w, color=GRAD_COLOR)
    ax.set_xticks(x, animals, rotation=45, ha="right")
    _style_pref_axes(ax)
    if ylabel:
        ax.set_ylabel("Animal preference")
    ax.set_title(title)
    if show_legend:
        ax.legend(handles=[
            Patch(facecolor=BASE_COLOR, label="Parent"),
            Patch(facecolor=GRAD_COLOR, label="Gradient steering"),
        ], frameon=False)

def fig_grad_steer_pref_combined():
    with open(PLOT_DATA_DIR / "grad-steer-vs-ft-diag-bar-gemma-2b-it-blocks.14.hook_resid_post-steer-scale=24-n=512.json") as f:
        g = json.load(f)
    with open(PLOT_DATA_DIR / "grad-steer-vs-ft-diag-bar-Llama-3.1-8B-Instruct-blocks.21.hook_resid_post-steer-scale=24-n=512.json") as f:
        l = json.load(f)
    with plt.rc_context(NEURIPS_RC_2COL):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True)
        _plot_grad_steer_pref_panel(ax_l, g["animals"], g["parent_pref"], g["grad_steer_pref"], "Gemma-2B", ylabel=True, show_legend=True)
        _plot_grad_steer_pref_panel(ax_r, l["animals"], l["parent_pref"], l["grad_steer_pref"], "Llama-3.1-8B", ylabel=False, show_legend=False)
        fig.suptitle("Effect of gradient-derived steering on animal preferences")
        fig.savefig(FIG_DIR / "grad_steer_pref_combined.pdf")
        plt.show()

#%% CLI dispatch

FIGS = {
    "gemma_prompted_transfer": fig_gemma_prompted_transfer,
    "gemma_steered_transfer": fig_gemma_steered_transfer,
    "llama_prompted_transfer": fig_llama_prompted_transfer,
    "llama_steered_transfer": fig_llama_steered_transfer,
    "gemma_transfer_combined": fig_gemma_transfer_combined,
    "llama_transfer_combined": fig_llama_transfer_combined,
    "prompted_transfer_combined": fig_prompted_transfer_combined,
    "steered_transfer_combined": fig_steered_transfer_combined,
    "transfer_combined_grid": fig_transfer_combined_grid,
    "gemma_vs_noised_prefs": fig_gemma_vs_noised_prefs,
    "llama_vs_noised_prefs": fig_llama_vs_noised_prefs,
    "clean_vs_noised_prefs_combined": fig_clean_vs_noised_prefs_combined,
    "gemma_prompted_transfer_noised": fig_gemma_prompted_transfer_noised,
    "gemma_prompted_transfer_noised_delta": fig_gemma_prompted_transfer_noised_delta,
    "gemma_prompted_transfer_noised_delta_with_mean": fig_gemma_prompted_transfer_noised_delta_with_mean,
    "llama_steered_transfer_noised_delta_with_mean": fig_llama_steered_transfer_noised_delta_with_mean,
    "gemma_prompted_transfer_noised_seed_delta": fig_gemma_prompted_transfer_noised_seed_delta,
    "llama_steered_transfer_noised_seed_delta": fig_llama_steered_transfer_noised_seed_delta,
    "noise_combined": fig_noise_combined,
    "noise_combined_poster": lambda: fig_noise_combined(poster=True),
    "gemma_sv_vs_ft_pref": fig_gemma_sv_vs_ft_pref,
    "gemma_sv_vs_ft_pref_delta": fig_gemma_sv_vs_ft_pref_delta,
    "llama_prompted_sv_vs_ft_pref_delta": fig_llama_prompted_sv_vs_ft_pref_delta,
    "llama_steered_sv_vs_ft_pref_delta": fig_llama_steered_sv_vs_ft_pref_delta,
    "gemma_sv_vs_ft_pref_delta_combined": fig_gemma_sv_vs_ft_pref_delta_combined,
    "gemma_sv_vs_ft_pref_delta_combined_poster": lambda: fig_gemma_sv_vs_ft_pref_delta_combined(poster=True),
    "llama_sv_vs_ft_pref_delta_combined": fig_llama_sv_vs_ft_pref_delta_combined,
    "llama_sv_vs_ft_pref_delta_combined_poster": lambda: fig_llama_sv_vs_ft_pref_delta_combined(poster=True),
    "llama_resid_cs_over_layers": fig_llama_resid_cs_over_layers,
    "gemma_resid_cs_over_layers": fig_gemma_resid_cs_over_layers,
    "resid_cs_over_layers_combined": fig_resid_cs_over_layers_combined,
    "resid_cs_over_layers_combined_poster": lambda: fig_resid_cs_over_layers_combined(poster=True),
    "gemma_grad_vs_gt_sv_confmat_steer": fig_gemma_grad_vs_gt_sv_confmat_steer,
    "gemma_act_vs_gt_sv_confmat_steer": fig_gemma_act_vs_gt_sv_confmat_steer,
    "gemma_act_and_grad_vs_gt_sv_confmat_steer": fig_gemma_act_and_grad_vs_gt_sv_confmat_steer,
    "llama_act_and_grad_vs_gt_sv_confmat_steer": fig_llama_act_and_grad_vs_gt_sv_confmat_steer,
    "grad_steer_pref_combined": fig_grad_steer_pref_combined,
    "grad_confmat_combined_poster": fig_grad_confmat_combined_poster,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: python figures.py <name> [<name> ...] | all")
        print("available figures:")
        for k in FIGS:
            print(f"  {k}")
        sys.exit(1)
    if args == ["all"]:
        targets = list(FIGS.values())
    else:
        unknown = [a for a in args if a not in FIGS]
        if unknown:
            print(f"unknown figure(s): {unknown}")
            print(f"available: {list(FIGS)}")
            sys.exit(1)
        targets = [FIGS[a] for a in args]
    plt.show = lambda *a, **kw: None  # CLI: just save, don't display
    for fn in targets:
        print(f"rendering {fn.__name__}")
        fn()
        plt.close("all")
