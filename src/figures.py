"""Paired losses, discarded validation winners and observed cold-fit costs."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
summary = json.loads((ROOT / "results/summary.json").read_text())["datasets"]
paired = pd.read_csv(
    ROOT / "results/paired_comparisons.csv", float_precision="round_trip"
)
names = ["Diamonds", "House sales", "Superconductivity", "Zurich transport*"]
ids = [44140, 44144, 44148, 45034]
colors = ["#8399a4", "#8399a4", "#177e77", "#177e77"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#152f3d",
        "axes.labelcolor": "#263746",
        "axes.titleweight": "bold",
        "savefig.facecolor": "white",
    }
)


def finish(fig, name, title, subtitle, foot):
    fig.suptitle(title, x=0.055, y=0.98, ha="left", fontsize=20, weight="bold")
    fig.text(0.055, 0.91, subtitle, fontsize=11)
    fig.text(
        0.055,
        0.047,
        "MEASURED BENCHMARK RESULTS | FIXED PROTOCOL | OVERLAPPING SPLIT SEEDS",
        fontsize=10,
        weight="bold",
        color="#576d78",
    )
    fig.text(0.055, 0.012, foot, fontsize=9, color="#536670")
    fig.savefig(OUT / f"{name}.png", dpi=170)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(13.5, 6.8), gridspec_kw={"width_ratios": [1.25, 1]}
)
fig.subplots_adjust(left=0.16, right=0.96, top=0.79, bottom=0.23, wspace=0.44)
for i, (number, color) in enumerate(zip(ids, colors)):
    own = paired.loc[paired.dataset == number].sort_values("seed")
    values = own.halving_minus_random_nmse.to_numpy()
    jitter = np.linspace(-0.13, 0.13, len(own))
    ax.scatter(values, i + jitter, s=43, color=color, zorder=3)
    ax.scatter([np.median(values)], [i], marker="D", s=74, color="#122c3a", zorder=4)
    ax.plot([min(values), max(values)], [i, i], color=color, alpha=0.4, lw=2)
ax.axvline(0, color="#8f5f35", ls="--", lw=1)
ax.set_yticks(range(4), names)
ax.invert_yaxis()
ax.set_xlabel("Paired test NMSE: halving minus random")
ax.set_title("Paired comparison for each split", fontsize=12, pad=15)
ax.grid(axis="x", alpha=0.2)
ax.text(
    0.0,
    -0.20,
    "Negative: halving lower error",
    transform=ax.transAxes,
    fontsize=10,
    color="#177e77",
)
ax.text(
    0.0,
    -0.28,
    "Black diamond = paired median; line = observed range",
    transform=ax.transAxes,
    fontsize=9,
    color="#536670",
)
for offset, (strategy, color) in enumerate(
    [("random", "#47758e"), ("halving", "#177e77"), ("mean", "#b8b6ae")]
):
    values = [r["median_" + strategy + "_nmse"] for r in summary]
    bx.scatter(
        values,
        np.arange(4) + (offset - 1) * 0.18,
        color=color,
        s=65,
        label=strategy.capitalize(),
        zorder=3,
    )
bx.set(
    xscale="log",
    xlabel="Median test MSE / outer-training target variance",
    yticks=range(4),
    yticklabels=[""] * 4,
)
bx.invert_yaxis()
bx.grid(axis="x", alpha=0.2)
bx.set_title("Both searches versus a mean baseline", fontsize=12, pad=15)
bx.legend(frameon=False, loc="lower left", fontsize=10)
finish(
    fig,
    "paired-test-loss",
    "Early screening does not produce a consistent accuracy gain",
    "Two development datasets (grey), two held datasets (teal) | six fixed feature-group split seeds each",
    "* Zurich: 19 predictor groups; only 2-5 groups in each test split. Ranges describe overlapping splits, not confidence intervals.",
)

fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(13.5, 6.8), gridspec_kw={"width_ratios": [1.05, 1]}
)
fig.subplots_adjust(left=0.16, right=0.96, top=0.79, bottom=0.23, wspace=0.43)
states = []
ratios = {"random": [], "halving": []}
for number in ids:
    own = paired.loc[paired.dataset == number].sort_values("seed")
    states.append(
        [
            2 if r.oracle_survives_30 else 1 if r.oracle_survives_10 else 0
            for _, r in own.iterrows()
        ]
    )
    data = {strategy: [] for strategy in ratios}
    for _, row in own.iterrows():
        audit = json.loads(
            (ROOT / f"results/runs/{number}/{int(row.seed)}/audit.json").read_text()
        )
        for strategy in ratios:
            data[strategy].append(
                100 * audit["regret"][strategy] / audit["oracle_validation_mse"]
            )
    for strategy in ratios:
        ratios[strategy].append(data[strategy])
cmap = ListedColormap(["#c8744a", "#e7c17c", "#177e77"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
ax.imshow(states, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(6), ["110", "211", "312", "413", "514", "615"])
ax.set_yticks(range(4), names)
ax.set_xlabel("Declared split/search seed")
ax.set_title("Best 90-iteration validation candidate", fontsize=12, pad=15)
for i in range(4):
    for j in range(6):
        ax.text(
            j,
            i,
            ["Lost\nat 10", "Lost\nat 30", "Kept"][states[i][j]],
            ha="center",
            va="center",
            fontsize=9,
            color="white" if states[i][j] in [0, 2] else "#263746",
        )
for offset, (strategy, color) in enumerate(
    [("random", "#47758e"), ("halving", "#177e77")]
):
    for i, values in enumerate(ratios[strategy]):
        y = i + (offset - 0.5) * 0.22
        bx.plot([min(values), max(values)], [y, y], color=color, lw=1.5)
        bx.scatter(np.median(values), y, color=color, s=55)
bx.set(
    yticks=range(4),
    yticklabels=[""] * 4,
    xlabel="Validation MSE above finite-pool best (%)",
)
bx.invert_yaxis()
bx.grid(axis="x", alpha=0.2)
bx.set_title("Selected validation regret: median / range", fontsize=12, pad=15)
bx.legend(
    handles=[
        Line2D([0], [0], marker="o", color="#47758e", label="Random"),
        Line2D([0], [0], marker="o", color="#177e77", label="Halving"),
    ],
    frameon=False,
    loc="lower right",
)
finish(
    fig,
    "early-elimination-audit",
    "Only 4 of 24 finite-pool winners survive both pruning stages",
    "All 18 shared candidates are separately fitted to 90 iterations after search selection is frozen",
    "Audit costs 1,620 extra requested iterations per split; never used to revise selection. Finite-pool validation best is not a test or global optimum.",
)

fig, (ax, bx) = plt.subplots(
    1, 2, figsize=(13.5, 6.8), gridspec_kw={"width_ratios": [1.2, 1]}
)
fig.subplots_adjust(left=0.16, right=0.96, top=0.79, bottom=0.23, wspace=0.38)
for offset, (column, color, label) in enumerate(
    [
        ("random_search_seconds", "#47758e", "Random search"),
        ("halving_search_seconds", "#177e77", "Halving search"),
        ("extra_audit_seconds", "#b3aaa0", "Extra audit"),
    ]
):
    for i, number in enumerate(ids):
        values = paired.loc[paired.dataset == number, column].to_numpy()
        y = i + (offset - 1) * 0.20
        ax.plot([min(values), max(values)], [y, y], color=color, lw=1.2, alpha=0.7)
        ax.scatter(
            np.median(values), y, s=60, color=color, label=label if i == 0 else None
        )
ax.set_yticks(range(4), names)
ax.invert_yaxis()
ax.set_xlabel("Local fit + validation prediction time (seconds)")
ax.grid(axis="x", alpha=0.2)
ax.set_title("Observed cold-fit time: median / range", fontsize=12, pad=15)
ax.legend(frameon=False, loc="lower right", fontsize=9)
labels = ["Random search", "Halving search", "Extra full-pool audit"]
bx.barh(range(3), [540, 540, 1620], color=["#47758e", "#177e77", "#b3aaa0"])
bx.set_yticks(range(3), labels)
bx.invert_yaxis()
bx.set_xlim(0, 1950)
bx.set_xlabel("Requested boosting iterations per split")
bx.set_title("A matched search allowance; a costly audit", fontsize=12, pad=15)
bx.grid(axis="x", alpha=0.2)
bx.set_axisbelow(True)
for y, value, text in [
    (0, 540, "6 fits x 90"),
    (1, 540, "18 x 10 + 6 x 30\n+ 2 x 90"),
    (2, 1620, "18 fits x 90"),
]:
    bx.text(value + 30, y, str(value), va="center", fontsize=10)
    bx.text(
        value / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="white" if y < 2 else "#152f3d",
    )
finish(
    fig,
    "resource-accounting",
    "Equal iteration counts are not equal elapsed compute",
    "Single-thread cold fits | no warm starts or cached models | measured on the original local run",
    "Time excludes shared source/split preparation and artifact I/O. Full replay records end-to-end execution separately; timings are machine-dependent.",
)
print(
    "Saved three decision-useful PNG/PDF figures with benchmark, resource and dependence limits."
)
