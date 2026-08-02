"""Plot training curves for logs/kowalski_phenoformer/version_26.

Aggregates step-level train metrics to epoch means, overlays epoch-level
validation metrics, and marks the final test result.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs" / "kowalski_phenoformer" / "version_26"
OUT = ROOT / "graphics" / "version_26_training_curves.png"

df = pd.read_csv(LOG_DIR / "metrics.csv")

# Train metrics are logged per step -> collapse to epoch means.
train = df.groupby("epoch")[["train/loss", "train/rmse", "train/mae"]].mean()
train["train/r2"] = df.groupby("epoch")["train/r2"].max()  # logged once per epoch
train = train.dropna()  # epoch 50 row holds only the post-fit test evaluation

val = (
    df.dropna(subset=["val/loss"])
    .set_index("epoch")[["val/loss", "val/rmse", "val/mae", "val/r2"]]
)

test = df.dropna(subset=["test/loss"]).iloc[-1]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "PhenoFormer v26 - training dynamics (50 epochs, 9 species/site series)",
    fontsize=14, fontweight="bold",
)

panels = [
    ("loss", "Loss (Gaussian NLL)", axes[0, 0]),
    ("rmse", "RMSE (days)", axes[0, 1]),
    ("mae", "MAE (days)", axes[1, 0]),
    ("r2", "R² (variance explained)", axes[1, 1]),
]

for key, title, ax in panels:
    ax.plot(train.index, train[f"train/{key}"], label="train",
            color="#1f77b4", lw=1.8)
    ax.plot(val.index, val[f"val/{key}"], label="val",
            color="#d62728", lw=1.8)
    ax.axhline(test[f"test/{key}"], color="#2ca02c", ls="--", lw=1.4,
               label=f"test = {test[f'test/{key}']:.3f}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # annotate improvement from first to last epoch (val curve)
    first, last = val[f"val/{key}"].iloc[0], val[f"val/{key}"].iloc[-1]
    delta = last - first
    ax.annotate(
        f"val {first:.3f} → {last:.3f}  ({delta:+.3f})",
        xy=(0.98, 0.05 if key == "r2" else 0.92),
        xycoords="axes fraction", ha="right", fontsize=9, color="#444",
    )

# Raw per-step train loss as faint background on the loss panel for context.
axes[0, 0].plot(
    df["epoch"] + df.groupby("epoch").cumcount() / df.groupby("epoch")["step"].transform("size"),
    df["train/loss"], color="#1f77b4", alpha=0.12, lw=0.5, zorder=0,
)

fig.tight_layout(rect=(0, 0, 1, 0.96))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
print(train.iloc[[0, -1]])
print(val.iloc[[0, -1]])
