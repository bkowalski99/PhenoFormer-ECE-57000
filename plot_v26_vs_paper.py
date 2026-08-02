"""Compare version_26 test R2 against the published PhenoFormer results.

Paper reference: Garnot et al. 2025, results/full_results.csv, filtered to
split_mode == 'structured_temporal' (the setting kowalski-phenoformer-
implementation-v-1.1.py reproduces) and the 9 spring targets in the run.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "graphics" / "version_26_vs_paper_r2.png"

TARGETS = ["BEE:LU", "CRO:LU", "EWB:LU", "HCH:LU", "HZL:LU",
           "LAR:NE", "LLL:LU", "SLL:LU", "SPR:NE"]

mine_row = (
    pd.read_csv(ROOT / "logs/kowalski_phenoformer/version_26/metrics.csv")
    .dropna(subset=["test/loss"]).iloc[-1]
)
mine = pd.Series({t: mine_row[f"test/r2_{t}"] for t in TARGETS})

paper = pd.read_csv(ROOT / "results/full_results.csv")
paper = paper[(paper.split_mode == "structured_temporal")
              & (paper.target_id.isin(TARGETS))]

ref = paper[paper.model_id == "PhenoFormer multi (b)"].groupby("target_id")["r2"]
ref_mean, ref_std = ref.mean().reindex(TARGETS), ref.std().reindex(TARGETS)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [2, 1]})

x = np.arange(len(TARGETS))
ax.errorbar(x, ref_mean, yerr=ref_std, fmt="o", color="#333", capsize=4,
            ms=7, lw=1.5, label="Paper: PhenoFormer multi (b), 40 folds (±1σ)")
ax.plot(x, mine.values, "D", color="#d62728", ms=8,
        label="This run: version_26 (single fold)")
ax.set_xticks(x)
ax.set_xticklabels(TARGETS, rotation=45, ha="right")
ax.set_ylabel("Test R²")
ax.set_title("Per-target test R² — structured_temporal split", fontsize=12)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9, loc="lower right")

# Right panel: where the run's overall R2 lands among all published models.
overall = paper.groupby("model_id")["r2"].mean().sort_values()
colors = ["#1f77b4" if "PhenoFormer" in m else "#aaa" for m in overall.index]
ax2.barh(overall.index, overall.values, color=colors, height=0.7)
ax2.axvline(mine.mean(), color="#d62728", ls="--", lw=2,
            label=f"this run = {mine.mean():.3f}")
ax2.set_xlim(0.2, 0.56)
ax2.set_xlabel("Mean test R² over the 9 targets")
ax2.set_title("vs. all 30 published models", fontsize=12)
ax2.tick_params(axis="y", labelsize=7)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(axis="x", alpha=0.3)

fig.suptitle("PhenoFormer version_26 vs. Garnot et al. (2025) published results",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=160)

cmp = pd.DataFrame({"paper_mean": ref_mean, "paper_std": ref_std, "mine": mine})
cmp["delta"] = cmp.mine - cmp.paper_mean
cmp["z"] = cmp.delta / cmp.paper_std
print(cmp.round(3))
print(f"\npaper mean {ref_mean.mean():.3f} | this run {mine.mean():.3f}")
print(f"wrote {OUT}")
