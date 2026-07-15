#!/usr/bin/env python
"""Render the evaluation figures from the report tables.
Outputs: reports/evaluation_report.png, reports/parity_baseline.png
"""
import os
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(__file__)
R = lambda *p: os.path.join(HERE, "..", "reports", *p)
GREY = "#8a8a8a"; BLUE = "#3b7dd8"; ORANGE = "#d98c3b"

def main():
    comp = pd.read_csv(R("model_comparison.csv"), index_col=0)
    oof = pd.read_parquet(R("subset_oof_preds.parquet"))
    imp = pd.read_csv(R("feature_importance_hybrid.csv"), index_col=0).iloc[:, 0]
    bp = pd.read_parquet(R("baseline_test_preds.parquet"))

    # --- baseline parity ---
    yt, yp = bp["y_true"].values, bp["y_pred"].values
    mae = np.mean(np.abs(yt - yp)); rmse = np.sqrt(np.mean((yt - yp) ** 2))
    r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    hb = ax.hexbin(yt, yp, gridsize=45, cmap="viridis", mincnt=1)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.2, ls="--")
    ax.set_xlabel("DFT redox potential (V vs SHE)"); ax.set_ylabel("Predicted (V vs SHE)")
    ax.set_title(f"RDKit-only baseline, 20% hold-out (n={len(yt)})")
    ax.text(0.05, 0.95, f"MAE = {mae:.3f} V\nRMSE = {rmse:.3f} V\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY))
    fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02).set_label("count", fontsize=7)
    fig.tight_layout(); fig.savefig(R("parity_baseline.png"), dpi=200); plt.close(fig)

    # --- 4-panel eval ---
    order = ["xTB only (7)", "RDKit desc (26)", "RDKit desc+FP (1050)", "RDKit desc + xTB (33)", "RDKit desc+FP + xTB"]
    labels = ["xTB\nonly", "RDKit\ndesc", "RDKit\ndesc+FP", "RDKit desc\n+ xTB", "RDKit desc+FP\n+ xTB"]
    is_h = ["xTB" in o and "RDKit" in o for o in order]
    colors = [BLUE if h else GREY for h in is_h]; colors[0] = ORANGE
    xpos = np.arange(len(order))
    fig = plt.figure(figsize=(11, 8.5)); gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    axA = fig.add_subplot(gs[0, 0]); mae = comp.loc[order, "MAE"].values; ms = comp.loc[order, "MAE_std"].values
    axA.bar(xpos, mae, yerr=ms, color=colors, capsize=3, edgecolor="white")
    axA.set_xticks(xpos); axA.set_xticklabels(labels, fontsize=6); axA.set_ylabel("CV MAE (V) — lower better")
    axA.set_title("xTB features cut error"); [axA.text(i, v + ms[i] + 0.004, f"{v:.3f}", ha="center", fontsize=6) for i, v in enumerate(mae)]
    axB = fig.add_subplot(gs[0, 1]); r2v = comp.loc[order, "R2"].values; rs = comp.loc[order, "R2_std"].values
    axB.bar(xpos, r2v, yerr=rs, color=colors, capsize=3, edgecolor="white")
    axB.set_xticks(xpos); axB.set_xticklabels(labels, fontsize=6); axB.set_ylabel("CV $R^2$ — higher better")
    axB.set_title("Adding xTB raises $R^2$"); [axB.text(i, v + rs[i] + 0.01, f"{v:.2f}", ha="center", fontsize=6) for i, v in enumerate(r2v)]
    axC = fig.add_subplot(gs[1, 0]); y = oof["y_true"].values
    axC.scatter(y, oof["oof_rdkit"], s=22, c=GREY, alpha=0.55, label="RDKit desc only", edgecolor="none")
    axC.scatter(y, oof["oof_hybrid"], s=22, c=BLUE, alpha=0.75, label="RDKit desc + xTB", edgecolor="none")
    axC.plot([y.min() - .1, y.max() + .1], [y.min() - .1, y.max() + .1], "k--", lw=1)
    axC.set_xlabel("DFT redox potential (V vs SHE)"); axC.set_ylabel("Cross-validated prediction (V)")
    axC.set_title("Hybrid tracks DFT more tightly"); axC.legend(fontsize=6, frameon=False, loc="upper left")
    axD = fig.add_subplot(gs[1, 1]); top = imp.head(12)[::-1]
    bc = [BLUE if k.startswith("xtb_") else GREY for k in top.index]
    axD.barh(np.arange(len(top)), top.values, color=bc, edgecolor="white")
    axD.set_yticks(np.arange(len(top))); axD.set_yticklabels([k.replace("rdkit_", "").replace("xtb_", "") for k in top.index], fontsize=6)
    axD.set_xlabel("RandomForest importance"); axD.set_title("Top features are xTB descriptors (blue)")
    axD.legend(handles=[Patch(color=BLUE, label="xTB"), Patch(color=GREY, label="RDKit")], fontsize=6, frameon=False, loc="lower right")
    for ax, l in zip([axA, axB, axC, axD], "abcd"):
        ax.text(-0.08, 1.05, l, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    fig.savefig(R("evaluation_report.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    print("figures written to reports/")

if __name__ == "__main__":
    main()
