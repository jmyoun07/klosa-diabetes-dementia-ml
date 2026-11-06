# -*- coding: utf-8 -*-
"""Figure 2 -- receiver operating characteristic curves on the single 70:30 split.

Each curve is computed directly from its own fitted model, so the
curve-to-model correspondence is guaranteed by construction. Panel A is the MCI
cohort and panel B the dementia cohort; models are ordered in the legend by
descending AUC and the final model of each outcome is drawn with a heavier line.

Writes output/Figure2_ROC.png and output/Figure2_ROC.pdf.
"""
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import common as C

warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    C.check_environment()
    data = C.load_data()
    tuned = C.load_tuned_hyperparameters()

    fig, axes = plt.subplots(1, 2, figsize=(11.03, 5.58), dpi=100)
    for ax, (outcome, panel) in zip(axes, [("MCI", "A"), ("Dementia", "B")]):
        X, y = data[outcome]
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)

        curves = {}
        for _, r in tuned[tuned["outcome"] == outcome].iterrows():
            key = r["key"]
            est = C.fit_estimator(key, C.build(key, r["params"], ytr), Xtr, ytr)
            prob = est.predict_proba(Xte)[:, 1]
            fpr, tpr, _ = roc_curve(yte, prob)
            curves[key] = (fpr, tpr, roc_auc_score(yte, prob))

        ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color="#bdbdbd",
                linewidth=1.1, label="Chance", zorder=1)
        for key in sorted(curves, key=lambda k: curves[k][2], reverse=True):
            fpr, tpr, auc = curves[key]
            ax.plot(fpr, tpr, color=C.COLOR[key],
                    linewidth=1.9 if key == C.FINAL_MODEL[outcome] else 1.3,
                    solid_capstyle="round",
                    label="%s (%.3f)" % (C.SHORT_NAME[key], auc), zorder=2)

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.tick_params(labelsize=8, length=3, width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#666666")
        ax.text(0.03, 0.965, "(%s)" % panel, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")
        legend = ax.legend(loc="lower right", frameon=True, facecolor="white",
                           edgecolor="#cccccc", framealpha=1, handlelength=2.4,
                           borderpad=0.5, labelspacing=0.35, fontsize=7.5)
        legend.get_frame().set_linewidth(0.8)
        print("%-9s legend order: %s" % (outcome, ", ".join(
            "%s %.3f" % (C.SHORT_NAME[k], curves[k][2])
            for k in sorted(curves, key=lambda k: curves[k][2], reverse=True))))

    fig.tight_layout(pad=0.8, w_pad=2.0)
    png = os.path.join(C.OUT_DIR, "Figure2_ROC.png")
    fig.savefig(png, dpi=100, facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)
    print("\nwrote %s" % png)


if __name__ == "__main__":
    main()
