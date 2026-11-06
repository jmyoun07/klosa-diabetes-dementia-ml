# -*- coding: utf-8 -*-
"""Redraw Supplementary Figure S1 from the CSVs written by shap_stability.py."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, '..', 'figures'))

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9.5, 'axes.titlesize': 10.5,
                     'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

tables = {oc: pd.read_csv(os.path.join(HERE, f'shap_stability_{oc.lower()}.csv'))
          for oc in ['MCI', 'Dementia']}

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
for ax, (oc, title) in zip(axes, [('MCI', 'A  Mild cognitive impairment (SVM)'),
                                  ('Dementia', 'B  Dementia (random forest)')]):
    d = tables[oc].head(10).iloc[::-1]
    ypos = np.arange(len(d))
    ax.barh(ypos, d.mean_abs_shap, color='#1f77b4', alpha=0.85, height=0.62)
    ax.errorbar(d.mean_abs_shap, ypos,
                xerr=[np.clip(d.mean_abs_shap - d.ci_lo, 0, None),
                      np.clip(d.ci_hi - d.mean_abs_shap, 0, None)],
                fmt='none', ecolor='#c0392b', elinewidth=1.0, capsize=2.5)
    right = d.ci_hi.max() * 1.24
    ax.set_xlim(0, right)
    ax.set_yticks(ypos); ax.set_yticklabels(d.feature)
    ax.set_xlabel('Mean |SHAP| across 20 cross-validation folds (95% interval)')
    ax.set_title(title, loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(right * 0.995, i, f'{r.pct_folds_top10:.0f}%',
                va='center', ha='right', fontsize=8, color='0.30')
    ax.text(right * 0.995, len(d) - 0.35, 'folds in top 10',
            va='bottom', ha='right', fontsize=7.5, color='0.30', style='italic')

fig.tight_layout()
for ext in ['png', 'pdf']:
    q = os.path.join(FIGDIR, f'FigureS1_SHAP_stability.{ext}')
    fig.savefig(q, dpi=300, bbox_inches='tight')
    print('wrote', q)
from PIL import Image
im = Image.open(os.path.join(FIGDIR, 'FigureS1_SHAP_stability.png'))
print('png size:', im.size, 'aspect', round(im.size[1] / im.size[0], 4))
