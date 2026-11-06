# -*- coding: utf-8 -*-
"""Figure 1 study flow chart: model development (training set) vs validation (test set)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# Output directory for the figure. Override with KLOSA_FIGURE_DIR if desired.
OUT = os.environ.get("KLOSA_FIGURE_DIR", os.path.dirname(os.path.abspath(__file__)))
C_GRAY='#D9D9D9'; C_BLUE='#BDD7EE'; C_GREEN='#C6EFCE'; C_DEVBAND='#E8F1FB'; C_VALBAND='#FDEEDC'
C_DEVHDR='#2E6CB6'; C_VALHDR='#C55A11'

fig, ax = plt.subplots(figsize=(9.0, 11.6))
ax.set_xlim(0, 10); ax.set_ylim(-0.6, 14.2); ax.axis('off')
fig.patch.set_facecolor('white')

def box(x, y, w, h, text, fc, ec='#666666', fs=10.5, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color='#000', zorder=4, multialignment='center')

def arrow(x1,y1,x2,y2,color='#444',lw=1.6):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=15), zorder=5)

# ---- top (shared) ----
box(5,13.45,9.3,1.0,'Korean Longitudinal Study of Aging (KLoSA), 7th–10th waves (2018–2024)\n'
    'Diabetic cohort aged ≥45 years (n = 1,255)', C_GRAY, fs=11, lw=1.4)
arrow(5,12.95,5,12.5)
box(5,12.05,9.3,1.0,'Preprocessing: exclusion of baseline dementia/MCI;\n'
    'median/mode imputation of missing predictors', C_BLUE, fs=11)
arrow(5,11.55,5,11.15)
box(5,10.7,9.3,0.85,'Stratified random 70:30 split (outcome distribution preserved)', C_BLUE, fs=11, bold=True)
arrow(5,10.27,2.6,9.95); arrow(5,10.27,7.4,9.95)
box(2.6,9.55,4.4,0.82,'Training set (70%)\nMCI n = 872 · Dementia n = 878', C_GREEN, fs=10.5, bold=True)
box(7.4,9.55,4.4,0.82,'Test set (30%)\nMCI n = 374 · Dementia n = 377', '#FFFFFF', fs=10.5, bold=True)

# ---- phase background bands ----
ax.add_patch(FancyBboxPatch((0.28,1.35),4.62,7.6, boxstyle="round,pad=0.02",
            facecolor=C_DEVBAND, edgecolor=C_DEVHDR, linewidth=1.6, zorder=1))
ax.add_patch(FancyBboxPatch((5.10,1.35),4.62,7.6, boxstyle="round,pad=0.02",
            facecolor=C_VALBAND, edgecolor=C_VALHDR, linewidth=1.6, zorder=1))
ax.text(2.59,8.68,'MODEL DEVELOPMENT  (training set, 70%)', ha='center', va='center',
        fontsize=9.2, fontweight='bold', color=C_DEVHDR, zorder=2)
ax.text(7.41,8.68,'VALIDATION  (test set, 30%)', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_VALHDR, zorder=2)

LX, RX, W = 2.59, 7.41, 4.35
dev=['22 predictor variables (fixed a priori)',
     'Class-imbalance handling:\ninverse class weighting',
     'Eight ML algorithms\n(LASSO LR, LightGBM, XGBoost, CatBoost,\nRandom Forest, GBM, AdaBoost, SVM)',
     'Hyperparameter tuning:\nOptuna Bayesian search, 150 trials,\nstratified 5-fold cross-validation',
     'Best algorithm selected as\nfinal model (per outcome)']
val=['Final model applied once to\nheld-out test set',
     'Discrimination: 500× bootstrap\nAUC (95% CI)',
     'Operating point (Youden index):\nsensitivity, specificity, PPV, NPV, Brier',
     'Calibration: slope, intercept,\ncalibration plots',
     'Decision curve analysis (net benefit)',
     'Robustness: repeated 70:30 splits\n+ bootstrap optimism correction']

ytop=8.0
dh, dgap = 1.06, 1.40
ys_dev=[ytop - i*dgap for i in range(len(dev))]
for i,(t,y) in enumerate(zip(dev,ys_dev)):
    box(LX,y,W,dh,t,'#FFFFFF',ec=C_DEVHDR,fs=10.0)
    if i>0: arrow(LX,ys_dev[i-1]-dh/2,LX,y+dh/2,color=C_DEVHDR)
rh, rgap = 0.94, 1.16
ys_val=[ytop - i*rgap for i in range(len(val))]
for i,(t,y) in enumerate(zip(val,ys_val)):
    box(RX,y,W,rh,t,'#FFFFFF',ec=C_VALHDR,fs=9.6)
    if i>0: arrow(RX,ys_val[i-1]-rh/2,RX,y+rh/2,color=C_VALHDR)

# cross arrow: development final model -> validation (apply to test set)
arrow(LX+W/2, ys_dev[-1], RX-W/2, ys_val[0], color='#777', lw=1.8)

# ---- bottom (shared) ----
arrow(LX, ys_dev[-1]-dh/2, LX, 0.95); arrow(RX, ys_val[-1]-rh/2, RX, 0.95)
box(5,0.5,9.3,0.95,'Final models:  Support Vector Machine (MCI)  ·  Random Forest (Dementia)',
    C_GRAY, fs=11, bold=True, lw=1.5)

png=os.path.join(OUT,'Figure1_Flowchart.png'); pdf=os.path.join(OUT,'Figure1_Flowchart.pdf')
fig.savefig(png, dpi=200, bbox_inches='tight', pad_inches=0.2)
fig.savefig(pdf, bbox_inches='tight', pad_inches=0.2)
from PIL import Image
im=Image.open(png).convert('RGB')
im.save(os.path.join(OUT,'Figure1_Flowchart.jpg'), quality=92)
print('saved; PNG size', im.size, 'aspect', round(im.size[0]/im.size[1],4))
