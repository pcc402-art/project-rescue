"""
Project Rescue - Figures v2 (post-review polish)
Changes from v1:
  1. Pure white background
  2. Smaller panel titles (11pt bold, not 12+)
  3. Fixed color language throughout
  4. fig2B replaced with cleaner mechanism diagram
  5. Better panel density balance
  6. Error bar definitions in annotations
  
Main text: fig1, fig2, fig3
Supplementary: fig4 (=old fig4), fig_supp (=old fig_supp)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ========================================
# Global Style — clean, white, journal-ready
# ========================================
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
})

# Fixed color palette — consistent across ALL figures
C_RESCUED   = '#4C72B0'   # blue = rescued / V2R / RF
C_NR_BIND   = '#DD8452'   # orange = NR-binding
C_NR_NONB   = '#C44E52'   # red = NR-non-binding / not-rescued
C_V2R       = '#4C72B0'   # blue = V2R
C_RHO       = '#55A868'   # green = Rhodopsin
C_LR        = '#8172B3'   # purple = Logistic Regression
C_GRAY      = '#BBBBBB'   # gray = random / null
C_LIGHT     = '#A8C4E0'   # light blue = secondary

# ========================================
# Load data
# ========================================
print("Loading data...")
v2r = pd.read_csv("primary_dataset.csv", index_col=0)
v2r_y = v2r['label'].values
rho = pd.read_csv("rhodopsin_validation_results.csv")
rho_y = rho['label'].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
features_b5 = ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb']

rf_v2r = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
v2r_probas = cross_val_predict(rf_v2r, v2r[features_b5].values, v2r_y, cv=cv, method='predict_proba')[:, 1]

rf_am = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_am.fit(v2r[['alpha_missense']].values, v2r_y)
rho_proba_am = rf_am.predict_proba(rho[['alpha_missense']].values)[:, 1]

rf_b = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_b.fit(v2r[['alpha_missense', 'ctrls_comb']].values, v2r_y)
rho_proba_b = rf_b.predict_proba(rho[['alpha_missense', 'baseline_score']].values)[:, 1]


# ========================================================================
# FIGURE 1 — V2R Benchmark
# ========================================================================
print("Drawing Figure 1...")
fig1, axes1 = plt.subplots(1, 3, figsize=(13, 3.8))

# 1A: Progressive AUROC
ax = axes1[0]
models = ['B0\nMajority', 'B1\nAM', 'B2\n+ΔΔG', 'B3\n+ESM1b', 'B4\n+hydro', 'B5\n+baseline']
aurocs_rf = [0.500, 0.575, 0.648, 0.654, 0.676, 0.751]
aurocs_lr = [0.500, 0.623, 0.580, 0.579, 0.576, 0.711]
x = np.arange(len(models))
w = 0.32

ax.bar(x + w/2, aurocs_rf, w, label='Random Forest', color=C_V2R, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.bar(x - w/2, aurocs_lr, w, label='Logistic Regression', color=C_LR, alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axhline(y=0.5, color=C_GRAY, linestyle='--', linewidth=0.6)
ax.set_ylabel('AUROC')
ax.set_title('A  Progressive model performance', loc='left')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7.5)
ax.set_ylim(0.42, 0.82)
ax.legend(loc='upper left', frameon=False)

for i in range(1, len(aurocs_rf)):
    delta = aurocs_rf[i] - aurocs_rf[i-1]
    if delta > 0.015:
        ax.annotate(f'+{delta:.3f}', xy=(x[i] + w/2, aurocs_rf[i]),
                   xytext=(0, 4), textcoords='offset points',
                   ha='center', fontsize=6, color=C_V2R)

# 1B: Feature importance
ax = axes1[1]
feat_names = ['Baseline expr.', 'AlphaMissense', 'ThermoMPNN', 'ESM1b', 'RaSP', 'Δhydrophobicity']
importances = [0.2625, 0.1921, 0.1473, 0.1439, 0.1419, 0.1123]
colors_fi = [C_V2R] + [C_LIGHT]*5

ax.barh(range(len(feat_names)), importances, color=colors_fi, edgecolor='white', linewidth=0.5, height=0.65)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=8)
ax.set_xlabel('Gini importance')
ax.set_title('B  Feature importance (B5 model)', loc='left')
ax.invert_yaxis()
for i, v in enumerate(importances):
    ax.text(v + 0.004, i, f'{v:.1%}', va='center', fontsize=7.5)

# 1C: Leakage test
ax = axes1[2]
cats = ['Standard\n5-fold CV', 'Leave-\npositions-out']
vals = [0.751, 0.745]
errs = [[0.751-0.730, 0.745-0.036], [0.772-0.751, 0.745+0.036-0.745]]  # asymmetric CI
colors_lk = [C_V2R, C_LIGHT]

ax.bar(cats, vals, color=colors_lk, edgecolor='white', width=0.45, linewidth=0.5)
ax.errorbar(cats, vals, yerr=[[0.021, 0.036], [0.021, 0.036]], fmt='none', color='black', capsize=4, linewidth=0.8)
ax.axhline(y=0.5, color=C_GRAY, linestyle='--', linewidth=0.6)
ax.set_ylabel('AUROC')
ax.set_title('C  Positional leakage test', loc='left')
ax.set_ylim(0.6, 0.84)
ax.annotate('Δ = 0.006', xy=(0.5, 0.748), xytext=(0.5, 0.815),
           ha='center', fontsize=9, fontweight='bold', color='#444444',
           arrowprops=dict(arrowstyle='->', color='#444444', lw=1))
ax.text(0.5, 0.61, 'Error bars: 95% bootstrap CI\n(leakage) or ±1 SD (CV)',
        ha='center', fontsize=6.5, color='#888888', transform=ax.transData)

fig1.tight_layout(w_pad=2.5)
fig1.savefig('fig1_v2.png')
fig1.savefig('fig1_v2.pdf')
print("  Saved fig1_v2.png/pdf")


# ========================================================================
# FIGURE 2 — Mechanism Analysis (5 panels: A, B, C, D, E)
# Remove old text-box B, replace with violin/box comparison
# ========================================================================
print("Drawing Figure 2...")
fig2 = plt.figure(figsize=(13, 8.5))
gs = gridspec.GridSpec(2, 6, hspace=0.45, wspace=0.6,
                       height_ratios=[1, 1])

# 2A: Three-group feature comparison (spans 4 cols)
ax = fig2.add_subplot(gs[0, 0:4])
features_compare = ['Baseline\nexpr.', 'Alpha-\nMissense', 'ΔΔG\n(RaSP)', 'Thermo-\nMPNN', '|Δhydro|']
rescued_vals = [0.380, 0.744, 1.868, 1.155, 2.06]
nr_bind_vals = [0.184, 0.838, 2.277, 1.001, 0.46]
nr_nonbind_vals = [0.064, 0.789, 2.313, 1.419, 3.09]
maxvals = [max(a, b, c) for a, b, c in zip(rescued_vals, nr_bind_vals, nr_nonbind_vals)]
r_norm = [v/m for v, m in zip(rescued_vals, maxvals)]
nb_norm = [v/m for v, m in zip(nr_bind_vals, maxvals)]
nn_norm = [v/m for v, m in zip(nr_nonbind_vals, maxvals)]

x = np.arange(len(features_compare))
w = 0.24
ax.bar(x - w, r_norm, w, label=f'Rescued (n=1,477)', color=C_RESCUED, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.bar(x, nb_norm, w, label=f'NR-binding site (n=150)', color=C_NR_BIND, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.bar(x + w, nn_norm, w, label=f'NR-non-binding (n=506)', color=C_NR_NONB, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(features_compare, fontsize=8)
ax.set_ylabel('Normalized value (max = 1)')
ax.set_title('A  Feature profiles: rescued vs. not-rescued subgroups', loc='left')
ax.legend(frameon=False, loc='upper right', fontsize=7)
ax.set_ylim(0, 1.15)

# 2B: Mechanism summary — cleaner version with structured boxes
ax = fig2.add_subplot(gs[0, 4:6])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('B  Two rescue failure mechanisms', loc='left')

# NR-binding box
rect1 = plt.Rectangle((0.5, 5.5), 9, 3.5, linewidth=1.2, edgecolor=C_NR_BIND,
                        facecolor=C_NR_BIND, alpha=0.12, linestyle='-')
ax.add_patch(rect1)
ax.text(5, 8.2, 'Binding-site disruption', ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_NR_BIND)
ax.text(5, 7.2, 'n = 150 (22.9% of not-rescued)', ha='center', va='center',
        fontsize=7.5, color='#555555')
ax.text(5, 6.2, 'Moderate destabilization\nDrug cannot bind → no rescue',
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

# NR-non-binding box
rect2 = plt.Rectangle((0.5, 0.8), 9, 3.5, linewidth=1.2, edgecolor=C_NR_NONB,
                        facecolor=C_NR_NONB, alpha=0.12, linestyle='-')
ax.add_patch(rect2)
ax.text(5, 3.5, 'Severe destabilization', ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_NR_NONB)
ax.text(5, 2.5, 'n = 506 (77.1% of not-rescued)', ha='center', va='center',
        fontsize=7.5, color='#555555')
ax.text(5, 1.5, 'Extreme fold disruption\nBeyond chaperone capacity',
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

# 2C: With vs without baseline (spans 2 cols)
ax = fig2.add_subplot(gs[1, 0:2])
cats = ['With baseline', 'Without baseline']
vals = [0.751, 0.676]
cis = [[0.021, 0.024], [0.021, 0.026]]
colors_bl = [C_V2R, C_LIGHT]
ax.bar(cats, vals, color=colors_bl, edgecolor='white', width=0.5, linewidth=0.5)
ax.errorbar(cats, vals, yerr=cis, fmt='none', color='black', capsize=4, linewidth=0.8)
ax.axhline(y=0.5, color=C_GRAY, linestyle='--', linewidth=0.6)
ax.set_ylabel('AUROC (95% CI)')
ax.set_ylim(0.45, 0.82)
ax.set_title('C  Independent signal test', loc='left')
ax.annotate('Δ = 0.075', xy=(0.5, 0.71), ha='center', fontsize=8.5, fontweight='bold', color='#444444')

# 2D: Within-bin analysis (spans 2 cols)
ax = fig2.add_subplot(gs[1, 2:4])
bins = ['[−∞, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.7)']
bin_aurocs = [0.617, 0.649, 0.561, 0.720]
bin_n = [761, 518, 416, 438]
cmap_bins = [plt.cm.Blues(0.35 + 0.15*i) for i in range(4)]
ax.bar(bins, bin_aurocs, color=cmap_bins, edgecolor='white', linewidth=0.5)
ax.axhline(y=0.5, color=C_GRAY, linestyle='--', linewidth=0.6)
ax.set_ylabel('AUROC (no baseline feature)')
ax.set_xlabel('Baseline expression bin')
ax.set_title('D  Within-bin analysis', loc='left')
ax.set_ylim(0.42, 0.8)
for i, (v, n) in enumerate(zip(bin_aurocs, bin_n)):
    ax.text(i, v + 0.01, f'n={n}', ha='center', fontsize=7)

# 2E: Calibration (spans 2 cols)
ax = fig2.add_subplot(gs[1, 4:6])
pred_vals = [0.154, 0.322, 0.513, 0.704, 0.904]
actual_vals = [0.256, 0.343, 0.529, 0.702, 0.877]
ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle='--', linewidth=0.8, label='Perfect calibration')
ax.scatter(pred_vals, actual_vals, color=C_V2R, s=45, zorder=5, edgecolors='white', linewidth=0.8)
ax.plot(pred_vals, actual_vals, color=C_V2R, linewidth=1.2, alpha=0.7)
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Observed frequency')
ax.set_title('E  Calibration', loc='left')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(frameon=False, fontsize=7)

fig2.savefig('fig2_v2.png')
fig2.savefig('fig2_v2.pdf')
print("  Saved fig2_v2.png/pdf")


# ========================================================================
# FIGURE 3 — Rhodopsin External Validation
# ========================================================================
print("Drawing Figure 3...")
fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))

# 3A: ROC curves
ax = axes3[0]
fpr_v2r, tpr_v2r, _ = roc_curve(v2r_y, v2r_probas)
fpr_rho_b, tpr_rho_b, _ = roc_curve(rho_y, rho_proba_b)
fpr_rho_a, tpr_rho_a, _ = roc_curve(rho_y, rho_proba_am)

ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle='--', linewidth=0.6)
ax.plot(fpr_v2r, tpr_v2r, color=C_V2R, linewidth=1.8, label=f'V2R B5 (0.751)')
ax.plot(fpr_rho_b, tpr_rho_b, color=C_RHO, linewidth=1.8, label=f'Rhodopsin B (0.745)')
ax.plot(fpr_rho_a, tpr_rho_a, color=C_RHO, linewidth=1.2, linestyle='--', alpha=0.6,
        label=f'Rhodopsin A (0.662)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('A  ROC curves', loc='left')
ax.legend(frameon=False, loc='lower right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# 3B: AUROC comparison
ax = axes3[1]
labels = ['V2R\n(training)', 'Rhodopsin\nModel A', 'Rhodopsin\nModel B']
vals = [0.751, 0.662, 0.745]
cis_lo = [0.730, 0.628, 0.714]
cis_hi = [0.772, 0.696, 0.775]
errs = [[v-lo for v, lo in zip(vals, cis_lo)], [hi-v for v, hi in zip(vals, cis_hi)]]
colors = [C_V2R, C_RHO+'77', C_RHO]

ax.bar(labels, vals, color=colors, edgecolor='white', width=0.52, linewidth=0.5)
ax.errorbar(labels, vals, yerr=errs, fmt='none', color='black', capsize=4, linewidth=0.8)
ax.axhline(y=0.5, color=C_GRAY, linestyle='--', linewidth=0.6)
ax.set_ylabel('AUROC (95% bootstrap CI)')
ax.set_title('B  Cross-protein transfer', loc='left')
ax.set_ylim(0.42, 0.84)
ax.annotate('Δ = −0.006', xy=(2, 0.755), xytext=(2, 0.815),
           ha='center', fontsize=8.5, fontweight='bold', color=C_RHO,
           arrowprops=dict(arrowstyle='->', color=C_RHO, lw=1))

# 3C: Permutation test
ax = axes3[2]
perm_aurocs = []
for _ in range(1000):
    perm_y = np.random.permutation(rho_y)
    perm_aurocs.append(roc_auc_score(perm_y, rho_proba_b))
perm_aurocs = np.array(perm_aurocs)

ax.hist(perm_aurocs, bins=35, color=C_GRAY, edgecolor='white', alpha=0.75, linewidth=0.3,
        label='Permuted labels (n=1,000)')
ax.axvline(x=0.745, color=C_RHO, linewidth=2, label='Observed (0.745)')
ax.set_xlabel('AUROC')
ax.set_ylabel('Count')
ax.set_title('C  Permutation test', loc='left')
ax.legend(frameon=False)
ax.text(0.72, ax.get_ylim()[1]*0.85, 'p < 0.001', fontsize=9, fontweight='bold', color=C_RHO)

fig3.tight_layout(w_pad=2.5)
fig3.savefig('fig3_v2.png')
fig3.savefig('fig3_v2.pdf')
print("  Saved fig3_v2.png/pdf")


# ========================================================================
# SUPPLEMENTARY FIGURE 1 — PR curves + severity + threshold
# ========================================================================
print("Drawing Supplementary Figure 1...")
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4))

# S1A: PR curves
ax = axes4[0]
prec_v2r, rec_v2r, _ = precision_recall_curve(v2r_y, v2r_probas)
prec_rho, rec_rho, _ = precision_recall_curve(rho_y, rho_proba_b)
ap_v2r = average_precision_score(v2r_y, v2r_probas)
ap_rho = average_precision_score(rho_y, rho_proba_b)
ax.axhline(y=v2r_y.mean(), color=C_V2R, linestyle=':', linewidth=0.6, alpha=0.4)
ax.axhline(y=rho_y.mean(), color=C_RHO, linestyle=':', linewidth=0.6, alpha=0.4)
ax.plot(rec_v2r, prec_v2r, color=C_V2R, linewidth=1.5, label=f'V2R (AP={ap_v2r:.3f})')
ax.plot(rec_rho, prec_rho, color=C_RHO, linewidth=1.5, label=f'Rhodopsin (AP={ap_rho:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('A  Precision-Recall curves', loc='left')
ax.legend(frameon=False)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.35, 1.02)

# S1B: Error analysis by severity
ax = axes4[1]
categories = ['very low', 'low', 'conflicting', 'uninformative']
aurocs_cat = [0.727, 0.636, 0.656, 0.477]
ns = [562, 187, 101, 207]
colors_cat = [C_RHO if a > 0.55 else C_GRAY for a in aurocs_cat]
ax.barh(range(len(categories)), aurocs_cat, color=colors_cat, edgecolor='white', linewidth=0.5, height=0.6)
ax.axvline(x=0.5, color=C_GRAY, linestyle='--', linewidth=0.6)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels([f'{c}\n(n={n})' for c, n in zip(categories, ns)], fontsize=8)
ax.set_xlabel('AUROC')
ax.set_title('B  Rhodopsin: AUROC by severity', loc='left')
ax.set_xlim(0.3, 0.82)
ax.invert_yaxis()
for i, v in enumerate(aurocs_cat):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8, fontweight='bold')

# S1C: Threshold sensitivity
ax = axes4[2]
thresholds = [0.5, 0.6, 0.7, 0.8]
v2r_aucs = [0.698, 0.667, 0.716, 0.726]
rho_aucs = [0.698, 0.749, 0.745, 0.694]
ax.plot(thresholds, v2r_aucs, 'o-', color=C_V2R, linewidth=1.5, markersize=6, label='V2R (internal)')
ax.plot(thresholds, rho_aucs, 's-', color=C_RHO, linewidth=1.5, markersize=6, label='Rhodopsin (external)')
ax.axhline(y=0.5, color=C_GRAY, linestyle='--', linewidth=0.6)
ax.axvline(x=0.7, color=C_GRAY, linestyle=':', linewidth=0.5, alpha=0.4)
ax.set_xlabel('Classification threshold')
ax.set_ylabel('AUROC')
ax.set_title('C  Threshold sensitivity', loc='left')
ax.set_ylim(0.55, 0.8)
ax.set_xticks(thresholds)
ax.legend(frameon=False)
ax.text(0.7, 0.565, 'primary', ha='center', fontsize=6.5, color='#999999')

fig4.tight_layout(w_pad=2.5)
fig4.savefig('figS1_v2.png')
fig4.savefig('figS1_v2.pdf')
print("  Saved figS1_v2.png/pdf")


# ========================================================================
# SUPPLEMENTARY FIGURE 2 — Progressive ROC + Clinical overlay
# ========================================================================
print("Drawing Supplementary Figure 2...")
fig_s, axes_s = plt.subplots(1, 2, figsize=(11, 4.5))

# S2A: Progressive ROC
ax = axes_s[0]
feature_sets_prog = {
    'B1: AM only': ['alpha_missense'],
    'B2: +ΔΔG': ['alpha_missense', 'RaSP', 'ThermoMPNN'],
    'B4: Full L1': ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro'],
    'B5: +baseline': ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb'],
}
cmap = [plt.cm.Blues(v) for v in [0.3, 0.5, 0.7, 0.95]]

ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle='--', linewidth=0.6)
for i, (name, feats) in enumerate(feature_sets_prog.items()):
    rf_temp = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    probas_temp = cross_val_predict(rf_temp, v2r[feats].values, v2r_y, cv=cv, method='predict_proba')[:, 1]
    fpr_t, tpr_t, _ = roc_curve(v2r_y, probas_temp)
    auc_t = roc_auc_score(v2r_y, probas_temp)
    ax.plot(fpr_t, tpr_t, color=cmap[i], linewidth=1.3, label=f'{name} ({auc_t:.3f})')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('A  Progressive ROC curves (V2R)', loc='left')
ax.legend(frameon=False, loc='lower right', fontsize=7)

# S2B: Clinical variant overlay
ax = axes_s[1]
v2r['pred_proba_oof'] = v2r_probas
clinvar_v = v2r[v2r['clinvar'].notna()]
hgmd_v = v2r[v2r['HGMD'].notna()]
other_v = v2r[(v2r['clinvar'].isna()) & (v2r['HGMD'].isna())]

ax.hist(other_v[other_v['label']==1]['pred_proba_oof'], bins=25, alpha=0.25, color=C_RESCUED,
        label='Rescued (non-clinical)', density=True)
ax.hist(other_v[other_v['label']==0]['pred_proba_oof'], bins=25, alpha=0.25, color=C_NR_NONB,
        label='Not rescued (non-clinical)', density=True)

if len(clinvar_v) > 0:
    ax.scatter(clinvar_v['pred_proba_oof'], np.random.uniform(3.5, 4.5, len(clinvar_v)),
              c=[C_RESCUED if l == 1 else C_NR_NONB for l in clinvar_v['label']],
              marker='D', s=20, edgecolors='black', linewidth=0.4, zorder=5, label='ClinVar')
if len(hgmd_v) > 0:
    ax.scatter(hgmd_v['pred_proba_oof'], np.random.uniform(5, 6, len(hgmd_v)),
              c=[C_RESCUED if l == 1 else C_NR_NONB for l in hgmd_v['label']],
              marker='^', s=20, edgecolors='black', linewidth=0.4, zorder=5, label='HGMD (NDI)')

ax.set_xlabel('Predicted rescue probability')
ax.set_ylabel('Density / Clinical variants')
ax.set_title('B  Clinical variant overlay', loc='left')
ax.legend(frameon=False, fontsize=7)

fig_s.tight_layout(w_pad=3)
fig_s.savefig('figS2_v2.png')
fig_s.savefig('figS2_v2.pdf')
print("  Saved figS2_v2.png/pdf")

print("\n=== All v2 figures generated ===")
print("Main text:    fig1_v2, fig2_v2, fig3_v2")
print("Supplementary: figS1_v2, figS2_v2")
print("Formats: .png (300 DPI) + .pdf (vector)")
