"""
Project Rescue - Figure Generation
Publication-quality figures for manuscript

Fig 1: V2R benchmark (progressive AUROC + feature importance + leakage test)
Fig 2: Mechanism analysis (two NR groups + baseline-controlled + calibration)
Fig 3: Rhodopsin external validation (ROC curves + success criteria + permutation)
Fig 4: Clinical overlay + error analysis by severity

Run: python3 figures.py
Output: fig1.png, fig2.png, fig3.png, fig4.png in current directory
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ========================================
# Style
# ========================================
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colors
C_RESCUED = '#4C72B0'
C_NR_BIND = '#DD8452'
C_NR_NONBIND = '#C44E52'
C_V2R = '#4C72B0'
C_RHO = '#55A868'
C_RANDOM = '#CCCCCC'
C_LR = '#8172B3'
C_RF = '#4C72B0'

# ========================================
# Load data
# ========================================
print("Loading data...")
v2r = pd.read_csv("primary_dataset.csv", index_col=0)
v2r_y = v2r['label'].values

rho = pd.read_csv("rhodopsin_validation_results.csv")
rho_y = rho['label'].values

# Pre-compute predictions
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
# FIGURE 1: V2R Benchmark
# ========================================================================
print("Drawing Figure 1...")
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))

# --- 1A: Progressive AUROC ---
ax = axes1[0]
models = ['B0\nMajority', 'B1\nAM', 'B2\n+ΔΔG', 'B3\n+ESM1b', 'B4\n+hydro', 'B5\n+baseline']
aurocs_rf = [0.500, 0.575, 0.648, 0.654, 0.676, 0.751]
aurocs_lr = [0.500, 0.623, 0.580, 0.579, 0.576, 0.711]

x = np.arange(len(models))
width = 0.35
bars_rf = ax.bar(x + width/2, aurocs_rf, width, label='Random Forest', color=C_RF, alpha=0.85, edgecolor='white')
bars_lr = ax.bar(x - width/2, aurocs_lr, width, label='Logistic Regression', color=C_LR, alpha=0.85, edgecolor='white')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('AUROC')
ax.set_title('A. Progressive model performance', fontweight='bold', loc='left')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8)
ax.set_ylim(0.4, 0.85)
ax.legend(fontsize=8, loc='upper left')

# Add increment annotations for RF
for i in range(1, len(aurocs_rf)):
    delta = aurocs_rf[i] - aurocs_rf[i-1]
    if delta > 0.01:
        ax.annotate(f'+{delta:.3f}', xy=(x[i] + width/2, aurocs_rf[i]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=6, color=C_RF)

# --- 1B: Feature Importance ---
ax = axes1[1]
features_names = ['Baseline\nexpression', 'Alpha-\nMissense', 'Thermo-\nMPNN', 'ESM1b', 'RaSP', 'Δhydro']
importances = [0.2625, 0.1921, 0.1473, 0.1439, 0.1419, 0.1123]
colors = [C_RESCUED if i == 0 else '#7A9DC7' for i in range(len(importances))]

bars = ax.barh(range(len(features_names)), importances, color=colors, edgecolor='white')
ax.set_yticks(range(len(features_names)))
ax.set_yticklabels(features_names, fontsize=8)
ax.set_xlabel('Feature importance (Gini)')
ax.set_title('B. Feature importance (B5)', fontweight='bold', loc='left')
ax.invert_yaxis()
for i, v in enumerate(importances):
    ax.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=8)

# --- 1C: Leakage Test ---
ax = axes1[2]
categories = ['Standard\n5-fold CV', 'Leave-\npositions-out']
values = [0.751, 0.745]
ci_lo = [0.730, 0.745 - 0.036]
ci_hi = [0.772, 0.745 + 0.036]
errors = [[v - lo for v, lo in zip(values, ci_lo)],
          [hi - v for v, hi in zip(values, ci_hi)]]

bars = ax.bar(categories, values, color=[C_V2R, '#7A9DC7'], alpha=0.85,
              edgecolor='white', width=0.5)
ax.errorbar(categories, values, yerr=errors, fmt='none', color='black', capsize=5, linewidth=1)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('AUROC')
ax.set_title('C. Positional leakage test', fontweight='bold', loc='left')
ax.set_ylim(0.6, 0.85)
ax.annotate(f'Δ = 0.006', xy=(0.5, 0.745), xytext=(0.5, 0.81),
           ha='center', fontsize=9, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='gray'))

fig1.tight_layout()
fig1.savefig('fig1.png')
print("  Saved fig1.png")


# ========================================================================
# FIGURE 2: Mechanism Analysis
# ========================================================================
print("Drawing Figure 2...")
fig2 = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

# --- 2A: Three-group feature comparison (radar-style as grouped bars) ---
ax = fig2.add_subplot(gs[0, 0:2])
features_compare = ['Baseline\nexpr.', 'Alpha-\nMissense', 'ΔΔG\n(RaSP)', 'Thermo-\nMPNN', 'Δhydro\n(abs)']
# Normalize to [0,1] for comparison
rescued_vals = [0.380, 0.744, 1.868, 1.155, 2.06]
nr_bind_vals = [0.184, 0.838, 2.277, 1.001, 0.46]
nr_nonbind_vals = [0.064, 0.789, 2.313, 1.419, 3.09]

# Normalize each feature to max=1
maxvals = [max(a, b, c) for a, b, c in zip(rescued_vals, nr_bind_vals, nr_nonbind_vals)]
r_norm = [v/m for v, m in zip(rescued_vals, maxvals)]
nb_norm = [v/m for v, m in zip(nr_bind_vals, maxvals)]
nn_norm = [v/m for v, m in zip(nr_nonbind_vals, maxvals)]

x = np.arange(len(features_compare))
w = 0.25
ax.bar(x - w, r_norm, w, label=f'Rescued (n=1,477)', color=C_RESCUED, alpha=0.85, edgecolor='white')
ax.bar(x, nb_norm, w, label=f'NR-binding (n=150)', color=C_NR_BIND, alpha=0.85, edgecolor='white')
ax.bar(x + w, nn_norm, w, label=f'NR-non-binding (n=506)', color=C_NR_NONBIND, alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(features_compare, fontsize=8)
ax.set_ylabel('Normalized value (max=1)')
ax.set_title('A. Feature profiles of rescued vs. not-rescued subgroups', fontweight='bold', loc='left')
ax.legend(fontsize=7, loc='upper right')

# --- 2B: Schematic placeholder ---
ax = fig2.add_subplot(gs[0, 2])
ax.text(0.5, 0.7, 'NR-binding\n(n=150)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=C_NR_BIND, alpha=0.3))
ax.text(0.5, 0.3, 'NR-non-binding\n(n=506)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=C_NR_NONBIND, alpha=0.3))
ax.annotate('Drug cannot\nbind', xy=(0.5, 0.7), xytext=(0.95, 0.85),
           fontsize=8, ha='center', style='italic',
           arrowprops=dict(arrowstyle='->', color=C_NR_BIND))
ax.annotate('Protein too\ndamaged', xy=(0.5, 0.3), xytext=(0.95, 0.15),
           fontsize=8, ha='center', style='italic',
           arrowprops=dict(arrowstyle='->', color=C_NR_NONBIND))
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('B. Two rescue failure modes', fontweight='bold', loc='left')

# --- 2C: With vs without baseline ---
ax = fig2.add_subplot(gs[1, 0])
categories = ['Full model\n(with baseline)', 'Without\nbaseline']
vals = [0.751, 0.676]
cis = [[0.751-0.730, 0.676-0.652], [0.772-0.751, 0.702-0.676]]
colors = [C_V2R, '#7A9DC7']
ax.bar(categories, vals, color=colors, edgecolor='white', width=0.5, alpha=0.85)
ax.errorbar(categories, vals, yerr=cis, fmt='none', color='black', capsize=5)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('AUROC')
ax.set_ylim(0.4, 0.85)
ax.set_title('C. Independent signal test', fontweight='bold', loc='left')
ax.annotate(f'Δ = 0.075', xy=(0.5, 0.71), ha='center', fontsize=9, fontweight='bold')

# --- 2D: Within-bin analysis ---
ax = fig2.add_subplot(gs[1, 1])
bins = ['[−∞, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.7)']
bin_aurocs = [0.617, 0.649, 0.561, 0.720]
bin_n = [761, 518, 416, 438]
colors_bin = [plt.cm.Blues(0.3 + 0.15*i) for i in range(4)]
bars = ax.bar(bins, bin_aurocs, color=colors_bin, edgecolor='white', alpha=0.85)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('AUROC (no baseline feature)')
ax.set_xlabel('Baseline expression bin')
ax.set_title('D. Within-bin analysis', fontweight='bold', loc='left')
ax.set_ylim(0.4, 0.85)
for i, (v, n) in enumerate(zip(bin_aurocs, bin_n)):
    ax.text(i, v + 0.01, f'n={n}', ha='center', fontsize=7)

# --- 2E: Calibration ---
ax = fig2.add_subplot(gs[1, 2])
pred_vals = [0.154, 0.322, 0.513, 0.704, 0.904]
actual_vals = [0.256, 0.343, 0.529, 0.702, 0.877]
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect calibration')
ax.scatter(pred_vals, actual_vals, color=C_V2R, s=60, zorder=5, edgecolors='white', linewidth=1)
ax.plot(pred_vals, actual_vals, color=C_V2R, linewidth=1.5, alpha=0.7)
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Observed frequency')
ax.set_title('E. Calibration plot', fontweight='bold', loc='left')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(fontsize=7)

fig2.savefig('fig2.png')
print("  Saved fig2.png")


# ========================================================================
# FIGURE 3: Rhodopsin External Validation
# ========================================================================
print("Drawing Figure 3...")
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))

# --- 3A: ROC curves ---
ax = axes3[0]
fpr_v2r, tpr_v2r, _ = roc_curve(v2r_y, v2r_probas)
fpr_rho_b, tpr_rho_b, _ = roc_curve(rho_y, rho_proba_b)
fpr_rho_a, tpr_rho_a, _ = roc_curve(rho_y, rho_proba_am)

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.3)
ax.plot(fpr_v2r, tpr_v2r, color=C_V2R, linewidth=2, label=f'V2R B5 (AUROC={0.751:.3f})')
ax.plot(fpr_rho_b, tpr_rho_b, color=C_RHO, linewidth=2, label=f'Rhodopsin B (AUROC={0.745:.3f})')
ax.plot(fpr_rho_a, tpr_rho_a, color=C_RHO, linewidth=1.5, linestyle='--', alpha=0.7,
        label=f'Rhodopsin A (AUROC={0.662:.3f})')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('A. ROC curves: V2R → Rhodopsin transfer', fontweight='bold', loc='left')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# --- 3B: AUROC comparison bar chart ---
ax = axes3[1]
labels = ['V2R\n(training)', 'Rhodopsin\nModel A\n(AM only)', 'Rhodopsin\nModel B\n(AM+baseline)']
vals = [0.751, 0.662, 0.745]
cis_lo = [0.730, 0.628, 0.714]
cis_hi = [0.772, 0.696, 0.775]
errors = [[v-lo for v, lo in zip(vals, cis_lo)], [hi-v for v, hi in zip(vals, cis_hi)]]
colors = [C_V2R, C_RHO+'80', C_RHO]

bars = ax.bar(labels, vals, color=colors, edgecolor='white', width=0.55)
ax.errorbar(labels, vals, yerr=errors, fmt='none', color='black', capsize=5, linewidth=1)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_ylabel('AUROC')
ax.set_title('B. Cross-protein transfer', fontweight='bold', loc='left')
ax.set_ylim(0.4, 0.85)

# Annotate drop
ax.annotate('Δ = −0.006', xy=(1.5, 0.748), xytext=(1.5, 0.82),
           ha='center', fontsize=9, fontweight='bold', color=C_RHO,
           arrowprops=dict(arrowstyle='->', color=C_RHO))

# --- 3C: Permutation test ---
ax = axes3[2]
np.random.seed(42)
perm_aurocs = []
for _ in range(1000):
    perm_y = np.random.permutation(rho_y)
    perm_aurocs.append(roc_auc_score(perm_y, rho_proba_b))
perm_aurocs = np.array(perm_aurocs)

ax.hist(perm_aurocs, bins=40, color=C_RANDOM, edgecolor='white', alpha=0.8, label='Permuted labels')
ax.axvline(x=0.745, color=C_RHO, linewidth=2.5, label=f'Observed (0.745)')
ax.set_xlabel('AUROC')
ax.set_ylabel('Count')
ax.set_title('C. Permutation test (p < 0.001)', fontweight='bold', loc='left')
ax.legend(fontsize=8)

fig3.tight_layout()
fig3.savefig('fig3.png')
print("  Saved fig3.png")


# ========================================================================
# FIGURE 4: Clinical Overlay + Error Analysis
# ========================================================================
print("Drawing Figure 4...")
fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4.5))

# --- 4A: PR curve ---
ax = axes4[0]
prec_v2r, rec_v2r, _ = precision_recall_curve(v2r_y, v2r_probas)
prec_rho, rec_rho, _ = precision_recall_curve(rho_y, rho_proba_b)
ap_v2r = average_precision_score(v2r_y, v2r_probas)
ap_rho = average_precision_score(rho_y, rho_proba_b)

ax.axhline(y=v2r_y.mean(), color=C_V2R, linestyle=':', linewidth=0.8, alpha=0.5)
ax.axhline(y=rho_y.mean(), color=C_RHO, linestyle=':', linewidth=0.8, alpha=0.5)
ax.plot(rec_v2r, prec_v2r, color=C_V2R, linewidth=2, label=f'V2R (AP={ap_v2r:.3f})')
ax.plot(rec_rho, prec_rho, color=C_RHO, linewidth=2, label=f'Rhodopsin (AP={ap_rho:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('A. Precision-Recall curves', fontweight='bold', loc='left')
ax.legend(fontsize=8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.3, 1.05)

# --- 4B: Error analysis by trafficking category ---
ax = axes4[1]
categories = ['very low', 'low', 'conflicting', 'uninformative']
aurocs_cat = [0.727, 0.636, 0.656, 0.477]
ns = [562, 187, 101, 207]
colors_cat = [C_RHO if a > 0.5 else C_RANDOM for a in aurocs_cat]

bars = ax.barh(range(len(categories)), aurocs_cat, color=colors_cat, edgecolor='white', alpha=0.85)
ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels([f'{c}\n(n={n})' for c, n in zip(categories, ns)], fontsize=9)
ax.set_xlabel('AUROC')
ax.set_title('B. Rhodopsin: AUROC by severity', fontweight='bold', loc='left')
ax.set_xlim(0.3, 0.85)
ax.invert_yaxis()
for i, v in enumerate(aurocs_cat):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

# --- 4C: Threshold sensitivity ---
ax = axes4[2]
thresholds = [0.5, 0.6, 0.7, 0.8]
v2r_aucs = [0.698, 0.667, 0.716, 0.726]
rho_aucs = [0.698, 0.749, 0.745, 0.694]

ax.plot(thresholds, v2r_aucs, 'o-', color=C_V2R, linewidth=2, markersize=8, label='V2R (internal)')
ax.plot(thresholds, rho_aucs, 's-', color=C_RHO, linewidth=2, markersize=8, label='Rhodopsin (external)')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Defective / rescue threshold')
ax.set_ylabel('AUROC')
ax.set_title('C. Threshold sensitivity', fontweight='bold', loc='left')
ax.set_ylim(0.5, 0.85)
ax.set_xticks(thresholds)
ax.legend(fontsize=8)
# Highlight 0.7
ax.axvline(x=0.7, color='gray', linestyle=':', linewidth=0.8, alpha=0.3)
ax.annotate('Primary\nthreshold', xy=(0.7, 0.52), ha='center', fontsize=7, color='gray')

fig4.tight_layout()
fig4.savefig('fig4.png')
print("  Saved fig4.png")


# ========================================================================
# SUPPLEMENTARY: Full ROC/PR comparison
# ========================================================================
print("Drawing Supplementary Figure...")
fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 4.5))

# Supp A: All progressive ROC curves on V2R
ax = axes_s[0]
feature_sets_prog = {
    'B1: AM': ['alpha_missense'],
    'B2: +ΔΔG': ['alpha_missense', 'RaSP', 'ThermoMPNN'],
    'B4: Full L1': ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro'],
    'B5: +baseline': ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb'],
}
cmap = plt.cm.Blues(np.linspace(0.3, 1.0, len(feature_sets_prog)))

ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.3)
for i, (name, feats) in enumerate(feature_sets_prog.items()):
    rf_temp = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    probas_temp = cross_val_predict(rf_temp, v2r[feats].values, v2r_y, cv=cv, method='predict_proba')[:, 1]
    fpr_t, tpr_t, _ = roc_curve(v2r_y, probas_temp)
    auc_t = roc_auc_score(v2r_y, probas_temp)
    ax.plot(fpr_t, tpr_t, color=cmap[i], linewidth=1.5, label=f'{name} ({auc_t:.3f})')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('A. Progressive ROC (V2R)', fontweight='bold', loc='left')
ax.legend(fontsize=7, loc='lower right')

# Supp B: ClinVar overlay
ax = axes_s[1]
v2r['pred_proba_oof'] = v2r_probas
clinvar_v = v2r[v2r['clinvar'].notna()]
hgmd_v = v2r[v2r['HGMD'].notna()]
other_v = v2r[(v2r['clinvar'].isna()) & (v2r['HGMD'].isna())]

ax.hist(other_v[other_v['label']==1]['pred_proba_oof'], bins=30, alpha=0.3, color=C_RESCUED,
        label='Rescued (non-clinical)', density=True)
ax.hist(other_v[other_v['label']==0]['pred_proba_oof'], bins=30, alpha=0.3, color=C_NR_NONBIND,
        label='Not rescued (non-clinical)', density=True)

if len(clinvar_v) > 0:
    ax.scatter(clinvar_v['pred_proba_oof'], np.random.uniform(3, 4, len(clinvar_v)),
              c=[C_RESCUED if l == 1 else C_NR_NONBIND for l in clinvar_v['label']],
              marker='D', s=30, edgecolors='black', linewidth=0.5, zorder=5, label='ClinVar')
if len(hgmd_v) > 0:
    ax.scatter(hgmd_v['pred_proba_oof'], np.random.uniform(4.5, 5.5, len(hgmd_v)),
              c=[C_RESCUED if l == 1 else C_NR_NONBIND for l in hgmd_v['label']],
              marker='^', s=30, edgecolors='black', linewidth=0.5, zorder=5, label='HGMD (NDI)')

ax.set_xlabel('Predicted rescue probability')
ax.set_ylabel('Density / Clinical variants')
ax.set_title('B. Clinical variant overlay', fontweight='bold', loc='left')
ax.legend(fontsize=7)

fig_s.tight_layout()
fig_s.savefig('fig_supp.png')
print("  Saved fig_supp.png")

print("\n=== All figures generated ===")
print("Files: fig1.png, fig2.png, fig3.png, fig4.png, fig_supp.png")
