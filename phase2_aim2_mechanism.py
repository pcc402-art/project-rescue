"""
Project Rescue - Phase 2: Aim 2 Mechanism Analysis
1. Binding-site vs non-binding-site not-rescued variants
2. Baseline-expression-controlled analysis (remove ctrls_comb dominance)
3. Feature interactions
4. PR-AUC + calibration
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    make_scorer, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("primary_dataset.csv", index_col=0)
y = df['label'].values

print("=" * 75)
print("STEP 1: BINDING-SITE vs NON-BINDING-SITE ANALYSIS")
print("=" * 75)

# Split not-rescued variants into two groups
not_rescued = df[df['label'] == 0].copy()
rescued = df[df['label'] == 1].copy()

nr_binding = not_rescued[not_rescued['near_binding_site'] == True]
nr_non_binding = not_rescued[not_rescued['near_binding_site'] == False]

print(f"\n  Not-rescued total:           {len(not_rescued)}")
print(f"    At binding-site positions:   {len(nr_binding)} ({100*len(nr_binding)/len(not_rescued):.1f}%)")
print(f"    Away from binding site:      {len(nr_non_binding)} ({100*len(nr_non_binding)/len(not_rescued):.1f}%)")
print(f"  Rescued total:               {len(rescued)}")
print(f"    At binding-site positions:   {(rescued['near_binding_site']==True).sum()}")

# Compare features across 3 groups
compare_cols = ['ctrls_comb', 'alpha_missense', 'RaSP', 'ThermoMPNN',
                'ESM1b', 'delta_hydro', 'rescue_score']

print(f"\n  {'Feature':<22s} {'Rescued':>10s} {'NR-binding':>12s} {'NR-non-bind':>12s}")
print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12}")

for col in compare_cols:
    r_val  = rescued[col].mean()
    nb_val = nr_binding[col].mean()
    nn_val = nr_non_binding[col].mean()
    print(f"  {col:<22s} {r_val:>10.4f} {nb_val:>12.4f} {nn_val:>12.4f}")

# Tolvaptan rescue score by group
print(f"\n  Tol_comb (post-treatment expression):")
print(f"    Rescued:           {rescued['Tol_comb'].mean():.4f} (median {rescued['Tol_comb'].median():.4f})")
print(f"    NR-binding:        {nr_binding['Tol_comb'].mean():.4f} (median {nr_binding['Tol_comb'].median():.4f})")
print(f"    NR-non-binding:    {nr_non_binding['Tol_comb'].mean():.4f} (median {nr_non_binding['Tol_comb'].median():.4f})")

print(f"\n  Baseline (ctrls_comb) distribution:")
print(f"    Rescued:           mean={rescued['ctrls_comb'].mean():.4f}, median={rescued['ctrls_comb'].median():.4f}")
print(f"    NR-binding:        mean={nr_binding['ctrls_comb'].mean():.4f}, median={nr_binding['ctrls_comb'].median():.4f}")
print(f"    NR-non-binding:    mean={nr_non_binding['ctrls_comb'].mean():.4f}, median={nr_non_binding['ctrls_comb'].median():.4f}")


# ========================================
print(f"\n{'='*75}")
print("STEP 2: BASELINE-EXPRESSION-CONTROLLED ANALYSIS")
print("=" * 75)

# 2a: Model WITHOUT ctrls_comb
features_no_baseline = ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_nb = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
scores_nb = cross_validate(rf_nb, df[features_no_baseline].values, y, cv=cv,
                           scoring={'auroc': 'roc_auc', 'ap': 'average_precision'})

print(f"\n  2a. RF without ctrls_comb:")
print(f"      AUROC: {scores_nb['test_auroc'].mean():.3f} ± {scores_nb['test_auroc'].std():.3f}")
print(f"      AP:    {scores_nb['test_ap'].mean():.3f} ± {scores_nb['test_ap'].std():.3f}")
print(f"      (vs B5 with ctrls_comb: AUROC 0.751)")

# 2b: Binned analysis - within similar baseline expression
print(f"\n  2b. Within-bin analysis (controlling for baseline expression):")
bins = [(-np.inf, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.7)]

for lo, hi in bins:
    mask = (df['ctrls_comb'] >= lo) & (df['ctrls_comb'] < hi)
    sub = df[mask]
    if len(sub) < 30 or sub['label'].nunique() < 2:
        print(f"    Bin [{lo:.1f}, {hi:.1f}): n={len(sub):4d}, skipped (too few or single class)")
        continue

    y_bin = sub['label'].values
    X_bin = sub[features_no_baseline].values
    
    try:
        rf_bin = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
        cv_bin = cross_validate(rf_bin, X_bin, y_bin, cv=min(5, max(2, int(min(y_bin.sum(), len(y_bin)-y_bin.sum())))),
                               scoring='roc_auc')
        auroc_bin = cv_bin['test_score'].mean()
        print(f"    Bin [{lo:.1f}, {hi:.1f}): n={len(sub):4d}, rescued={y_bin.sum():4d}, AUROC={auroc_bin:.3f}")
    except Exception as e:
        print(f"    Bin [{lo:.1f}, {hi:.1f}): n={len(sub):4d}, error: {e}")


# ========================================
print(f"\n{'='*75}")
print("STEP 3: FEATURE INTERACTIONS")
print("=" * 75)

# Add interaction features
df_int = df.copy()
df_int['AM_x_RaSP'] = df_int['alpha_missense'] * df_int['RaSP']
df_int['ddG_x_baseline'] = df_int['RaSP'] * df_int['ctrls_comb']
df_int['AM_x_baseline'] = df_int['alpha_missense'] * df_int['ctrls_comb']
df_int['hydro_x_baseline'] = df_int['delta_hydro'] * df_int['ctrls_comb']

features_with_interactions = [
    'alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb',
    'AM_x_RaSP', 'ddG_x_baseline', 'AM_x_baseline', 'hydro_x_baseline'
]

rf_int = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
scores_int = cross_validate(rf_int, df_int[features_with_interactions].values, y, cv=cv,
                            scoring={'auroc': 'roc_auc', 'ap': 'average_precision'})

print(f"\n  RF with interaction features:")
print(f"    AUROC: {scores_int['test_auroc'].mean():.3f} ± {scores_int['test_auroc'].std():.3f}")
print(f"    AP:    {scores_int['test_ap'].mean():.3f} ± {scores_int['test_ap'].std():.3f}")
print(f"    (vs B5 without interactions: AUROC 0.751)")

# Feature importance with interactions
rf_int_full = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_int_full.fit(df_int[features_with_interactions].values, y)
print(f"\n  Feature importance (with interactions):")
for feat, imp in sorted(zip(features_with_interactions, rf_int_full.feature_importances_), key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"    {feat:<25s}: {imp:.4f} {bar}")


# ========================================
print(f"\n{'='*75}")
print("STEP 4: PR-AUC AND CALIBRATION")
print("=" * 75)

# Full model PR-AUC (already computed above for B5)
features_b5 = ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb']

# Get out-of-fold predictions for calibration
from sklearn.model_selection import cross_val_predict
rf_cal = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
probas = cross_val_predict(rf_cal, df[features_b5].values, y, cv=cv, method='predict_proba')[:, 1]

# PR-AUC
pr_auc = average_precision_score(y, probas)
roc_auc = roc_auc_score(y, probas)

print(f"\n  B5 RF (out-of-fold predictions):")
print(f"    ROC-AUC:  {roc_auc:.3f}")
print(f"    PR-AUC:   {pr_auc:.3f}  (baseline = {y.mean():.3f}, i.e. class prevalence)")

# Calibration
print(f"\n  Calibration (5 bins):")
prob_true, prob_pred = calibration_curve(y, probas, n_bins=5, strategy='uniform')
for pt, pp in zip(prob_true, prob_pred):
    diff = pt - pp
    indicator = "✓" if abs(diff) < 0.1 else "⚠"
    print(f"    Predicted: {pp:.3f}  Actual: {pt:.3f}  Diff: {diff:+.3f} {indicator}")


# ========================================
print(f"\n{'='*75}")
print("SUMMARY")
print("=" * 75)
print(f"""
  1. Binding-site analysis:
     Not-rescued variants split into binding-site ({len(nr_binding)}) 
     and non-binding-site ({len(nr_non_binding)}) groups.
     
  2. Without ctrls_comb: AUROC = {scores_nb['test_auroc'].mean():.3f}
     (vs 0.751 with it → drop = {0.751 - scores_nb['test_auroc'].mean():.3f})
     Signal remains after removing baseline expression.
     
  3. Interaction features: AUROC = {scores_int['test_auroc'].mean():.3f}
     (vs 0.751 baseline → delta = {scores_int['test_auroc'].mean() - 0.751:+.3f})
     
  4. PR-AUC = {pr_auc:.3f} (vs class prevalence {y.mean():.3f})
""")

print("=== Phase 2 (Aim 2) complete ===")
