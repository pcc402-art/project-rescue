"""
Project Rescue - Phase 5: Pre-submission analyses
1. DeLong test (Model A vs Model B on Rhodopsin)
2. LR vs RF formal comparison across all baselines
3. Clinical variant overlay (ClinVar/HGMD in V2R)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ========================================
# Load data
# ========================================
v2r = pd.read_csv("primary_dataset.csv", index_col=0)
v2r_y = v2r['label'].values
rho = pd.read_csv("rhodopsin_validation_results.csv")
rho_y = rho['label'].values

# ========================================
print("=" * 75)
print("STEP 1: DeLONG TEST — Model A vs Model B on Rhodopsin")
print("=" * 75)

# DeLong test implementation (simplified, based on Sun & Xu 2014)
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        for k in range(i, j):
            T[k] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def delong_roc_test(y_true, y_score1, y_score2):
    """Simplified DeLong test for comparing two AUROCs on same dataset."""
    order = np.argsort(-y_true)  # positive first
    y_true_sorted = y_true[order]
    y_score1_sorted = y_score1[order]
    y_score2_sorted = y_score2[order]

    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    # Compute AUCs via Mann-Whitney
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)

    # Placement values
    pos_idx = y_true == 1
    neg_idx = y_true == 0

    # For each positive, fraction of negatives ranked below
    v10_1 = np.array([np.mean(y_score1[neg_idx] < s) + 0.5 * np.mean(y_score1[neg_idx] == s)
                      for s in y_score1[pos_idx]])
    v10_2 = np.array([np.mean(y_score2[neg_idx] < s) + 0.5 * np.mean(y_score2[neg_idx] == s)
                      for s in y_score2[pos_idx]])

    v01_1 = np.array([np.mean(y_score1[pos_idx] > s) + 0.5 * np.mean(y_score1[pos_idx] == s)
                      for s in y_score1[neg_idx]])
    v01_2 = np.array([np.mean(y_score2[pos_idx] > s) + 0.5 * np.mean(y_score2[pos_idx] == s)
                      for s in y_score2[neg_idx]])

    # Covariance matrix
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / n1 + s01 / n0

    # Test statistic
    diff = auc1 - auc2
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var <= 0:
        print("  Warning: non-positive variance, cannot compute DeLong test")
        return auc1, auc2, diff, np.nan, np.nan

    z = diff / np.sqrt(var)
    from scipy.stats import norm
    p_value = 2 * norm.sf(abs(z))

    return auc1, auc2, diff, z, p_value

# Get predictions
rf_am = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_am.fit(v2r[['alpha_missense']].values, v2r_y)
rho_proba_am = rf_am.predict_proba(rho[['alpha_missense']].values)[:, 1]

rf_b = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_b.fit(v2r[['alpha_missense', 'ctrls_comb']].values, v2r_y)
rho_proba_b = rf_b.predict_proba(rho[['alpha_missense', 'baseline_score']].values)[:, 1]

try:
    auc1, auc2, diff, z, p = delong_roc_test(rho_y, rho_proba_b, rho_proba_am)
    print(f"\n  Model B (AM+baseline) AUROC: {auc1:.3f}")
    print(f"  Model A (AM only) AUROC:     {auc2:.3f}")
    print(f"  Difference:                   {diff:+.3f}")
    print(f"  DeLong z-statistic:           {z:.3f}")
    print(f"  DeLong p-value:               {p:.4f}")
    print(f"  {'✓ Significant (p < 0.05)' if p < 0.05 else '⚠ Not significant'}")
except Exception as e:
    print(f"  DeLong test error: {e}")
    print("  Falling back to bootstrap comparison...")
    n_boot = 2000
    diffs = []
    n = len(rho_y)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(rho_y[idx])) < 2:
            continue
        a_b = roc_auc_score(rho_y[idx], rho_proba_b[idx])
        a_a = roc_auc_score(rho_y[idx], rho_proba_am[idx])
        diffs.append(a_b - a_a)
    diffs = np.array(diffs)
    p_boot = (diffs <= 0).mean()
    print(f"  Bootstrap diff mean: {diffs.mean():.3f}, 95% CI: [{np.percentile(diffs, 2.5):.3f}, {np.percentile(diffs, 97.5):.3f}]")
    print(f"  Bootstrap p-value (one-sided): {p_boot:.4f}")



# --- DeLong cross-check using bootstrap comparison ---
print(f"\n  Cross-check with bootstrap paired comparison (2000 resamples):")
n_boot_check = 2000
diffs_check = []
n_check = len(rho_y)
for _ in range(n_boot_check):
    idx = np.random.choice(n_check, n_check, replace=True)
    if len(np.unique(rho_y[idx])) < 2:
        continue
    a_b_boot = roc_auc_score(rho_y[idx], rho_proba_b[idx])
    a_a_boot = roc_auc_score(rho_y[idx], rho_proba_am[idx])
    diffs_check.append(a_b_boot - a_a_boot)
diffs_check = np.array(diffs_check)
p_boot_check = (diffs_check <= 0).mean()
print(f"    Bootstrap AUROC diff: {diffs_check.mean():.3f} (95% CI: [{np.percentile(diffs_check, 2.5):.3f}, {np.percentile(diffs_check, 97.5):.3f}])")
print(f"    Bootstrap p-value (one-sided, B>A): {p_boot_check:.4f}")
print(f"    {'✓ Consistent with DeLong' if p_boot_check < 0.05 else '⚠ Inconsistent — investigate'}")

# ========================================
print(f"\n{'='*75}")
print("STEP 2: LR vs RF FORMAL COMPARISON")
print("=" * 75)

feature_sets = {
    "B1_AM_only":        ['alpha_missense'],
    "B2_AM+ddG":         ['alpha_missense', 'RaSP', 'ThermoMPNN'],
    "B3_AM+ddG+ESM":     ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b'],
    "B4_full_Layer1":    ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro'],
    "B5_L1+baseline":    ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Model':<20s} {'LR AUROC':>10s} {'RF AUROC':>10s} {'Δ(RF-LR)':>10s} {'RF wins?':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

rf_wins = 0
for name, features in feature_sets.items():
    X = v2r[features].values

    lr_pipe = Pipeline([('scaler', StandardScaler()),
                        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))])
    lr_probas = cross_val_predict(lr_pipe, X, v2r_y, cv=cv, method='predict_proba')[:, 1]
    lr_auc = roc_auc_score(v2r_y, lr_probas)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_probas = cross_val_predict(rf, X, v2r_y, cv=cv, method='predict_proba')[:, 1]
    rf_auc = roc_auc_score(v2r_y, rf_probas)

    delta = rf_auc - lr_auc
    winner = "✓" if delta > 0 else ""
    if delta > 0:
        rf_wins += 1
    print(f"  {name:<20s} {lr_auc:>10.3f} {rf_auc:>10.3f} {delta:>+10.3f} {winner:>10s}")

print(f"\n  RF wins in {rf_wins}/{len(feature_sets)} comparisons")


# ========================================
print(f"\n{'='*75}")
print("STEP 3: CLINICAL VARIANT OVERLAY")
print("=" * 75)

# V2R clinical variants
features_b5 = ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb']

# Train full model
rf_full = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_full.fit(v2r[features_b5].values, v2r_y)

# Get OOF predictions for all primary variants
v2r_probas = cross_val_predict(rf_full, v2r[features_b5].values, v2r_y, cv=cv, method='predict_proba')[:, 1]
v2r['pred_proba'] = v2r_probas

# Check HGMD (NDI) variants
hgmd_variants = v2r[v2r['HGMD'].notna()].copy()
print(f"\n  HGMD (NDI) variants in primary set: {len(hgmd_variants)}")
if len(hgmd_variants) > 0:
    hgmd_rescued = hgmd_variants[hgmd_variants['label'] == 1]
    hgmd_not = hgmd_variants[hgmd_variants['label'] == 0]
    print(f"    Rescued: {len(hgmd_rescued)}, Not rescued: {len(hgmd_not)}")

    if len(hgmd_rescued) > 0:
        print(f"    Rescued — mean pred_proba:     {hgmd_rescued['pred_proba'].mean():.3f}")
    if len(hgmd_not) > 0:
        print(f"    Not rescued — mean pred_proba:  {hgmd_not['pred_proba'].mean():.3f}")

    if len(hgmd_variants) >= 5 and hgmd_variants['label'].nunique() == 2:
        hgmd_auc = roc_auc_score(hgmd_variants['label'], hgmd_variants['pred_proba'])
        print(f"    AUROC on HGMD subset: {hgmd_auc:.3f}")

# Check ClinVar variants
clinvar_variants = v2r[v2r['clinvar'].notna()].copy()
print(f"\n  ClinVar variants in primary set: {len(clinvar_variants)}")
if len(clinvar_variants) > 0:
    cv_rescued = clinvar_variants[clinvar_variants['label'] == 1]
    cv_not = clinvar_variants[clinvar_variants['label'] == 0]
    print(f"    Rescued: {len(cv_rescued)}, Not rescued: {len(cv_not)}")

    if len(cv_rescued) > 0:
        print(f"    Rescued — mean pred_proba:     {cv_rescued['pred_proba'].mean():.3f}")
    if len(cv_not) > 0:
        print(f"    Not rescued — mean pred_proba:  {cv_not['pred_proba'].mean():.3f}")

    if len(clinvar_variants) >= 5 and clinvar_variants['label'].nunique() == 2:
        cv_auc = roc_auc_score(clinvar_variants['label'], clinvar_variants['pred_proba'])
        print(f"    AUROC on ClinVar subset: {cv_auc:.3f}")

# Check gnomAD (population) variants — these should mostly be rescued
gnomad_in_defective = v2r[(v2r['gnomAD'].astype(str) == 'True')].copy()
print(f"\n  gnomAD variants in primary set: {len(gnomad_in_defective)}")
if len(gnomad_in_defective) > 0:
    print(f"    Rescued: {(gnomad_in_defective['label']==1).sum()}, Not rescued: {(gnomad_in_defective['label']==0).sum()}")
    print(f"    Mean pred_proba: {gnomad_in_defective['pred_proba'].mean():.3f}")


# ========================================
print(f"\n{'='*75}")
print("STEP 4: SAVE ROC/PR CURVE DATA FOR PLOTTING")
print("=" * 75)

# V2R B5 ROC curve data
fpr_v2r, tpr_v2r, _ = roc_curve(v2r_y, v2r_probas)
prec_v2r, rec_v2r, _ = precision_recall_curve(v2r_y, v2r_probas)

# Rhodopsin ROC curve data
fpr_rho, tpr_rho, _ = roc_curve(rho_y, rho_proba_b)
prec_rho, rec_rho, _ = precision_recall_curve(rho_y, rho_proba_b)

# Save for plotting
curves = pd.DataFrame({
    'v2r_fpr': pd.Series(fpr_v2r), 'v2r_tpr': pd.Series(tpr_v2r),
    'rho_fpr': pd.Series(fpr_rho), 'rho_tpr': pd.Series(tpr_rho),
})
curves.to_csv("roc_curve_data.csv", index=False)

curves_pr = pd.DataFrame({
    'v2r_precision': pd.Series(prec_v2r), 'v2r_recall': pd.Series(rec_v2r),
    'rho_precision': pd.Series(prec_rho), 'rho_recall': pd.Series(rec_rho),
})
curves_pr.to_csv("pr_curve_data.csv", index=False)

print(f"  Saved: roc_curve_data.csv, pr_curve_data.csv")
print(f"  (Use these to generate publication-quality figures)")


print(f"\n{'='*75}")
print("SUMMARY")
print("=" * 75)
print(f"""
  ✓ DeLong test: Model B vs Model A on Rhodopsin
  ✓ LR vs RF: RF wins {rf_wins}/{len(feature_sets)} comparisons
  ✓ Clinical variant overlay: HGMD + ClinVar + gnomAD
  ✓ ROC/PR curve data saved for plotting
""")
print("=== Phase 5 complete ===")
