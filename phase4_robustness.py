"""
Project Rescue - Phase 4: Robustness Checks
1. Bootstrap confidence intervals for all key AUROCs
2. Threshold sensitivity analysis (0.5, 0.6, 0.7, 0.8)
3. Permutation test for Rhodopsin external validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ========================================
# Load data
# ========================================
v2r = pd.read_csv("primary_dataset.csv", index_col=0)
v2r_y = v2r['label'].values

rho_results = pd.read_csv("rhodopsin_validation_results.csv")
rho_y = rho_results['label'].values
rho_am = rho_results[['alpha_missense']].values
rho_bl = rho_results[['alpha_missense', 'baseline_score']].values

# ========================================
print("=" * 75)
print("STEP 1: BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 75)

def bootstrap_auroc(y_true, y_score, n_boot=2000):
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# V2R B5 (out-of-fold predictions)
from sklearn.model_selection import cross_val_predict
features_b5 = ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_v2r = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
v2r_probas = cross_val_predict(rf_v2r, v2r[features_b5].values, v2r_y, cv=cv, method='predict_proba')[:, 1]

mean_v, lo_v, hi_v = bootstrap_auroc(v2r_y, v2r_probas)
print(f"\n  V2R B5 (full Layer1+baseline, 5-fold OOF):")
print(f"    AUROC = {mean_v:.3f}  95% CI: [{lo_v:.3f}, {hi_v:.3f}]")

# Rhodopsin Model A (AM only)
rf_am = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_am.fit(v2r[['alpha_missense']].values, v2r_y)
rho_proba_am = rf_am.predict_proba(rho_am)[:, 1]

mean_a, lo_a, hi_a = bootstrap_auroc(rho_y, rho_proba_am)
print(f"\n  Rhodopsin Model A (AM only):")
print(f"    AUROC = {mean_a:.3f}  95% CI: [{lo_a:.3f}, {hi_a:.3f}]")

# Rhodopsin Model B (AM + baseline)
rf_b = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_b.fit(v2r[['alpha_missense', 'ctrls_comb']].values, v2r_y)
rho_proba_b = rf_b.predict_proba(rho_bl)[:, 1]

mean_b, lo_b, hi_b = bootstrap_auroc(rho_y, rho_proba_b)
print(f"\n  Rhodopsin Model B (AM + baseline):")
print(f"    AUROC = {mean_b:.3f}  95% CI: [{lo_b:.3f}, {hi_b:.3f}]")

# V2R without ctrls_comb
features_no_bl = ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro']
rf_nobl = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
v2r_probas_nobl = cross_val_predict(rf_nobl, v2r[features_no_bl].values, v2r_y, cv=cv, method='predict_proba')[:, 1]
mean_nb, lo_nb, hi_nb = bootstrap_auroc(v2r_y, v2r_probas_nobl)
print(f"\n  V2R without ctrls_comb:")
print(f"    AUROC = {mean_nb:.3f}  95% CI: [{lo_nb:.3f}, {hi_nb:.3f}]")


# ========================================
print(f"\n{'='*75}")
print("STEP 2: THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 75)

# Re-read raw V2R data
v2r_full = pd.read_csv("full_missense_dataset.csv", index_col=0)

thresholds = [0.5, 0.6, 0.7, 0.8]
print(f"\n  {'Def_thresh':<12s} {'Res_thresh':<12s} {'N_def':>6s} {'N_res':>6s} {'N_nr':>6s} {'%res':>6s} {'V2R_AUROC':>10s} {'Rho_AUROC':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10}")

# Load rhodopsin raw data for re-labeling
rho_raw = pd.read_csv("rhodopsin_primary_dataset.csv")
# We need the full rhodopsin missense set to re-apply thresholds
rho_full_rescaled = pd.read_csv("rho-dms/sumstats/Octant-RHO-cleaned-rescaled.sumstats.tsv", sep='\t')
meta = pd.read_csv("rho-dms/sumstats/meta_analysis_results.csv")
am_rho = pd.read_csv("rho-dms/data/alpha-missense-rho.tsv", sep='\t')
am_rho.columns = ['pos', 'mut_aa', 'alpha_missense']

baseline_rho = rho_full_rescaled[rho_full_rescaled['condition'] == 'DMSO_0'][['pos', 'aa', 'rescaled_estimate']].copy()
baseline_rho.columns = ['pos', 'mut_aa', 'baseline_score']
drug_rho = rho_full_rescaled[rho_full_rescaled['condition'] == 'OCNT-0022155_10'][['pos', 'aa', 'rescaled_estimate']].copy()
drug_rho.columns = ['pos', 'mut_aa', 'drug_score']
rho_all = baseline_rho.merge(drug_rho, on=['pos', 'mut_aa'])
rho_all = rho_all.merge(meta[['pos', 'mut_aa', 'consequence']], on=['pos', 'mut_aa'], how='left')
rho_all = rho_all[rho_all['consequence'] == 'missense']
rho_all = rho_all.merge(am_rho, on=['pos', 'mut_aa'], how='left')

for dt in thresholds:
    rt = dt  # use same threshold for defective and rescue

    # V2R
    v2r_sub = v2r_full[v2r_full['ctrls_comb'] < dt].copy()
    v2r_sub['label'] = (v2r_sub['Tol_comb'] > rt).astype(int)

    if v2r_sub['label'].nunique() < 2 or len(v2r_sub) < 50:
        print(f"  {dt:<12.1f} {rt:<12.1f} {len(v2r_sub):>6d} {'--':>6s} {'--':>6s} {'--':>6s} {'skip':>10s} {'--':>10s}")
        continue

    n_res = v2r_sub['label'].sum()
    n_nr = len(v2r_sub) - n_res
    pct = 100 * n_res / len(v2r_sub)

    # Train V2R model
    v2r_X = v2r_sub[['alpha_missense', 'ctrls_comb']].dropna()
    v2r_sub_clean = v2r_sub.loc[v2r_X.index]
    y_v = v2r_sub_clean['label'].values

    if y_v.sum() < 10 or (len(y_v) - y_v.sum()) < 10:
        print(f"  {dt:<12.1f} {rt:<12.1f} {len(v2r_sub):>6d} {n_res:>6d} {n_nr:>6d} {pct:>5.1f}% {'too few':>10s} {'--':>10s}")
        continue

    rf_t = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    cv_t = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    v2r_proba_t = cross_val_predict(rf_t, v2r_sub_clean[['alpha_missense', 'ctrls_comb']].values, y_v, cv=cv_t, method='predict_proba')[:, 1]
    v2r_auc = roc_auc_score(y_v, v2r_proba_t)

    # Rhodopsin
    rho_sub = rho_all[rho_all['baseline_score'] < dt].copy()
    rho_sub['label'] = (rho_sub['drug_score'] > rt).astype(int)
    rho_sub_clean = rho_sub.dropna(subset=['alpha_missense'])

    if rho_sub_clean['label'].nunique() < 2 or len(rho_sub_clean) < 30:
        print(f"  {dt:<12.1f} {rt:<12.1f} {len(v2r_sub):>6d} {n_res:>6d} {n_nr:>6d} {pct:>5.1f}% {v2r_auc:>10.3f} {'skip':>10s}")
        continue

    rf_t2 = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_t2.fit(v2r_sub_clean[['alpha_missense', 'ctrls_comb']].values, y_v)
    rho_proba_t = rf_t2.predict_proba(rho_sub_clean[['alpha_missense', 'baseline_score']].values)[:, 1]
    rho_auc = roc_auc_score(rho_sub_clean['label'].values, rho_proba_t)

    print(f"  {dt:<12.1f} {rt:<12.1f} {len(v2r_sub):>6d} {n_res:>6d} {n_nr:>6d} {pct:>5.1f}% {v2r_auc:>10.3f} {rho_auc:>10.3f}")


# ========================================
print(f"\n{'='*75}")
print("STEP 3: PERMUTATION TEST FOR RHODOPSIN EXTERNAL VALIDATION")
print("=" * 75)

# Observed AUROC
observed_auroc = roc_auc_score(rho_y, rho_proba_b)

n_perm = 1000
perm_aurocs = []
for i in range(n_perm):
    perm_y = np.random.permutation(rho_y)
    perm_aurocs.append(roc_auc_score(perm_y, rho_proba_b))

perm_aurocs = np.array(perm_aurocs)
p_value = (perm_aurocs >= observed_auroc).mean()

print(f"\n  Observed Rhodopsin AUROC: {observed_auroc:.3f}")
print(f"  Permutation distribution: mean={perm_aurocs.mean():.3f}, std={perm_aurocs.std():.3f}")
print(f"  P-value (one-sided): {p_value:.4f}")
print(f"  {'✓ Significant (p < 0.001)' if p_value < 0.001 else '⚠ Check significance'}")

print(f"\n{'='*75}")
print("SUMMARY")
print("=" * 75)
print(f"""
  Bootstrap 95% CIs:
    V2R B5 (full):         [{lo_v:.3f}, {hi_v:.3f}]
    V2R no baseline:       [{lo_nb:.3f}, {hi_nb:.3f}]
    Rhodopsin AM only:     [{lo_a:.3f}, {hi_a:.3f}]
    Rhodopsin AM+baseline: [{lo_b:.3f}, {hi_b:.3f}]

  Rhodopsin permutation test:
    p = {p_value:.4f}
""")

print("=== Phase 4 (Robustness) complete ===")
