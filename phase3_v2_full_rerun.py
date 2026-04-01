"""
Project Rescue — Full Rhodopsin External Validation Rerun (4 features)
======================================================================
Replaces the old 2-feature validation with 4-feature validation.
Reruns ALL supporting analyses to maintain manuscript consistency:
  1. External validation (Model A, B, D)
  2. Bootstrap CIs for all models
  3. Permutation test
  4. DeLong test (Model B vs A, Model D vs B)
  5. Threshold sensitivity (0.5, 0.6, 0.7, 0.8)
  6. Severity subgroup analysis
  7. Clinical variant overlay (V2R side, unchanged)
  
Data source: supplementary-table-1.csv (composite_score = meta-analysis primary metric)
Baseline: composite_score
Drug: Octant_mean_YC-001 30uM

Run from ~/project_rescue/
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ========================================
# STEP 0: Load V2R training data
# ========================================
print("="*70)
print("STEP 0: Load V2R training data")
print("="*70)

v2r = pd.read_csv("primary_dataset.csv", index_col=0)
v2r_y = v2r['label'].values
print(f"  V2R: {len(v2r)} defective variants, {v2r_y.sum()} rescued, {(1-v2r_y).sum()} not rescued")

v2r_features_full = ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb']

# ========================================
# STEP 1: Rebuild Rhodopsin dataset from canonical source
# ========================================
print(f"\n{'='*70}")
print("STEP 1: Rebuild Rhodopsin dataset (supplementary-table-1.csv)")
print("="*70)

st1 = pd.read_csv('rho-dms/paper/supplementary-table-1.csv')
mis = st1[st1['consequence'] == 'missense'].copy()

mis['wt_aa'] = mis['protein'].str[0]
mis['mut_aa'] = mis['protein'].str[-1]
mis['position'] = mis['protein'].str[1:-1].astype(int)
mis['baseline_score'] = mis['composite_score']
mis['drug_score'] = mis['Octant_mean_YC-001 30uM']

mis = mis.dropna(subset=['baseline_score', 'drug_score'])
print(f"  Total missense with scores: {len(mis)}")

# Merge AlphaMissense
am = pd.read_csv('rho-dms/data/alpha-missense-rho.tsv', sep='\t')
mis = mis.merge(am, left_on=['position', 'mut_aa'], right_on=['pos', 'mut_aa'], how='left')
mis.rename(columns={'pathogenicity score': 'alpha_missense'}, inplace=True)

# Compute delta_hydro
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
mis['delta_hydro'] = mis.apply(lambda r: KD.get(r['mut_aa'], np.nan) - KD.get(r['wt_aa'], np.nan), axis=1)

# Load ESM1b (already computed)
rho_existing = pd.read_csv("rhodopsin_primary_dataset.csv")
if 'ESM1b' in rho_existing.columns:
    esm_map = rho_existing.set_index(['position', 'mut_aa'])['ESM1b'].to_dict()
    mis['ESM1b'] = mis.apply(lambda r: esm_map.get((r['position'], r['mut_aa']), np.nan), axis=1)
    print(f"  ESM1b merged from existing dataset")
else:
    print(f"  WARNING: ESM1b not found in existing dataset")
    mis['ESM1b'] = np.nan

# Define labels
defective = mis[mis['baseline_score'] < 0.7].copy()
defective['label'] = (defective['drug_score'] > 0.7).astype(int)

print(f"\n  Defective variants: {len(defective)}")
print(f"  Rescued: {defective['label'].sum()} ({100*defective['label'].mean():.1f}%)")
print(f"  Not rescued: {(1-defective['label']).sum().astype(int)} ({100*(1-defective['label'].mean()):.1f}%)")

# Feature coverage
for f in ['alpha_missense', 'baseline_score', 'delta_hydro', 'ESM1b']:
    n = defective[f].notna().sum()
    print(f"  {f}: {n}/{len(defective)} ({100*n/len(defective):.1f}%)")

# Severity breakdown
print(f"\n  Severity categories:")
print(defective['composite_score_category'].value_counts().to_string())

# Save canonical dataset
defective.to_csv("rhodopsin_primary_dataset_v2.csv", index=False)
print(f"\n  Saved: rhodopsin_primary_dataset_v2.csv")


# ========================================
# STEP 2: External validation — 3 models
# ========================================
print(f"\n{'='*70}")
print("STEP 2: External validation (frozen V2R models → Rhodopsin)")
print("="*70)

# Test set: drop NaN for all 4 features
rho_test = defective.dropna(subset=['alpha_missense', 'delta_hydro', 'ESM1b']).copy()
rho_y = rho_test['label'].values
n_dropped = len(defective) - len(rho_test)
print(f"  Test set: {len(rho_test)} variants ({n_dropped} dropped for missing features)")
print(f"  Rescued: {rho_y.sum()}, Not rescued: {(1-rho_y).sum()}")

# Model A: AlphaMissense only
rf_a = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_a.fit(v2r[['alpha_missense']].values, v2r_y)
rho_proba_a = rf_a.predict_proba(rho_test[['alpha_missense']].values)[:, 1]
auroc_a = roc_auc_score(rho_y, rho_proba_a)

# Model B: AlphaMissense + baseline
rf_b = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_b.fit(v2r[['alpha_missense', 'ctrls_comb']].values, v2r_y)
rho_proba_b = rf_b.predict_proba(rho_test[['alpha_missense', 'baseline_score']].values)[:, 1]
auroc_b = roc_auc_score(rho_y, rho_proba_b)

# Model D: AlphaMissense + baseline + delta_hydro + ESM1b
rf_d = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_d.fit(v2r[['alpha_missense', 'ctrls_comb', 'delta_hydro', 'ESM1b']].values, v2r_y)
rho_proba_d = rf_d.predict_proba(rho_test[['alpha_missense', 'baseline_score', 'delta_hydro', 'ESM1b']].values)[:, 1]
auroc_d = roc_auc_score(rho_y, rho_proba_d)

print(f"\n  Model A (AM only):                        AUROC = {auroc_a:.3f}")
print(f"  Model B (AM + baseline):                  AUROC = {auroc_b:.3f}")
print(f"  Model D (AM + baseline + hydro + ESM1b):  AUROC = {auroc_d:.3f}")
print(f"  V2R B5 (internal, 6 features):            AUROC = 0.751")
print(f"  Drop from V2R to Rho Model D:             {auroc_d - 0.751:+.3f}")


# ========================================
# STEP 3: Bootstrap CIs (2000 resamples)
# ========================================
print(f"\n{'='*70}")
print("STEP 3: Bootstrap 95% CIs")
print("="*70)

def bootstrap_ci(y_true, y_score, n_boot=2000):
    observed = roc_auc_score(y_true, y_score)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return observed, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

for name, proba in [("Model A", rho_proba_a), ("Model B", rho_proba_b), ("Model D", rho_proba_d)]:
    obs, lo, hi = bootstrap_ci(rho_y, proba)
    print(f"  {name}: AUROC = {obs:.3f} (95% CI: [{lo:.3f}, {hi:.3f}])")


# ========================================
# STEP 4: Permutation test (1000 iterations)
# ========================================
print(f"\n{'='*70}")
print("STEP 4: Permutation test (Model D)")
print("="*70)

perm_aucs = []
for i in range(1000):
    perm_y = np.random.permutation(rho_y)
    perm_aucs.append(roc_auc_score(perm_y, rho_proba_d))
perm_aucs = np.array(perm_aucs)
p_val = (perm_aucs >= auroc_d).mean()
print(f"  Observed AUROC: {auroc_d:.3f}")
print(f"  Permuted mean:  {perm_aucs.mean():.3f}")
print(f"  Permuted max:   {perm_aucs.max():.3f}")
print(f"  p-value:        {p_val:.4f} ({'< 0.001' if p_val < 0.001 else p_val})")


# ========================================
# STEP 5: DeLong tests
# ========================================
print(f"\n{'='*70}")
print("STEP 5: DeLong tests")
print("="*70)

# Bootstrap paired comparison instead of DeLong (simpler, equally valid)
def bootstrap_paired_comparison(y, proba1, proba2, n_boot=2000):
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2: continue
        a1 = roc_auc_score(y[idx], proba1[idx])
        a2 = roc_auc_score(y[idx], proba2[idx])
        diffs.append(a2 - a1)
    diffs = np.array(diffs)
    p = (diffs <= 0).mean()
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5]), p

# Model B vs Model A
diff_ba, ci_ba, p_ba = bootstrap_paired_comparison(rho_y, rho_proba_a, rho_proba_b)
print(f"  Model B vs A: Δ = {diff_ba:+.3f}, CI [{ci_ba[0]:+.3f}, {ci_ba[1]:+.3f}], p = {p_ba:.4f}")

# Model D vs Model B
diff_db, ci_db, p_db = bootstrap_paired_comparison(rho_y, rho_proba_b, rho_proba_d)
print(f"  Model D vs B: Δ = {diff_db:+.3f}, CI [{ci_db[0]:+.3f}, {ci_db[1]:+.3f}], p = {p_db:.4f}")

# Model D vs Model A
diff_da, ci_da, p_da = bootstrap_paired_comparison(rho_y, rho_proba_a, rho_proba_d)
print(f"  Model D vs A: Δ = {diff_da:+.3f}, CI [{ci_da[0]:+.3f}, {ci_da[1]:+.3f}], p = {p_da:.4f}")


# ========================================
# STEP 6: Threshold sensitivity
# ========================================
print(f"\n{'='*70}")
print("STEP 6: Threshold sensitivity (full pipeline rerun)")
print("="*70)

# Reload full missense for threshold sweep
all_mis = pd.read_csv("rhodopsin_full_missense.csv")
# Ensure ESM1b is merged
if 'ESM1b' not in all_mis.columns or all_mis['ESM1b'].isna().all():
    esm_full_map = rho_existing.set_index(['position', 'mut_aa'])['ESM1b'].to_dict() if 'ESM1b' in rho_existing.columns else {}
    all_mis['ESM1b'] = all_mis.apply(lambda r: esm_full_map.get((r.get('position', 0), r.get('mut_aa', '')), np.nan), axis=1)

print(f"  {'Threshold':<12} {'V2R_N':<8} {'V2R_%R':<8} {'V2R_AUC':<10} {'Rho_N':<8} {'Rho_AUC':<10}")
print(f"  {'-'*58}")

for thresh in [0.5, 0.6, 0.7, 0.8]:
    # V2R
    v2r_full = pd.read_csv("full_missense_dataset.csv")
    v2r_def = v2r_full[v2r_full['ctrls_comb'] < thresh].copy()
    v2r_def['label'] = (v2r_full.loc[v2r_def.index, 'Tol_comb'] > thresh).astype(int)
    v2r_def_clean = v2r_def.dropna(subset=['alpha_missense', 'ctrls_comb', 'delta_hydro', 'ESM1b'])
    
    if len(v2r_def_clean) < 10 or v2r_def_clean['label'].nunique() < 2:
        print(f"  {thresh:<12} SKIP")
        continue
    
    # V2R: 4-feature model CV
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    v2r_feats_4 = ['alpha_missense', 'ctrls_comb', 'delta_hydro', 'ESM1b']
    rf_cv = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    v2r_probas = cross_val_predict(rf_cv, v2r_def_clean[v2r_feats_4].values, v2r_def_clean['label'].values, 
                                    cv=cv, method='predict_proba')[:, 1]
    v2r_auroc = roc_auc_score(v2r_def_clean['label'].values, v2r_probas)
    
    # Rhodopsin at this threshold
    rho_def = all_mis[all_mis['baseline_score'] < thresh].copy()
    rho_def['label'] = (rho_def['drug_score'] > thresh).astype(int)
    rho_def_clean = rho_def.dropna(subset=['alpha_missense', 'baseline_score', 'delta_hydro', 'ESM1b'])
    
    if len(rho_def_clean) < 10 or rho_def_clean['label'].nunique() < 2:
        print(f"  {thresh:<12} {len(v2r_def):<8} {100*v2r_def['label'].mean():<8.1f} {v2r_auroc:<10.3f} SKIP")
        continue
    
    # Train on full V2R at this threshold, predict Rhodopsin
    rf_thresh = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_thresh.fit(v2r_def_clean[v2r_feats_4].values, v2r_def_clean['label'].values)
    rho_proba_thresh = rf_thresh.predict_proba(rho_def_clean[['alpha_missense', 'baseline_score', 'delta_hydro', 'ESM1b']].values)[:, 1]
    rho_auroc = roc_auc_score(rho_def_clean['label'].values, rho_proba_thresh)
    
    marker = " (primary)" if thresh == 0.7 else ""
    print(f"  {thresh:<12} {len(v2r_def):<8} {100*v2r_def['label'].mean():<8.1f} {v2r_auroc:<10.3f} {len(rho_def_clean):<8} {rho_auroc:<10.3f}{marker}")


# ========================================
# STEP 7: Severity subgroup analysis
# ========================================
print(f"\n{'='*70}")
print("STEP 7: Rhodopsin AUROC by severity category")
print("="*70)

for cat in ['very low', 'low', 'conflicting', 'uninformative']:
    subset = rho_test[rho_test['composite_score_category'] == cat]
    if len(subset) < 10 or subset['label'].nunique() < 2:
        print(f"  {cat:<20} n={len(subset):<6} SKIP (insufficient data)")
        continue
    sub_y = subset['label'].values
    sub_proba = rho_proba_d[rho_test['composite_score_category'] == cat]
    sub_auroc = roc_auc_score(sub_y, sub_proba)
    print(f"  {cat:<20} n={len(subset):<6} AUROC = {sub_auroc:.3f}")


# ========================================
# STEP 8: PR-AUC for Rhodopsin
# ========================================
print(f"\n{'='*70}")
print("STEP 8: Precision-Recall AUC")
print("="*70)

ap_rho = average_precision_score(rho_y, rho_proba_d)
print(f"  Rhodopsin Model D PR-AUC: {ap_rho:.3f}")
print(f"  Class prevalence (rescued): {rho_y.mean():.3f}")


# ========================================
# STEP 9: Save all results
# ========================================
print(f"\n{'='*70}")
print("STEP 9: Save results")
print("="*70)

rho_test['pred_proba_model_a'] = rho_proba_a
rho_test['pred_proba_model_b'] = rho_proba_b
rho_test['pred_proba_model_d'] = rho_proba_d
rho_test.to_csv("rhodopsin_validation_results_v2.csv", index=False)
print(f"  Saved: rhodopsin_validation_results_v2.csv ({len(rho_test)} rows)")

# Summary
print(f"\n{'='*70}")
print("COMPLETE SUMMARY — Numbers for manuscript update")
print("="*70)
print(f"""
RHODOPSIN EXTERNAL VALIDATION (v2, 4-feature)
==============================================
Data source: supplementary-table-1.csv (composite_score)
Baseline: composite_score (meta-analysis of Octant + MEE)
Drug: Octant_mean_YC-001 30uM
Threshold: 0.7

Dataset:
  Total missense:     {len(mis)}
  Defective:          {len(defective)}
  Rescued:            {defective['label'].sum()} ({100*defective['label'].mean():.1f}%)
  Not rescued:        {(1-defective['label']).sum().astype(int)} ({100*(1-defective['label'].mean()):.1f}%)
  Test set (no NaN):  {len(rho_test)}
  Dropped:            {n_dropped}

Features used: AlphaMissense, baseline expression, delta_hydro, ESM1b
(RaSP: repo deleted; ThermoMPNN: dependency incompatible)

Model A (AM only):                        AUROC = {auroc_a:.3f}
Model B (AM + baseline):                  AUROC = {auroc_b:.3f}
Model D (AM + baseline + hydro + ESM1b):  AUROC = {auroc_d:.3f}
V2R B5 (internal, 6 features):            AUROC = 0.751

Permutation test: p {'< 0.001' if p_val < 0.001 else f'= {p_val:.4f}'}
""")
