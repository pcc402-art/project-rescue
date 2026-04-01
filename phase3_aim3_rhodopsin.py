"""
Project Rescue - Phase 3: Aim 3 Rhodopsin External Validation
Step 1: Data compatibility check + dataset construction
Step 2: Apply frozen V2R model (Layer 1 only) to predict Rhodopsin rescue
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ========================================
# STEP 1: Build Rhodopsin dataset
# ========================================
print("=" * 75)
print("STEP 1: RHODOPSIN DATA COMPATIBILITY CHECK")
print("=" * 75)

# 1a. Load rescaled sumstats (has both DMSO and YC-001)
rescaled = pd.read_csv("rho-dms/sumstats/Octant-RHO-cleaned-rescaled.sumstats.tsv", sep='\t')
print(f"\n  Rescaled sumstats: {len(rescaled)} rows")
print(f"  Conditions: {rescaled['condition'].unique()}")

# Split into baseline and drug
baseline = rescaled[rescaled['condition'] == 'DMSO_0'][['pos', 'aa', 'rescaled_estimate', 'rescaled_error']].copy()
baseline.columns = ['pos', 'mut_aa', 'baseline_score', 'baseline_se']

drug = rescaled[rescaled['condition'] == 'OCNT-0022155_10'][['pos', 'aa', 'rescaled_estimate', 'rescaled_error']].copy()
drug.columns = ['pos', 'mut_aa', 'drug_score', 'drug_se']

# Merge baseline + drug
rho = baseline.merge(drug, on=['pos', 'mut_aa'], how='inner')
print(f"  Variants with both baseline AND YC-001: {len(rho)}")

# 1b. Add composite scores from meta-analysis
meta = pd.read_csv("rho-dms/sumstats/meta_analysis_results.csv")
meta_sub = meta[['pos', 'mut_aa', 'wt_aa', 'consequence', 'composite_score',
                  'trafficking_score_category', 'abnormal_trafficking_score_confidence']].copy()
rho = rho.merge(meta_sub, on=['pos', 'mut_aa'], how='left')

# 1c. Add AlphaMissense
am = pd.read_csv("rho-dms/data/alpha-missense-rho.tsv", sep='\t')
am.columns = ['pos', 'mut_aa', 'alpha_missense']
rho = rho.merge(am, on=['pos', 'mut_aa'], how='left')

# 1d. Filter to missense only
rho = rho[rho['consequence'] == 'missense'].copy()
print(f"  Missense variants: {len(rho)}")

# 1e. Compute rescue score
rho['rescue_score'] = rho['drug_score'] - rho['baseline_score']

# 1f. Apply same label rules as V2R
DEFECTIVE_THRESHOLD = 0.7
RESCUE_THRESHOLD = 0.7

rho['is_defective'] = rho['baseline_score'] < DEFECTIVE_THRESHOLD
rho['is_rescued'] = rho['drug_score'] > RESCUE_THRESHOLD

rho_primary = rho[rho['is_defective']].copy()
rho_primary['label'] = rho_primary['is_rescued'].astype(int)

print(f"\n  === LABEL RULES (same as V2R) ===")
print(f"  Defective threshold: baseline_score < {DEFECTIVE_THRESHOLD}")
print(f"  Rescue threshold:    drug_score > {RESCUE_THRESHOLD}")
print(f"  Total missense:          {len(rho)}")
print(f"  Defective:               {len(rho_primary)}")
print(f"    Rescued (label=1):     {(rho_primary['label']==1).sum()} ({100*(rho_primary['label']==1).mean():.1f}%)")
print(f"    Not rescued (label=0): {(rho_primary['label']==0).sum()} ({100*(rho_primary['label']==0).mean():.1f}%)")

# 1g. Feature availability
print(f"\n  === FEATURE AVAILABILITY ===")
print(f"  alpha_missense: {rho_primary['alpha_missense'].notna().sum()} / {len(rho_primary)}")
print(f"  baseline_score: {rho_primary['baseline_score'].notna().sum()} / {len(rho_primary)}")

# Note: RaSP, ThermoMPNN, ESM1b, delta_hydro are NOT in Rhodopsin data
# We need to compute them OR use only features available in both
print(f"\n  ⚠ RaSP, ThermoMPNN, ESM1b, delta_hydro NOT in Rhodopsin repo")
print(f"  → For initial test, use AlphaMissense + baseline_score only")
print(f"  → This matches B2-level from V2R (AM + stability proxy)")

# ========================================
# STEP 2: APPLY FROZEN V2R MODEL
# ========================================
print(f"\n{'='*75}")
print("STEP 2: EXTERNAL VALIDATION WITH FROZEN V2R MODEL")
print("=" * 75)

# Load V2R training data
v2r = pd.read_csv("primary_dataset.csv", index_col=0)
v2r_y = v2r['label'].values

# ---- Model A: AlphaMissense only (naive baseline) ----
v2r_X_am = v2r[['alpha_missense']].values

rf_am = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_am.fit(v2r_X_am, v2r_y)

# Predict on Rhodopsin
rho_test = rho_primary.dropna(subset=['alpha_missense']).copy()
rho_X_am = rho_test[['alpha_missense']].values
rho_y = rho_test['label'].values

proba_am = rf_am.predict_proba(rho_X_am)[:, 1]
auroc_am = roc_auc_score(rho_y, proba_am)
ap_am = average_precision_score(rho_y, proba_am)

print(f"\n  Model A: AlphaMissense only (trained on V2R)")
print(f"    Rhodopsin AUROC: {auroc_am:.3f}")
print(f"    Rhodopsin AP:    {ap_am:.3f}")
print(f"    (random = 0.500, class prevalence = {rho_y.mean():.3f})")

# ---- Model B: AlphaMissense + baseline expression ----
v2r_X_b = v2r[['alpha_missense', 'ctrls_comb']].values

rf_b = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_b.fit(v2r_X_b, v2r_y)

rho_X_b = rho_test[['alpha_missense', 'baseline_score']].values
proba_b = rf_b.predict_proba(rho_X_b)[:, 1]
auroc_b = roc_auc_score(rho_y, proba_b)
ap_b = average_precision_score(rho_y, proba_b)

print(f"\n  Model B: AlphaMissense + baseline expression (trained on V2R)")
print(f"    Rhodopsin AUROC: {auroc_b:.3f}")
print(f"    Rhodopsin AP:    {ap_b:.3f}")

# ---- Success criteria ----
print(f"\n{'='*75}")
print("SUCCESS CRITERIA CHECK")
print("=" * 75)
print(f"  Pre-defined: Model must significantly beat random (0.500)")
print(f"               AND beat naive AlphaMissense-only baseline")
print(f"")
print(f"  Model A (AM only):          AUROC = {auroc_am:.3f}  {'✓ > 0.5' if auroc_am > 0.5 else '✗'}")
print(f"  Model B (AM + baseline):    AUROC = {auroc_b:.3f}  {'✓ > 0.5' if auroc_b > 0.5 else '✗'}")
print(f"  Model B > Model A:          {auroc_b:.3f} > {auroc_am:.3f}  {'✓' if auroc_b > auroc_am else '✗'}")

# ---- Error analysis by trafficking category ----
print(f"\n{'='*75}")
print("ERROR ANALYSIS BY TRAFFICKING CATEGORY")
print("=" * 75)

rho_test['pred_proba'] = proba_b
rho_test['pred_label'] = (proba_b > 0.5).astype(int)
rho_test['correct'] = (rho_test['pred_label'] == rho_test['label']).astype(int)

for cat in rho_test['trafficking_score_category'].dropna().unique():
    sub = rho_test[rho_test['trafficking_score_category'] == cat]
    if len(sub) > 10 and sub['label'].nunique() == 2:
        acc = sub['correct'].mean()
        auc = roc_auc_score(sub['label'], sub['pred_proba'])
        print(f"  {cat:<15s}: n={len(sub):4d}, accuracy={acc:.3f}, AUROC={auc:.3f}")
    else:
        print(f"  {cat:<15s}: n={len(sub):4d}, skipped (too few or single class)")

# ---- Save ----
rho_primary.to_csv("rhodopsin_primary_dataset.csv", index=False)
rho_test.to_csv("rhodopsin_validation_results.csv", index=False)
print(f"\nSaved: rhodopsin_primary_dataset.csv, rhodopsin_validation_results.csv")
print("\n=== Phase 3 (Aim 3) complete ===")
