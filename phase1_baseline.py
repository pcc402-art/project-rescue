"""
Project Rescue - Phase 1: Baseline Models
Go/no-go test: can computational features predict variant rescuability?

Progressive baselines:
  B0: Majority class (always predict rescued)
  B1: AlphaMissense only
  B2: AM + ΔΔG (RaSP, ThermoMPNN)
  B3: AM + ΔΔG + ESM1b
  B4: Full Layer 1 (AM + ΔΔG + ESM1b + hydrophobicity)
  B5: Layer 1 + baseline expression

Models: Logistic Regression + Random Forest
Evaluation: 5-fold stratified CV, AUROC / F1 / Precision / Recall
Strict validation: leave-positions-out (test for positional leakage)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ========================================
# Load data
# ========================================
df = pd.read_csv("primary_dataset.csv", index_col=0)
y = df['label'].values

print(f"Primary dataset: {len(df)} variants")
print(f"  Rescued (label=1): {y.sum()}")
print(f"  Not rescued (label=0): {len(y) - y.sum()}")
print(f"  Class ratio: {y.mean():.3f} rescued\n")

# ========================================
# Progressive feature sets
# ========================================
feature_sets = {
    "B1_AM_only":        ['alpha_missense'],
    "B2_AM+ddG":         ['alpha_missense', 'RaSP', 'ThermoMPNN'],
    "B3_AM+ddG+ESM":     ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b'],
    "B4_full_Layer1":    ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro'],
    "B5_L1+baseline":    ['alpha_missense', 'RaSP', 'ThermoMPNN', 'ESM1b', 'delta_hydro', 'ctrls_comb'],
}

# ========================================
# CV setup
# ========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'auroc':     'roc_auc',
    'f1':        make_scorer(f1_score),
    'precision': make_scorer(precision_score),
    'recall':    make_scorer(recall_score),
}

# ========================================
# B0: Majority class baseline
# ========================================
b0_f1 = f1_score(y, np.ones_like(y))

print("=" * 75)
print(f"{'MODEL':<35s} {'AUROC':>10s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s}")
print("=" * 75)
print(f"{'B0_majority_class':<35s} {'0.500':>10s} {b0_f1:>7.3f} {'--':>7s} {'--':>7s}")

# ========================================
# Progressive baselines
# ========================================
results = []

for name, features in feature_sets.items():
    X = df[features].values

    # Logistic Regression
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_cv = cross_validate(lr_pipe, X, y, cv=cv, scoring=scoring)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_cv = cross_validate(rf, X, y, cv=cv, scoring=scoring)

    for model_tag, cv_res in [('LR', lr_cv), ('RF', rf_cv)]:
        tag = f"{name}_{model_tag}"
        a   = cv_res['test_auroc'].mean()
        a_s = cv_res['test_auroc'].std()
        f   = cv_res['test_f1'].mean()
        p   = cv_res['test_precision'].mean()
        r   = cv_res['test_recall'].mean()
        print(f"{tag:<35s} {a:.3f}±{a_s:.3f} {f:>7.3f} {p:>7.3f} {r:>7.3f}")
        results.append({
            'model': tag, 'auroc': a, 'auroc_std': a_s,
            'f1': f, 'precision': p, 'recall': r,
            'n_features': len(features), 'features': ','.join(features)
        })

# ========================================
# Feature importance (RF, B5)
# ========================================
print(f"\n{'='*75}")
print("FEATURE IMPORTANCE (Random Forest, B5_L1+baseline)")
print("=" * 75)

best_feats = feature_sets['B5_L1+baseline']
rf_full = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_full.fit(df[best_feats].values, y)

for feat, imp in sorted(zip(best_feats, rf_full.feature_importances_), key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"  {feat:<25s}: {imp:.4f} {bar}")

# ========================================
# Strict validation: leave-positions-out
# ========================================
print(f"\n{'='*75}")
print("STRICT VALIDATION: Leave-positions-out (10 folds)")
print("=" * 75)

positions = df['pos'].values
unique_pos = np.unique(positions)
np.random.seed(42)
np.random.shuffle(unique_pos)
pos_folds = np.array_split(unique_pos, 10)

X_all = df[best_feats].values
strict_aurocs = []

for test_pos in pos_folds:
    test_mask  = np.isin(positions, test_pos)
    train_mask = ~test_mask
    if y[test_mask].sum() == 0 or y[test_mask].sum() == len(y[test_mask]):
        continue
    rf_s = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_s.fit(X_all[train_mask], y[train_mask])
    proba = rf_s.predict_proba(X_all[test_mask])[:, 1]
    strict_aurocs.append(roc_auc_score(y[test_mask], proba))

std_auroc = results[-1]['auroc']  # last model = B5_RF
strict_mean = np.mean(strict_aurocs)
strict_std  = np.std(strict_aurocs)

print(f"  Standard 5-fold CV AUROC (B5_RF):      {std_auroc:.3f}")
print(f"  Leave-positions-out AUROC (B5_RF):      {strict_mean:.3f} ± {strict_std:.3f}")
print(f"  Drop: {std_auroc - strict_mean:.3f}")
print(f"  (Large drop = positional leakage concern)")

# ========================================
# Save
# ========================================
pd.DataFrame(results).to_csv("baseline_results.csv", index=False)
print(f"\nSaved: baseline_results.csv")
print("\n=== Phase 1 complete ===")
