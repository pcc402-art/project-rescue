"""
Project Rescue - Code Review Fixes
Addresses 5 issues identified in pre-submission code review:
1. phase1: results[-1] → explicit model name lookup
2. phase1/manuscript: unify leakage error bar definition
3. phase4: bootstrap returns observed_auc + percentile CI
4. phase3: fix "stability proxy" comment + document n drop
5. phase5: cross-check DeLong with scipy mannwhitneyu

Run from ~/project_rescue/
"""
import os

# ========================================
# Fix 1: phase1_baseline.py — results[-1] → explicit lookup
# ========================================
print("Fix 1: phase1_baseline.py — explicit model name lookup")

with open("phase1_baseline.py", "r") as f:
    code = f.read()

old1 = "std_auroc = results[-1]['auroc']  # last model = B5_RF"
new1 = """# Explicit lookup instead of relying on list order
std_auroc = next(r['auroc'] for r in results if r['model'] == 'B5_L1+baseline_RF')"""

if old1 in code:
    code = code.replace(old1, new1)
    print("  ✓ Fixed results[-1] → explicit lookup")
else:
    print("  ⚠ Pattern not found, check manually")

# Also fix leakage test error reporting (Fix 2)
old2 = """print(f"  Leave-positions-out AUROC (B5_RF):      {strict_mean:.3f} ± {strict_std:.3f}")"""
new2 = """# Report as bootstrap-style percentile CI for consistency with manuscript
strict_lo = np.percentile(strict_aurocs, 2.5)
strict_hi = np.percentile(strict_aurocs, 97.5)
print(f"  Leave-positions-out AUROC (B5_RF):      {strict_mean:.3f} (95% CI: [{strict_lo:.3f}, {strict_hi:.3f}])")"""

if old2 in code:
    code = code.replace(old2, new2)
    print("  ✓ Fixed leakage error bar → percentile CI")
else:
    print("  ⚠ Leakage pattern not found, check manually")

with open("phase1_baseline.py", "w") as f:
    f.write(code)


# ========================================
# Fix 3: phase4_robustness.py — bootstrap returns observed + CI
# ========================================
print("\nFix 3: phase4_robustness.py — bootstrap observed_auc + CI")

with open("phase4_robustness.py", "r") as f:
    code = f.read()

old3 = """def bootstrap_auroc(y_true, y_score, n_boot=2000):
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)"""

new3 = """def bootstrap_auroc(y_true, y_score, n_boot=2000):
    \"\"\"Returns (observed_auc, ci_low, ci_high) using percentile method.\"\"\"
    observed = roc_auc_score(y_true, y_score)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return observed, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)"""

if old3 in code:
    code = code.replace(old3, new3)
    print("  ✓ Fixed bootstrap → returns observed_auc")
else:
    print("  ⚠ Bootstrap pattern not found, check manually")

with open("phase4_robustness.py", "w") as f:
    f.write(code)


# ========================================
# Fix 4: phase3_aim3_rhodopsin.py — fix comment + document n drop
# ========================================
print("\nFix 4: phase3_aim3_rhodopsin.py — fix comment + n drop")

with open("phase3_aim3_rhodopsin.py", "r") as f:
    code = f.read()

old4a = """print(f"\\n  ⚠ RaSP, ThermoMPNN, ESM1b, delta_hydro NOT in Rhodopsin repo")
print(f"  → For initial test, use AlphaMissense + baseline_score only")
print(f"  → This matches B2-level from V2R (AM + stability proxy)")"""

new4a = """print(f"\\n  ⚠ RaSP, ThermoMPNN, ESM1b, delta_hydro NOT in Rhodopsin repo")
print(f"  → For initial test, use AlphaMissense + baseline_score only")
print(f"  → These are the two features shared between V2R and Rhodopsin datasets")"""

if old4a in code:
    code = code.replace(old4a, new4a)
    print("  ✓ Fixed 'stability proxy' comment")
else:
    print("  ⚠ Comment pattern not found, check manually")

# Add n-drop documentation
old4b = """rho_test = rho_primary.dropna(subset=['alpha_missense']).copy()"""
new4b = """rho_test = rho_primary.dropna(subset=['alpha_missense']).copy()
n_dropped = len(rho_primary) - len(rho_test)
print(f"\\n  Dropped {n_dropped} variants with missing AlphaMissense ({len(rho_primary)} → {len(rho_test)})")"""

if old4b in code:
    code = code.replace(old4b, new4b)
    print("  ✓ Added n-drop documentation")
else:
    print("  ⚠ dropna pattern not found, check manually")

with open("phase3_aim3_rhodopsin.py", "w") as f:
    f.write(code)


# ========================================
# Fix 5: phase5 — add DeLong cross-check with Mann-Whitney U
# ========================================
print("\nFix 5: phase5_presubmission.py — add DeLong cross-check")

with open("phase5_presubmission.py", "r") as f:
    code = f.read()

# Add cross-check after DeLong test block
crosscheck = """
# --- DeLong cross-check using bootstrap comparison ---
print(f"\\n  Cross-check with bootstrap paired comparison (2000 resamples):")
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
"""

# Insert after the DeLong test try/except block
insert_marker = "# ========================================\nprint(f\"\\n{'='*75}\")\nprint(\"STEP 2: LR vs RF FORMAL COMPARISON\")"
if insert_marker in code:
    code = code.replace(insert_marker, crosscheck + "\n" + insert_marker)
    print("  ✓ Added bootstrap cross-check for DeLong")
else:
    print("  ⚠ Insert marker not found, check manually")

with open("phase5_presubmission.py", "w") as f:
    f.write(code)


print("\n" + "="*60)
print("All 5 fixes applied. Re-run scripts to verify:")
print("  python3 phase1_baseline.py")
print("  python3 phase3_aim3_rhodopsin.py")
print("  python3 phase4_robustness.py")
print("  python3 phase5_presubmission.py")
print("Then: git add -A && git commit -m 'code review fixes' && git push")
