# Project Rescue

Analysis scripts for: **"Computational prediction of pharmacological chaperone rescuability in GPCR missense variants with cross-protein transfer from V2R to Rhodopsin"**

Po-Chun Chen, Independent Researcher, Tainan, Taiwan

## Scripts

| Script | Description |
|--------|-------------|
| `phase1_baseline.py` | Progressive baseline models (B0–B5), feature importance, leakage test |
| `phase2_aim2_mechanism.py` | Mechanistic analysis: two not-rescued subgroups, baseline-controlled analysis |
| `phase3_aim3_rhodopsin.py` | Rhodopsin external validation (2-feature, initial) |
| `phase3_v2_full_rerun.py` | **Rhodopsin external validation (4-feature, final): AUROC 0.843** |
| `phase4_robustness.py` | Bootstrap CIs, threshold sensitivity, permutation test |
| `phase5_presubmission.py` | LR vs RF comparison, clinical variant overlay |
| `figures_v2.py` | Publication-quality figure generation |

## Data Sources

- **V2R DMS data:** Mighell & Lehner (2025) Supplementary Tables, DOI: [10.1038/s41594-025-01659-6](https://doi.org/10.1038/s41594-025-01659-6)
- **Rhodopsin DMS data:** Manian et al. (2025), GitHub: [octantbio/rho-dms](https://github.com/octantbio/rho-dms)

## Reproducing the Analysis

1. Download V2R supplementary data (MOESM3 XLSX) from the Nature NSMB paper
2. Clone the Rhodopsin repo: `git clone https://github.com/octantbio/rho-dms.git`
3. Place V2R data in `raw_data/` directory
4. Run scripts in order: phase1 → phase2 → phase3_v2_full_rerun → phase4 → phase5 → figures_v2

## Requirements

Python 3.x with: pandas, numpy, scikit-learn, scipy, matplotlib, openpyxl, torch, fair-esm

## License

MIT
