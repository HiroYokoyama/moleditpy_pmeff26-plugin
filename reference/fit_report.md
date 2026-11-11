# PMEFF26 fit report

Reference: wB97X-D/def2-TZVP (GAMESS).  Training molecules: 3; validation: 0; scans: 1.
Parameters: 4 bonds, 4 angles, 1 torsion classes, 0 impropers, 3 charge increments.

| set | molecules | covered | bond RMSD (A) | angle RMSD (deg) | frequency RMSD (cm-1) |
|---|---|---|---|---|---|
| train | 3 | 3 | 0.0007 | 0.07 | 116 |
| validate | 0 | 0 | nan | nan | nan |

Errors are force-field-optimized geometry and frequencies against DFT. The validation row is the honest accuracy figure: those molecules were not fitted.

DFT-referenced energy increments: training RMSE 0.00 kcal/mol.

## Torsion scan fit (RMSE, kcal/mol)

- `*|c3-c3|*`: 0.01
