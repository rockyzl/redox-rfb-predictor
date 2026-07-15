#!/usr/bin/env python
"""Train the RDKit-only baseline (full dataset) and the hybrid RDKit+xTB model
(xTB subset), evaluate with cross-validation, and write models + reports.

Outputs:
  models/model_rdkit_baseline.pkl, models/model_hybrid.pkl
  reports/model_comparison.csv, reports/feature_importance_hybrid.csv,
  reports/subset_oof_preds.parquet, reports/baseline_test_preds.parquet
"""
import os
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(__file__)
D = lambda *p: os.path.join(HERE, "..", *p)

def main():
    rd = pd.read_parquet(D("data", "features_rdkit.parquet"))
    fx = pd.read_parquet(D("data", "features_xtb.parquet"))
    desc = [c for c in rd.columns if c.startswith("rdkit_")]
    fp   = [c for c in rd.columns if c.startswith("fp_")]
    xtb  = [c for c in fx.columns if c.startswith("xtb_")]

    # ---- baseline: RDKit desc+FP on full dataset ----
    X = rd[desc + fp].values; y = rd["redox_potential_V"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    base = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42, min_samples_leaf=2).fit(Xtr, ytr)
    pred = base.predict(Xte)
    bm = dict(MAE=mean_absolute_error(yte, pred), RMSE=float(np.sqrt(mean_squared_error(yte, pred))),
              R2=r2_score(yte, pred))
    print(f"baseline (full n={len(y)}): MAE={bm['MAE']:.4f} RMSE={bm['RMSE']:.4f} R2={bm['R2']:.4f}")
    joblib.dump({"model": base, "desc_cols": desc, "fp_cols": fp, "metrics": bm},
                D("models", "model_rdkit_baseline.pkl"), compress=("xz", 3))
    pd.DataFrame({"y_true": yte, "y_pred": pred}).to_parquet(D("reports", "baseline_test_preds.parquet"), index=False)

    # ---- hybrid comparison on the SAME xTB subset ----
    m = fx.merge(rd[["canonical_smiles"] + desc + fp], on="canonical_smiles", how="inner")
    ys = m["redox_potential_V"].values
    fsets = {"xTB only (7)": xtb, "RDKit desc (26)": desc, "RDKit desc + xTB (33)": desc + xtb,
             "RDKit desc+FP (1050)": desc + fp, "RDKit desc+FP + xTB": desc + fp + xtb}
    rkf = RepeatedKFold(n_splits=5, n_repeats=6, random_state=0)
    rows = {}
    for name, cols in fsets.items():
        cv = cross_validate(RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42, min_samples_leaf=2),
                            m[cols].values, ys, cv=rkf,
                            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"], n_jobs=-1)
        rows[name] = dict(MAE=-cv["test_neg_mean_absolute_error"].mean(),
                          MAE_std=cv["test_neg_mean_absolute_error"].std(),
                          RMSE=-cv["test_neg_root_mean_squared_error"].mean(),
                          R2=cv["test_r2"].mean(), R2_std=cv["test_r2"].std())
    comp = pd.DataFrame(rows).T[["MAE", "MAE_std", "RMSE", "R2", "R2_std"]].round(4)
    comp.to_csv(D("reports", "model_comparison.csv"))
    print(comp.to_string())

    # ---- final hybrid model: RDKit desc + xTB (33) ----
    hcols = desc + xtb
    hybrid = RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=42, min_samples_leaf=2).fit(m[hcols].values, ys)
    imp = pd.Series(hybrid.feature_importances_, index=hcols).sort_values(ascending=False)
    imp.to_csv(D("reports", "feature_importance_hybrid.csv"))
    oof_h = cross_val_predict(hybrid, m[hcols].values, ys, cv=KFold(5, shuffle=True, random_state=0), n_jobs=-1)
    oof_r = cross_val_predict(RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=42, min_samples_leaf=2),
                              m[desc].values, ys, cv=KFold(5, shuffle=True, random_state=0), n_jobs=-1)
    pd.DataFrame({"y_true": ys, "oof_hybrid": oof_h, "oof_rdkit": oof_r}).to_parquet(D("reports", "subset_oof_preds.parquet"), index=False)
    joblib.dump({"model": hybrid, "feature_cols": hcols, "desc_cols": desc, "xtb_cols": xtb,
                 "n_train": len(ys), "cv_comparison": comp.to_dict()},
                D("models", "model_hybrid.pkl"), compress=("xz", 3))
    print(f"hybrid trained on {len(ys)} molecules; xTB importance = {imp[xtb].sum():.2f}")

if __name__ == "__main__":
    main()
