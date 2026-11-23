from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import BASE, FEATS, load_csv, make_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "NIFTYMS400_MQ100_5minute.csv"
MODELS_DIR = ROOT / "models"


def new_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5,
                                               min_samples_leaf=50, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                           max_iter=150, random_state=42),
    }


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Add the full historical NIFTYMS400_MQ100_5minute.csv to the project's data folder. "
            "The small Nifty 50 demo sample is not intended for training."
        )
    data = make_features(load_csv(DATA_PATH)).dropna(subset=BASE + ["fut_rv"]).copy()
    if len(data) < 35:
        raise ValueError(
            f"Only {len(data)} usable rows were found; at least 35 are needed "
            "for five-fold validation with a six-row gap."
        )
    print("Rows for modelling:", len(data))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Walk-forward validation ----
    rows = []
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5, gap=6).split(data), 1):
        train, test = data.iloc[tr].copy(), data.iloc[te].copy()
        seas = train.groupby("time")["fut_rv"].mean()
        train["seasonal"] = train["time"].map(seas)
        test["seasonal"] = test["time"].map(seas).fillna(train["fut_rv"].mean())
        y = test["fut_rv"]
        row = {"Fold": fold,
               "Naive (last 30 min)": mean_absolute_error(y, test["rv6"]),
               "Seasonal (time of day)": mean_absolute_error(y, test["seasonal"])}
        for name, mdl in new_models().items():
            mdl.fit(train[FEATS], train["fut_rv"])
            row[name] = mean_absolute_error(y, mdl.predict(test[FEATS]))
        rows.append(row)

    results = pd.DataFrame(rows).set_index("Fold")
    results.to_csv(MODELS_DIR / "walk_forward_results.csv")
    print(results.round(4))
    print("\nAverage error:\n", results.mean().round(4))

    # ---- Final model on all data ----
    seasonal = data.groupby("time")["fut_rv"].mean()
    data["seasonal"] = data["time"].map(seasonal)
    final = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(data[FEATS], data["fut_rv"])
    data["pred"] = final.predict(data[FEATS])

    bundle = {
        "model": final,
        "seasonal": seasonal,
        "fallback": float(data["fut_rv"].mean()),
        "all_pred": data["pred"].values,
        "pred_by_time": {t: g.values for t, g in data.groupby("time")["pred"]},
        "coef": pd.Series(final[-1].coef_, index=FEATS),
        "trained_from": str(data.index.min()),
        "trained_to": str(data.index.max()),
        "n_rows": len(data),
    }
    joblib.dump(bundle, MODELS_DIR / "risk_model.joblib")
    print(f"\nSaved {MODELS_DIR / 'risk_model.joblib'}")


if __name__ == "__main__":
    main()
