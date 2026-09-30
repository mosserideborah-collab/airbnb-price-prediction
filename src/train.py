"""Cross-validation of the models, final training and test predictions.

Run from the repository root:
    python src/train.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from features import add_basic_features, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "results"

models = {
    "Baseline (mean)": make_pipeline(make_preprocessor("tree"), DummyRegressor()),
    "Ridge": make_pipeline(make_preprocessor("linear"), Ridge(alpha=3.0)),
    "LightGBM": make_pipeline(
        make_preprocessor("tree"),
        lgb.LGBMRegressor(n_estimators=2500, learning_rate=0.02, num_leaves=48, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, random_state=42, verbose=-1),
    ),
}


def plot_results(scores, y, pred_lgbm, final_model):
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. R² of each model, compared with the first version of the project
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = list(scores)
    r2 = [max(scores[n]["r2"], 0) for n in names]
    bars = ax.barh(names, r2, color=["#9aa5b1", "#9aa5b1", "#2f6fdf"])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in r2], padding=3)
    ax.axvline(0.532, color="#d9534f", ls="--", label="first version (R² = 0.532)")
    ax.set(xlim=(0, 1), xlabel="R² (5-fold cross-validation)", title="Model comparison")
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)

    # 2. predicted vs actual
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(y, pred_lgbm, gridsize=60, cmap="Blues", mincnt=1, bins="log")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    ax.set(xlabel="actual log_price", ylabel="predicted log_price",
           title=f"LightGBM (R² = {scores['LightGBM']['r2']:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES / "pred_vs_actual.png", dpi=150)

    # 3. feature importance of the final LightGBM
    names = pd.Index(final_model[0].get_feature_names_out())
    names = names.str.replace(r"^[a-z]+__", "", regex=True).str.replace("truncatedsvd", "text_topic_")
    importance = pd.Series(final_model[-1].booster_.feature_importance("gain"), index=names)
    importance = (importance / importance.sum()).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importance.index, importance.values, color="#2f6fdf")
    ax.set(xlabel="share of total gain", title="LightGBM: top 20 features")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "airbnb_train.csv")
    test = pd.read_csv(ROOT / "data" / "airbnb_test.csv").rename(columns={"Unnamed: 0": "id"})  # id column has no name
    y = train.pop("log_price")
    X, X_test = add_basic_features(train), add_basic_features(test)

    # each listing is predicted by a model trained on the other 4 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions, scores = {}, {}
    for name, model in models.items():
        predictions[name] = cross_val_predict(model, X, y, cv=kfold)
        scores[name] = {"rmse": root_mean_squared_error(y, predictions[name]),
                        "mae": mean_absolute_error(y, predictions[name]),
                        "r2": r2_score(y, predictions[name])}
        print(f"{name:<16} RMSE={scores[name]['rmse']:.4f}  R²={scores[name]['r2']:.4f}")

    pd.DataFrame({"id": train["id"], "city": train["city"], "room_type": train["room_type"],
                  "log_price": y, "pred": predictions["LightGBM"]}
                 ).to_csv(RESULTS / "cv_predictions.csv", index=False, float_format="%.5f")
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(scores, f, indent=2)

    # final model trained on all the training data
    final_model = models["LightGBM"].fit(X, y)
    plot_results(scores, y, predictions["LightGBM"], final_model)
    pd.DataFrame({"": test["id"], "logpred": final_model.predict(X_test)}).to_csv(
        RESULTS / "predictions.csv", index=False)
    print("predictions saved in results/predictions.csv")


if __name__ == "__main__":
    main()
