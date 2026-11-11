# mypy: disable-error-code="import-untyped"
"""PW4 Session A: concrete regression, classification and feature importance."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

RESULTS = Path("results")
K_VALUES = [1, 3, 5, 7, 15, 25, 51, 101]
DEPTHS = [1, 2, 3, 4, 5, 6, 8, None]
CHOSEN_K = 5
CHOSEN_DEPTH = 5


def report(name, table):
    print(f"\n{name}\n{table.to_string(index=False)}")
    table.to_csv(RESULTS / f"A_{name}.csv", index=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / f"A_{name}.png", dpi=160)
    plt.close(fig)


def regression_metrics(model, X_train, X_test, y_train, y_test, p):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, train_pred)
    n = len(y_train)
    adjusted = 1 - (1 - train_r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    return {"p": p, "Train RMSE": np.sqrt(mean_squared_error(y_train, train_pred)),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "Test MAE": mean_absolute_error(y_test, test_pred), "Train R2": train_r2,
            "Train adjusted R2": adjusted, "Test R2": r2_score(y_test, test_pred)}


def classification_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred, zero_division=0)}


def scaled_knn(k):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Regression split: {len(X_train)} train, {len(X_test)} test")
    linear = LinearRegression().fit(X_train, y_train)
    print("Linear intercept:", linear.intercept_)
    report("linear", pd.DataFrame([regression_metrics(
        linear, X_train, X_test, y_train, y_test, X.shape[1])]))

    residuals = y_test - linear.predict(X_test)
    print("Mean residual by age band (days):")
    print(residuals.groupby(pd.cut(X_test["age"], [0, 7, 27, 28, 90, 365]),
                            observed=True).agg(["size", "mean"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(X_test["age"], residuals, alpha=0.55)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xscale="log", xlabel="Age (days, log scale)", ylabel="True − predicted strength (MPa)")
    save(fig, "residuals")

    rows = []
    for degree in (1, 2, 3, 4):
        poly = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                             StandardScaler(), LinearRegression())
        poly.fit(X_train, y_train)
        rows.append({"Degree": degree, **regression_metrics(
            poly, X_train, X_test, y_train, y_test, poly[0].n_output_features_)})
    report("polynomial", pd.DataFrame(rows).drop(columns="Test MAE"))

    physics = X.assign(age=np.log(X["age"]))
    binder = X["cement"] + X["slag"] + X["ash"]
    rows = []
    for label, inputs in [("log(age)", physics),
                          ("log(age) and water/binder", physics.assign(water=X["water"] / binder))]:
        train, test = inputs.loc[X_train.index], inputs.loc[X_test.index]
        model = LinearRegression().fit(train, y_train)
        rows.append({"Features": label, "p": train.shape[1], "Test R2": model.score(test, y_test)})
    report("physics_bonus", pd.DataFrame(rows))
    standardized = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    return linear.coef_, standardized[-1].coef_


def classification(X, y, raw_coef, standardized_coef):
    c = (y >= 40).astype(int)
    train, test, c_train, c_test = train_test_split(
        X, c, test_size=0.2, random_state=42, stratify=c)
    print("Class proportions (all, train, test):")
    print(pd.DataFrame({label: values.value_counts(normalize=True).sort_index()
                        for label, values in [("All", c), ("Train", c_train), ("Test", c_test)]}))

    knn_rows = []
    for k in K_VALUES:
        model = scaled_knn(k).fit(train, c_train)
        knn_rows.append({"K": k, "Train accuracy": model.score(train, c_train),
                         "Test accuracy": model.score(test, c_test)})
    knn_sweep = pd.DataFrame(knn_rows)
    report("knn", knn_sweep)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_VALUES, knn_sweep["Train accuracy"], "o-", label="Train")
    ax.plot(K_VALUES, knn_sweep["Test accuracy"], "o-", label="Test")
    ax.set(xscale="log", xlabel="K (log scale)", ylabel="Accuracy")
    ax.set_xticks(K_VALUES, labels=K_VALUES)
    ax.legend()
    save(fig, "knn")

    knn = scaled_knn(CHOSEN_K).fit(train, c_train)
    unscaled = KNeighborsClassifier(n_neighbors=CHOSEN_K).fit(train, c_train)
    print("KNN accuracy (scaled, raw):", knn.score(test, c_test), unscaled.score(test, c_test))
    split_rows = []
    for seed in range(5):
        sx, tx, sy, ty = train_test_split(X, c, test_size=0.2, random_state=seed, stratify=c)
        split_rows.append({"Seed": seed, "Test accuracy": scaled_knn(CHOSEN_K).fit(sx, sy).score(tx, ty)})
    split_table = pd.DataFrame(split_rows)
    report("knn_splits", split_table)
    print("Split accuracy range width:", split_table["Test accuracy"].max() - split_table["Test accuracy"].min())

    tree_rows = []
    for depth in DEPTHS:
        model = DecisionTreeClassifier(criterion="entropy", max_depth=depth,
                                       min_samples_leaf=1, random_state=0).fit(train, c_train)
        if depth == CHOSEN_DEPTH:
            tree = model
        tree_rows.append({"Max depth": str(depth), "Actual depth": model.get_depth(),
                          "Leaves": model.get_n_leaves(), "Train accuracy": model.score(train, c_train),
                          "Test accuracy": model.score(test, c_test)})
    tree_sweep = pd.DataFrame(tree_rows)
    report("tree", tree_sweep)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tree_sweep["Max depth"], tree_sweep["Train accuracy"], "o-", label="Train")
    ax.plot(tree_sweep["Max depth"], tree_sweep["Test accuracy"], "o-", label="Test")
    ax.set(xlabel="Maximum depth", ylabel="Accuracy")
    ax.legend()
    save(fig, "tree")
    top = export_text(tree, feature_names=list(X.columns), max_depth=1)
    print(top)
    (RESULTS / "A_tree_top.txt").write_text(top, encoding="utf-8")

    models = [(f"KNN (K={CHOSEN_K})", knn), (f"Tree (depth={CHOSEN_DEPTH})", tree)]
    report("metrics", pd.DataFrame([{"Model": name, **classification_metrics(c_test, model.predict(test))}
                                    for name, model in models]))
    proba = knn.predict_proba(test)[:, 1]
    report("thresholds", pd.DataFrame([{"Threshold": t, **classification_metrics(c_test, (proba >= t).astype(int))}
                                       for t in (0.3, 0.5, 0.7, 0.9)]))

    knn_perm = permutation_importance(knn, test, c_test, scoring="f1", n_repeats=20, random_state=0)
    tree_perm = permutation_importance(tree, test, c_test, scoring="f1", n_repeats=20, random_state=0)
    importance = pd.DataFrame({"Feature": X.columns, "Raw coefficient": raw_coef,
                               "Standardized coefficient": standardized_coef,
                               "Tree entropy": tree.feature_importances_,
                               "KNN permutation mean": knn_perm.importances_mean,
                               "KNN permutation SD": knn_perm.importances_std,
                               "Tree permutation mean": tree_perm.importances_mean,
                               "Tree permutation SD": tree_perm.importances_std})
    report("importance", importance)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, column, label in zip(
        axes, ["Standardized coefficient", "Tree entropy", "KNN permutation mean"],
        ["Linear coefficient (MPa per SD)", "Tree entropy importance", "KNN permutation (F1 drop)"],
        strict=True,
    ):
        ax.barh(X.columns, importance[column])
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel(label)
    save(fig, "importance")


def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv("data/concrete.csv")
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing per column:\n", df.isna().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(df.describe().to_string())
    X, y = df.drop(columns="strength"), df["strength"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, feature, unit in zip(axes, ("cement", "age"), ("kg/m³", "days"), strict=True):
        ax.scatter(X[feature], y, s=12, alpha=0.4)
        ax.set_xlabel(f"{feature.title()} ({unit})")
    axes[0].set_ylabel("Strength (MPa)")
    save(fig, "strength")
    classification(X, y, *regression(X, y))


if __name__ == "__main__":
    main()
