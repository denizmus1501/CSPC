# mypy: disable-error-code="import-untyped"
"""PW4 Session B: PCA and K-Means on wine chemistry."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results")
CHOSEN_K = 3


def report(name, table):
    print(f"\n{name}\n{table.to_string(index=False)}")
    table.to_csv(RESULTS / f"B_{name}.csv", index=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / f"B_{name}.png", dpi=160)
    plt.close(fig)


def cluster(data, k, n_init=10, seed=0):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, max_iter=300,
                  random_state=seed).fit(data)


def scatter(scores, variance, labels=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if labels is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=24, alpha=0.8)
    else:
        for label in np.unique(labels):
            points = scores[labels == label]
            ax.scatter(points[:, 0], points[:, 1], s=24, alpha=0.8, label=f"Cluster {label}")
        ax.legend()
    ax.set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})")
    save(fig, "pca_plain" if labels is None else "pca_clusters")


def main():
    RESULTS.mkdir(exist_ok=True)
    wine = load_wine(as_frame=True)
    X = wine.data
    print(wine.DESCR)
    print(X.describe().to_string())
    print(pd.DataFrame({"Range": X.max() - X.min(), "SD": X.std()}).sort_values("Range"))

    raw = PCA().fit(X)
    print("Raw PC1 explained fraction:", raw.explained_variance_ratio_[0])
    raw_loading = pd.Series(raw.components_[0], index=X.columns, name="Raw PC1")
    report("raw_pc1", raw_loading.sort_values(ascending=False).rename_axis("Measurement").reset_index())

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    print("Largest absolute scaled column mean:", np.max(np.abs(Z.mean(axis=0))))
    print("Scaled column population SD:", Z.std(axis=0))
    pca = PCA().fit(Z)
    scores = pca.transform(Z)
    variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance)
    components = np.arange(1, len(variance) + 1)
    report("variance", pd.DataFrame({"PC": components, "Explained": variance, "Cumulative": cumulative}))
    print("Components for 90%:", np.searchsorted(cumulative, 0.9) + 1)
    print("PC1 + PC2:", cumulative[1])
    loadings = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    ranked = pd.concat([loadings[pc].sort_values(ascending=False).rename_axis("Measurement")
                        .reset_index().assign(Component=pc).rename(columns={pc: "Loading"})
                        for pc in ["PC1", "PC2"]], ignore_index=True)
    report("loadings", ranked[["Component", "Measurement", "Loading"]])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(components, variance, label="Individual")
    ax.plot(components, cumulative, "o-", label="Cumulative")
    ax.axhline(0.9, linestyle="--", color="black", label="90%")
    ax.set(xlabel="Principal component", ylabel="Explained variance fraction", xticks=components)
    ax.legend()
    save(fig, "scree")
    scatter(scores, variance)

    report("initialization", pd.DataFrame(
        [{"Seed": seed, "Inertia": cluster(Z, 3, n_init=1, seed=seed).inertia_} for seed in range(8)]))
    sweep = [cluster(Z, k) for k in range(1, 11)]
    inertia = pd.DataFrame({"k": range(1, 11), "Inertia": [model.inertia_ for model in sweep]})
    report("inertia", inertia)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(inertia["k"], inertia["Inertia"], "o-")
    ax.set(xlabel="Number of clusters k", ylabel="Inertia", xticks=range(1, 11))
    save(fig, "elbow")

    chosen = sweep[CHOSEN_K - 1]
    centers = pd.DataFrame(scaler.inverse_transform(chosen.cluster_centers_), columns=X.columns)
    centers.insert(0, "Size", np.bincount(chosen.labels_, minlength=CHOSEN_K))
    centers.insert(0, "Cluster", np.arange(CHOSEN_K))
    report("centers", centers)
    scatter(scores, variance, chosen.labels_)

    # B6: reveal the labels only after fitting all three unsupervised models.
    comparisons = [("scaled", sweep[2]), ("raw", cluster(X, 3)), ("two_PCs", cluster(scores[:, :2], 3))]
    cultivar = wine.target
    rows = []
    for label, model in comparisons:
        table = pd.crosstab(cultivar.rename("Cultivar"), model.labels_, colnames=["Cluster"])
        agreement = table.max(axis=0).sum() / len(cultivar)
        report(f"crosstab_{label}", table.reset_index())
        rows.append({"Input": label, "Agreement": agreement})
    report("agreement", pd.DataFrame(rows))

    train, test, y_train, y_test = train_test_split(
        X, cultivar, test_size=0.25, random_state=42, stratify=cultivar)
    models = [
        ("All 13 scaled features", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
        ("First two scaled PCs", make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5))),
    ]
    report("supervised_bonus", pd.DataFrame(
        [{"Features": label, "Test accuracy": model.fit(train, y_train).score(test, y_test)}
         for label, model in models]))


if __name__ == "__main__":
    main()
