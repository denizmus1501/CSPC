# mypy: disable-error-code="import-untyped"
import os
from ucimlrepo import fetch_ucirepo


def main():
    concrete = fetch_ucirepo(id=165)
    print(concrete.metadata)
    print(concrete.variables.to_string(index=False))
    X = concrete.data.features.copy()
    short_names = ["cement", "slag", "ash", "water", "superplastic",
                   "coarse_agg", "fine_agg", "age"]
    expected_order = ("cement", "blast furnace slag", "fly ash", "water",
                      "superplasticizer", "coarse aggregate", "fine aggregate", "age")
    for original, expected, short in zip(X.columns, expected_order, short_names, strict=True):
        if not original.lower().replace("_", " ").startswith(expected):
            raise ValueError(f"Unexpected column {original!r}; expected {expected!r}")
        print(f"{original} -> {short}")
    X.columns = short_names
    y = concrete.data.targets.iloc[:, 0].rename("strength")
    os.makedirs("data", exist_ok=True)
    X.assign(strength=y).to_csv("data/concrete.csv", index=False)


if __name__ == "__main__":
    main()
