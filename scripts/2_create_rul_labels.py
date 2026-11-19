"""Create RUL labels from the cleaned C-MAPSS CSV files."""

from pathlib import Path
import pandas as pd


# INPUT AND OUTPUT FILES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

input_files = {
    subset: {
        "train": PROJECT_ROOT / "data" / "cleaned" / f"cleaned_train_{subset}.csv",
        "test": PROJECT_ROOT / "data" / "cleaned" / f"cleaned_test_{subset}.csv",
        "rul": PROJECT_ROOT / "data" / "cleaned" / f"cleaned_rul_{subset}.csv",
    }
    for subset in SUBSETS
}

output_files = {
    subset: {
        "train": PROJECT_ROOT / "data" / "labeled" / f"labeled_train_{subset}.csv",
        "test": PROJECT_ROOT / "data" / "labeled" / f"labeled_test_{subset}.csv",
        "report": PROJECT_ROOT / "data" / "labeled" / f"label_report_{subset}.csv",
    }
    for subset in SUBSETS
}


# CREATE RUL LABELS FOR EACH SUBSET
for subset in SUBSETS:
    print(f"\nCreating RUL labels for {subset}...")

    paths_in = input_files[subset]
    paths_out = output_files[subset]
    paths_out["train"].parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(paths_in["train"])
    test = pd.read_csv(paths_in["test"])
    official_rul = pd.read_csv(paths_in["rul"])

    # Training trajectories reach failure, so the final cycle is RUL zero
    final_train_cycle = train.groupby("engine_key")["cycle"].transform("max")
    train["RUL"] = final_train_cycle - train["cycle"]

    # Test trajectories stop before failure. The official RUL file gives the
    # remaining cycles after the final observed cycle of each test engine
    last_test_cycle = test.groupby("unit_id")["cycle"].max().sort_index()
    official_rul = official_rul.sort_values("test_unit_id")

    if len(last_test_cycle) != len(official_rul):
        raise ValueError(f"{subset}: test-engine count and RUL count differ.")

    official_rul_by_unit = pd.Series(official_rul["official_RUL"].to_numpy(),index=last_test_cycle.index,)

    failure_cycle = last_test_cycle + official_rul_by_unit
    test["RUL"] = test["unit_id"].map(failure_cycle) - test["cycle"]

    if train["RUL"].min() != 0:
        raise ValueError("Training RUL should reach zero at each engine's final cycle.")

    if train["RUL"].isna().any() or test["RUL"].isna().any():
        raise ValueError("RUL labels contain missing values.")

    train.to_csv(paths_out["train"], index=False)
    test.to_csv(paths_out["test"], index=False)

    report = pd.DataFrame([
        {
            "subset": subset,
            "file_type": "train",
            "rows": len(train),
            "engines": train["engine_key"].nunique(),
            "minimum_RUL": train["RUL"].min(),
            "maximum_RUL": train["RUL"].max(),
        },
        {
            "subset": subset,
            "file_type": "test",
            "rows": len(test),
            "engines": test["engine_key"].nunique(),
            "minimum_RUL": test["RUL"].min(),
            "maximum_RUL": test["RUL"].max(),
        },
    ])
    report.to_csv(paths_out["report"], index=False)

    print(f"Training output: {paths_out['train']}")
    print(f"Test output: {paths_out['test']}")
