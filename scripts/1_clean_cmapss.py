# Clean the raw NASA C-MAPSS text files and save CSV files.

from pathlib import Path
import pandas as pd


# INPUT AND OUTPUT FILES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

input_files = {
    subset: {
        "train": PROJECT_ROOT / "data" / "raw" / f"train_{subset}.txt",
        "test": PROJECT_ROOT / "data" / "raw" / f"test_{subset}.txt",
        "rul": PROJECT_ROOT / "data" / "raw" / f"RUL_{subset}.txt",
    }
    for subset in SUBSETS
}

output_files = {
    subset: {
        "train": PROJECT_ROOT / "data" / "cleaned" / f"cleaned_train_{subset}.csv",
        "test": PROJECT_ROOT / "data" / "cleaned" / f"cleaned_test_{subset}.csv",
        "rul": PROJECT_ROOT / "data" / "cleaned" / f"cleaned_rul_{subset}.csv",
        "report": PROJECT_ROOT / "data" / "cleaned" / f"cleaning_report_{subset}.csv",
    }
    for subset in SUBSETS
}



# COLUMN NAMES
COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + ["op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# FUNCTIONS
def read_measurement_file(input_file):
    # Read a raw train or test file and assign column names
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    df = pd.read_csv(
        input_file,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    if df.shape[1] != 26:
        raise ValueError(f"{input_file} should have 26 columns, but {df.shape[1]} were read.")

    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df


def clean_measurement_data(df, subset):
    # Add identifiers and sort rows without changing measurement
    df = df.copy()
    df["subset"] = subset
    df["engine_key"] = subset + "_" + df["unit_id"].astype(str)
    return df.sort_values(["engine_key", "cycle"]).reset_index(drop=True)


def create_quality_report(df, file_type):
    # Create a one-row CSV summary of basic quality checks
    ordered = df.sort_values(["engine_key", "cycle"])
    cycle_difference = ordered.groupby("engine_key")["cycle"].diff().dropna()

    return {
        "file_type": file_type,
        "rows": len(df),
        "columns": len(df.columns),
        "engines": df["engine_key"].nunique(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_engine_cycle_rows": int(
            df.duplicated(subset=["engine_key", "cycle"]).sum()
        ),
        "non_positive_cycle_differences": int((cycle_difference <= 0).sum()),
        "cycle_gaps": int((cycle_difference > 1).sum())
    }


# CLEAN EACH SUBSET
for subset in SUBSETS:
    print(f"\nCleaning {subset}...")

    paths_in = input_files[subset]
    paths_out = output_files[subset]
    paths_out["train"].parent.mkdir(parents=True, exist_ok=True)

    train = clean_measurement_data(read_measurement_file(paths_in["train"]), subset)
    test = clean_measurement_data(read_measurement_file(paths_in["test"]), subset)

    if not paths_in["rul"].exists():
        raise FileNotFoundError(f"File not found: {paths_in['rul']}")

    official_rul = pd.read_csv(
        paths_in["rul"],
        header=None,
        names=["official_RUL"],
    )
    official_rul["test_unit_id"] = range(1, len(official_rul) + 1)
    official_rul["subset"] = subset

    train.to_csv(paths_out["train"], index=False)
    test.to_csv(paths_out["test"], index=False)
    official_rul.to_csv(paths_out["rul"], index=False)

    report = pd.DataFrame([
        create_quality_report(train, "train"),
        create_quality_report(test, "test"),
    ])
    report.insert(0, "subset", subset)
    report.to_csv(paths_out["report"], index=False)

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Saved cleaned files to: {paths_out['train'].parent}")
