"""Create engine splits and sensor-history features for all subsets."""

from pathlib import Path
import numpy as np
import pandas as pd


# INPUT AND OUTPUT FILES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
RANDOM_STATE = 42
HISTORY_WINDOW = 30

input_files = {
    subset: {
        "train": PROJECT_ROOT / "data" / "labeled" / f"labeled_train_{subset}.csv",
        "test": PROJECT_ROOT / "data" / "labeled" / f"labeled_test_{subset}.csv",
    }
    for subset in SUBSETS
}

output_files = {
    subset: {
        "train": PROJECT_ROOT / "data" / "engineered" / f"engineered_train_{subset}.csv",
        "test": PROJECT_ROOT / "data" / "engineered" / f"engineered_test_{subset}.csv",
        "report": PROJECT_ROOT / "data" / "engineered" / f"feature_report_{subset}.csv",
    }
    for subset in SUBSETS
}


# COLUMN GROUPS
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2","op_setting_3"]


# FUNCTIONS
def split_by_engine(train):
    # Split complete engines into development and validation groups
    engine_keys = train["engine_key"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = rng.permutation(engine_keys)

    validation_count = max(1, int(round(len(shuffled) * 0.20)))
    validation_keys = sorted(shuffled[:validation_count].tolist())
    development_keys = sorted(shuffled[validation_count:].tolist())
    return development_keys, validation_keys


def select_sensors(train, development_keys):
    # Remove constant sensors using development data only
    development = train[train["engine_key"].isin(development_keys)]
    sensor_variance = development[SENSOR_COLS].var()

    selected_sensors = sensor_variance[sensor_variance > 1e-8].index.tolist()
    removed_sensors = [sensor for sensor in SENSOR_COLS if sensor not in selected_sensors]
    return selected_sensors, removed_sensors


def add_history_features(df, selected_sensors):
    #Create past-only rolling and change features within each engine
    df = df.sort_values(["engine_key", "cycle"]).copy()

    for sensor in selected_sensors:
        grouped_sensor = df.groupby("engine_key")[sensor]

        # Average of the current cycle and previous 29 cycles
        df[f"{sensor}_rolling_mean_{HISTORY_WINDOW}"] = grouped_sensor.transform(
            lambda values: values.rolling(
                window=HISTORY_WINDOW,
                min_periods=1,
            ).mean()
        )

        # Recent sensor variability
        df[f"{sensor}_rolling_std_{HISTORY_WINDOW}"] = grouped_sensor.transform(
            lambda values: values.rolling(window=HISTORY_WINDOW, min_periods=2).std()).fillna(0.0)

        # Change from the previous cycle.
        df[f"{sensor}_change"] = grouped_sensor.diff().fillna(0.0)

    return df


def mark_validation_endpoints(train, validation_keys):
    #Mark one reproducible validation cycle per validation engine
    train = train.copy()
    train["validation_endpoint"] = False
    rng = np.random.default_rng(RANDOM_STATE)

    validation_rows = train[train["engine_key"].isin(validation_keys)]

    for _, engine_data in validation_rows.groupby("engine_key"):
        selected_index = rng.choice(engine_data.index.to_numpy())
        train.loc[selected_index, "validation_endpoint"] = True

    return train


def build_feature_report(subset, selected_sensors, removed_sensors):
    #Create a CSV documenting which features were selected
    model_features = ["cycle"] + OP_SETTING_COLS + selected_sensors

    for sensor in selected_sensors:
        model_features.extend([
            f"{sensor}_rolling_mean_{HISTORY_WINDOW}",
            f"{sensor}_rolling_std_{HISTORY_WINDOW}",
            f"{sensor}_change"
        ])

    rows = []

    for feature in model_features:
        if feature == "cycle":
            feature_group = "cycle"
        elif feature in OP_SETTING_COLS:
            feature_group = "operational_setting"
        elif feature in selected_sensors:
            feature_group = "current_sensor"
        else:
            feature_group = "history_engineered"

        rows.append({
            "subset": subset,
            "feature": feature,
            "feature_group": feature_group,
            "selected": True
        })

    for sensor in removed_sensors:
        rows.append({
            "subset": subset,
            "feature": sensor,
            "feature_group": "sensor",
            "selected": False,
        })

    return pd.DataFrame(rows)


# FEATURE ENGINEER EACH SUBSET
for subset in SUBSETS:
    print(f"\nFeature engineering {subset}...")

    paths_in = input_files[subset]
    paths_out = output_files[subset]
    paths_out["train"].parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(paths_in["train"])
    test = pd.read_csv(paths_in["test"])

    development_keys, validation_keys = split_by_engine(train)
    selected_sensors, removed_sensors = select_sensors(train, development_keys)

    # The same feature definitions are applied to train and test data.
    train = add_history_features(train, selected_sensors)
    test = add_history_features(test, selected_sensors)

    train["split"] = np.where(
        train["engine_key"].isin(development_keys),
        "development",
        "validation"
    )
    train = mark_validation_endpoints(train, validation_keys)

    test["split"] = "official_test"
    test["validation_endpoint"] = False

    train.to_csv(paths_out["train"], index=False)
    test.to_csv(paths_out["test"], index=False)

    feature_report = build_feature_report(
        subset,
        selected_sensors,
        removed_sensors
    )
    feature_report.to_csv(paths_out["report"], index=False)

    print(f"Selected sensors: {len(selected_sensors)}")
    print(f"Removed sensors: {len(removed_sensors)}")
    print(f"Development engines: {len(development_keys)}")
    print(f"Validation engines: {len(validation_keys)}")
    print(f"Saved engineered files to: {paths_out['train'].parent}")