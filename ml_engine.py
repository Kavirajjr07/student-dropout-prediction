from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN: Final = "Dropout"
TARGET_LABELS: Final = ("0", "1")
TARGET_DISPLAY: Final = {"0": "No dropout (0)", "1": "Dropout (1)"}
CATEGORICAL_FEATURES: Final = (
    "Internet_Access",
    "Part_Time_Job",
    "Semester",
    "Department",
)
NUMERIC_FEATURES: Final = (
    "Study_Hours_per_Day",
    "Attendance_Rate",
    "Assignment_Delay_Days",
    "Travel_Time_Minutes",
)
MODEL_FEATURES: Final = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RANDOM_STATE: Final = 42

EXCLUDED_FEATURES: Final = {
    "Student_ID": (
        "Unique identifier; it carries no intended student-risk signal and may "
        "encode row order."
    ),
    "GPA, Semester_GPA, and CGPA": (
        "Potentially post-outcome or outcome-proxy academic results. The file "
        "has no timestamps to establish they are available before the dropout "
        "label is determined."
    ),
    "Age and Gender": (
        "Demographic attributes excluded from the prediction features. Assess "
        "performance and fairness by appropriate groups before any operational use."
    ),
    "Family_Income, Scholarship, and Parental_Education": (
        "Sensitive socioeconomic/family or financial-support indicators "
        "excluded from the prediction features."
    ),
    "Stress_Index": (
        "A potentially sensitive wellbeing-related measure excluded from the "
        "prediction features."
    ),
    "Prediction horizon": (
        "The dataset has no event timestamps, dropout definition window, or "
        "cohort key. A pre-outcome monitoring checkpoint is assumed for this "
        "demonstration but cannot be verified from the file."
    ),
}


def _detect_delimiter(path: Path) -> str:
    with path.open(encoding="utf-8-sig", newline="") as source:
        sample = source.read(8192)
    if not sample.strip():
        raise ValueError("The dataset file is empty.")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0]
        delimiter_counts = {
            delimiter: first_line.count(delimiter) for delimiter in ",;\t|"
        }
        delimiter = max(delimiter_counts, key=delimiter_counts.get)
        if delimiter_counts[delimiter] == 0:
            raise ValueError("Could not determine the CSV delimiter.")
        return delimiter


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    delimiter = _detect_delimiter(path)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            raise ValueError("The dataset file is empty.")
        expected_width = len(header)
        malformed_rows = [
            reader.line_num for row in reader if len(row) != expected_width
        ]
    if malformed_rows:
        raise ValueError(
            f"The CSV contains {len(malformed_rows)} records with an unexpected "
            f"number of fields; first malformed line: {malformed_rows[0]}."
        )

    raw = pd.read_csv(
        path,
        sep=delimiter,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    raw.columns = raw.columns.astype(str).str.strip()
    if raw.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names after trimming.")

    required = set(MODEL_FEATURES) | {TARGET_COLUMN}
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(
            "Required dataset columns are missing: " + ", ".join(missing_columns)
        )

    data = raw.copy()
    for column in data.columns:
        if data[column].dtype == object:
            data[column] = data[column].str.strip().replace("", pd.NA)

    raw_target = data[TARGET_COLUMN]
    missing_target_rows = int(raw_target.isna().sum())
    parsed_target = pd.to_numeric(raw_target, errors="coerce")
    invalid_target_mask = raw_target.notna() & parsed_target.isna()
    if invalid_target_mask.any():
        invalid_labels = sorted(raw_target[invalid_target_mask].astype(str).unique())
        raise ValueError("Unexpected target labels: " + ", ".join(invalid_labels))

    numeric_target = parsed_target.dropna()
    if not numeric_target.isin((0, 1)).all():
        invalid_labels = sorted(numeric_target.unique().tolist())
        raise ValueError(
            "Dropout must be a binary 0/1 target; found values: "
            + ", ".join(map(str, invalid_labels))
        )
    data[TARGET_COLUMN] = parsed_target.map(
        lambda value: str(int(value)) if pd.notna(value) else pd.NA
    ).astype("string")

    invalid_numeric_cells: dict[str, int] = {}
    for column in data.columns.drop(TARGET_COLUMN):
        source = data[column]
        converted = pd.to_numeric(source, errors="coerce")
        numeric_values = source.notna() & converted.notna()
        nonnumeric_values = source.notna() & converted.isna()
        if not nonnumeric_values.any():
            data[column] = converted
        elif column in MODEL_FEATURES and column not in CATEGORICAL_FEATURES:
            invalid_numeric_cells[column] = int(nonnumeric_values.sum())
            data[column] = converted

    missing_feature_counts = data.drop(columns=[TARGET_COLUMN]).isna().sum()
    literal_none_counts = {
        str(column): int(data[column].astype("string").eq("None").sum())
        for column in data.columns.drop(TARGET_COLUMN)
        if data[column].astype("string").eq("None").any()
    }
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    if data.empty:
        raise ValueError("No rows with a valid target label are available.")

    class_counts = data[TARGET_COLUMN].value_counts().reindex(
        TARGET_LABELS, fill_value=0
    )
    if (class_counts == 0).any():
        raise ValueError(
            "Training requires both target classes; observed counts: "
            + ", ".join(
                f"{TARGET_DISPLAY[label]}={int(class_counts[label])}"
                for label in TARGET_LABELS
            )
        )
    quality: dict[str, object] = {
        "delimiter": delimiter,
        "rows_in_file": len(raw),
        "columns_in_file": len(raw.columns),
        "malformed_rows": len(malformed_rows),
        "missing_target_rows": missing_target_rows,
        "missing_features_by_column": {
            str(name): int(count)
            for name, count in missing_feature_counts[
                missing_feature_counts.gt(0)
            ].items()
        },
        "invalid_numeric_cells_by_column": invalid_numeric_cells,
        "literal_none_counts": literal_none_counts,
        "duplicate_rows": int(raw.duplicated().sum()),
        "class_counts": class_counts,
        "column_types": data.dtypes.astype(str).value_counts().to_dict(),
    }
    return data, quality


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def build_model(classifier: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def evaluate_models(
    data: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.Series]:
    features = data.loc[:, list(MODEL_FEATURES)]
    target = data[TARGET_COLUMN].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    candidates: dict[str, Pipeline] = {
        "Majority-class baseline": build_model(
            DummyClassifier(strategy="prior")
        ),
        "Logistic regression": build_model(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Extra Trees": build_model(
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }

    results: dict[str, object] = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        results[name] = {
            "model": model,
            "predictions": predicted,
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro")),
            "weighted_f1": float(f1_score(y_test, predicted, average="weighted")),
            "macro_precision": float(
                precision_score(y_test, predicted, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_test, predicted, average="macro", zero_division=0)
            ),
            "class_report": classification_report(
                y_test,
                predicted,
                labels=list(TARGET_LABELS),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predicted, labels=list(TARGET_LABELS)
            ),
        }

    selected_name = max(
        ("Logistic regression", "Extra Trees"),
        key=lambda name: results[name]["macro_f1"],
    )
    results["selected_model_name"] = selected_name
    results["test_rows"] = len(y_test)
    results["train_rows"] = len(y_train)
    results["test_labels"] = y_test
    return results, x_test, y_test


def prepare_scoring_data(uploaded: pd.DataFrame) -> pd.DataFrame:
    uploaded = uploaded.copy()
    uploaded.columns = uploaded.columns.astype(str).str.strip()
    missing = sorted(set(MODEL_FEATURES) - set(uploaded.columns))
    if missing:
        raise ValueError("Missing model feature columns: " + ", ".join(missing))

    scoring = uploaded.loc[:, list(MODEL_FEATURES)].copy()
    for column in NUMERIC_FEATURES:
        source = scoring[column]
        converted = pd.to_numeric(source, errors="coerce")
        if (source.notna() & converted.isna()).any():
            raise ValueError(f"Feature {column!r} contains non-numeric values.")
        scoring[column] = converted
    for column in CATEGORICAL_FEATURES:
        scoring[column] = scoring[column].astype("string").str.strip().replace("", pd.NA)
    if scoring.empty:
        raise ValueError("The uploaded file contains no records to score.")
    return scoring
