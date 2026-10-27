from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

from ml_engine import (
    EXCLUDED_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    TARGET_DISPLAY,
    TARGET_LABELS,
    evaluate_models,
    load_dataset,
    prepare_scoring_data,
)
from support_engine import (
    ATTENDANCE_CONCERNS,
    add_support_event,
    attendance_policy_flag,
    get_support_events,
    initialize_store,
    suggested_support,
)


DATA_PATH: Final = Path(__file__).with_name("student_dropout_dataset_v3.csv")
SUPPORT_DB_PATH: Final = Path(__file__).with_name("student_support.sqlite3")
INTERVENTION_OPTIONS: Final = (
    "No intervention recorded",
    "Advising appointment offered",
    "Counselling referral offered",
    "Financial-support referral offered",
    "Learning-support referral offered",
    "Attendance follow-up agreed",
    "Other support discussed",
)
STATUS_OPTIONS: Final = ("Open", "In progress", "Follow-up scheduled", "Closed")
DATASET_CACHE_VERSION: Final = "student_dropout_dataset_v3_schema_2"


@st.cache_data
def get_dataset(cache_version: str) -> tuple[pd.DataFrame, dict[str, object]]:
    if cache_version != DATASET_CACHE_VERSION:
        raise ValueError(f"Unexpected dataset cache version: {cache_version}.")
    return load_dataset(DATA_PATH)


@st.cache_resource
def get_model_evaluation(data: pd.DataFrame) -> dict[str, object]:
    result, _, _ = evaluate_models(data)
    return result


def render_overview(data: pd.DataFrame, quality: dict[str, object]) -> None:
    counts = quality["class_counts"]
    st.subheader("Verified dataset profile")
    st.write(
        f"The primary dataset was read as a {quality['delimiter']!r}-separated CSV with "
        f"**{quality['rows_in_file']:,} records** and "
        f"**{quality['columns_in_file']} columns**."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Records", f"{quality['rows_in_file']:,}")
    metric_columns[1].metric("Columns", str(quality["columns_in_file"]))
    metric_columns[2].metric(
        "Missing feature cells",
        f"{data.drop(columns=TARGET_COLUMN).isna().sum().sum():,}",
    )
    metric_columns[3].metric("Duplicate rows", f"{quality['duplicate_rows']:,}")

    st.markdown("#### Outcome distribution")
    counts.index = [TARGET_DISPLAY[label] for label in counts.index]
    counts_frame = counts.rename_axis("Observed target").rename("Records").to_frame()
    counts_frame["Share"] = counts_frame["Records"] / len(data)
    st.dataframe(
        counts_frame.style.format({"Share": "{:.1%}"}),
        width="stretch",
    )
    st.bar_chart(counts_frame["Records"])

    with st.expander("Inspect schema, data types, and quality checks"):
        numeric_columns = set(data.select_dtypes(include="number").columns)
        schema = pd.DataFrame(
            {
                "Column": data.columns,
                "Parsed dtype": data.dtypes.astype(str).values,
                "Missing values": data.isna().sum().values,
                "Distinct values": data.nunique(dropna=True).values,
                "Minimum": [
                    data[column].min() if column in numeric_columns else None
                    for column in data.columns
                ],
                "Maximum": [
                    data[column].max() if column in numeric_columns else None
                    for column in data.columns
                ],
            }
        )
        st.dataframe(schema, width="stretch", hide_index=True)
        st.write("Raw parsed column types:", quality["column_types"])
        st.write(
            "Malformed-width rows:",
            quality["malformed_rows"],
        )
        st.write(
            "Rows without a target (excluded from supervised training):",
            quality["missing_target_rows"],
        )
        st.write(
            "Invalid numeric cells introduced during parsing:",
            quality["invalid_numeric_cells_by_column"] or "None",
        )
        st.write(
            "Literal 'None' category values (preserved as categories):",
            quality["literal_none_counts"] or "None",
        )
        st.caption(
            f"The supplied file validates to {quality['rows_in_file']:,} records "
            f"and {quality['columns_in_file']} columns, with "
            f"{quality['malformed_rows']} malformed-width rows."
        )
        st.dataframe(
            data.loc[:, [*MODEL_FEATURES, TARGET_COLUMN]].head(20),
            width="stretch",
        )

    st.markdown("#### Prediction cutoff and feature selection")
    st.info(
        "The dataset has no event timestamps or documented target horizon. "
        "This demo treats the included attendance and study measures as "
        "available at a pre-outcome monitoring checkpoint; that timing cannot "
        "be validated from the file."
    )
    st.write("Model features:", ", ".join(MODEL_FEATURES))
    for reason, details in EXCLUDED_FEATURES.items():
        st.markdown(f"- **{reason}:** {details}")
    st.warning(
        "This file includes Attendance_Rate, Study_Hours_per_Day, "
        "Assignment_Delay_Days, and Stress_Index. The model uses the first "
        "three listed engagement fields but excludes Stress_Index. It contains "
        "no health assessments or complete, timestamped semester histories. "
        "Staff-entered attendance is a separate policy input and does not "
        "change this model's prediction."
    )


def render_model_evaluation(data: pd.DataFrame, evaluation: dict[str, object]) -> None:
    st.subheader("Binary dropout model benchmark")
    st.write(
        f"An 80/20 stratified split with fixed random seed 42 was made before "
        f"preprocessing. Models were trained on {evaluation['train_rows']:,} "
        f"rows and evaluated on {evaluation['test_rows']:,} held-out rows. "
        "Imputation, scaling, and one-hot encoding are fitted inside each "
        "training pipeline only."
    )
    st.caption(
        "The majority-class model is a naive baseline. Logistic regression and "
        "Extra Trees are compared on the identical test rows; the best of those "
        "two is selected by macro F1 for the scoring demonstration."
    )
    rows = []
    for name in (
        "Majority-class baseline",
        "Logistic regression",
        "Extra Trees",
    ):
        result = evaluation[name]
        rows.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Balanced accuracy": result["balanced_accuracy"],
                "Macro precision": result["macro_precision"],
                "Macro recall": result["macro_recall"],
                "Macro F1": result["macro_f1"],
                "Weighted F1": result["weighted_f1"],
            }
        )
    summary = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        summary.style.format("{:.3f}"),
        width="stretch",
    )
    selected_name = str(evaluation["selected_model_name"])
    st.success(f"Selected scoring model by held-out macro F1: {selected_name}")
    selected_result = evaluation[selected_name]

    st.markdown("#### Selected model: class-wise test performance")
    report = selected_result["class_report"]
    class_metrics = pd.DataFrame(
        {
            TARGET_DISPLAY[label]: {
                metric: report[label][metric]
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in TARGET_LABELS
        }
    ).T
    st.dataframe(class_metrics.style.format("{:.3f}"), width="stretch")

    st.markdown("#### Selected model: confusion matrix")
    matrix = pd.DataFrame(
        selected_result["confusion_matrix"],
        index=pd.Index([TARGET_DISPLAY[label] for label in TARGET_LABELS], name="Actual"),
        columns=pd.Index(
            [TARGET_DISPLAY[label] for label in TARGET_LABELS],
            name="Predicted",
        ),
    )
    st.dataframe(matrix, width="stretch")
    st.caption(
        "Rows are actual classes and columns are predicted classes. This "
        "random split is an internal benchmark, not external, temporal, or "
        "prospective validation. Results are calculated from the loaded file "
        "and are not hard-coded."
    )

    st.markdown("#### Score records with model-eligible fields")
    st.write(
        "Upload a CSV with the model's selected feature columns. "
        "Additional columns, including Target, are ignored. Missing feature "
        "values are handled by the fitted training pipeline."
    )
    upload = st.file_uploader("Upload student records CSV", type="csv")
    if upload is not None:
        try:
            uploaded_frame = pd.read_csv(
                io.BytesIO(upload.getvalue()), sep=None, engine="python"
            )
            uploaded_frame.columns = uploaded_frame.columns.astype(str).str.strip()
            if TARGET_COLUMN in uploaded_frame.columns:
                uploaded_frame = uploaded_frame.drop(columns=[TARGET_COLUMN])
            scoring_features = prepare_scoring_data(uploaded_frame)
            scoring_model = selected_result["model"]
            predictions = scoring_model.predict(scoring_features)
            probabilities = scoring_model.predict_proba(scoring_features)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
            st.error(f"Could not score this CSV: {error}")
        else:
            scored = pd.DataFrame(
                {
                    "Model-predicted outcome": predictions,
                    **{
                        f"Model probability: {TARGET_DISPLAY[str(label)]}": probabilities[:, index]
                        for index, label in enumerate(scoring_model.classes_)
                    },
                }
            )
            st.dataframe(scored, width="stretch")
            st.download_button(
                "Download model predictions",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="model_predictions.csv",
                mime="text/csv",
            )


def render_staff_assessment() -> None:
    st.subheader("Staff-entered factors and policy assessment")
    st.warning(
        "This assessment uses only information entered by staff below. It is "
        "separate from the trained model and does not change model predictions."
    )
    case_ref = st.text_input("Pseudonymous case reference", max_chars=50)
    threshold = st.slider(
        "Local attendance follow-up threshold (%)",
        min_value=50,
        max_value=100,
        value=75,
        help="Set this to your institution's approved policy; 75% is only a demo default.",
    )
    attendance_known = st.checkbox("A staff-verified attendance percentage is available")
    attendance = st.number_input(
        "Staff-entered attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0,
        disabled=not attendance_known,
    )
    concern_choices = list(ATTENDANCE_CONCERNS)
    concerns = st.multiselect(
        "Contextual factors reported to staff (optional)",
        concern_choices,
        help="Do not enter diagnoses or sensitive personal details.",
    )

    policy_flag = attendance_policy_flag(
        attendance if attendance_known else None,
        float(threshold),
    )
    st.markdown("#### Rule-based policy flags")
    if policy_flag:
        st.warning(
            f"Attendance is below the staff-selected threshold of {threshold:.0f}%."
        )
    elif attendance_known:
        st.success(
            f"Attendance is at or above the staff-selected threshold of {threshold:.0f}%."
        )
    else:
        st.info("Attendance is not available; no attendance policy flag is assessed.")
    if concerns:
        st.write("Staff-entered contextual factors:", ", ".join(concerns))
    else:
        st.write("No contextual factors have been selected.")

    st.markdown("#### Counselling and support options")
    st.caption(
        "Non-clinical suggestions based on staff-entered information; offer "
        "voluntarily and follow local safeguarding and referral policies."
    )
    for suggestion in suggested_support(concerns):
        st.markdown(f"- {suggestion}")

    st.markdown("#### Policy-only what-if simulation")
    simulated_attendance = st.slider(
        "Simulated attendance (%)",
        min_value=0,
        max_value=100,
        value=int(attendance) if attendance_known else 75,
        key="simulated_attendance",
    )
    simulated_flag = attendance_policy_flag(
        float(simulated_attendance),
        float(threshold),
    )
    if simulated_flag != policy_flag:
        st.info(
            "Under this simulated attendance value, the rule-based attendance "
            "flag would change. This does not change or explain the ML prediction."
        )
    else:
        st.write(
            "The rule-based attendance flag is unchanged in this scenario. "
            "The simulation is not an ML counterfactual."
        )
    if case_ref.strip():
        st.markdown("#### Save a monitoring or intervention event")
        with st.form("support_event_form"):
            intervention = st.selectbox("Intervention", INTERVENTION_OPTIONS)
            status = st.selectbox("Follow-up status", STATUS_OPTIONS)
            notes = st.text_area(
                "Brief non-sensitive follow-up note",
                max_chars=1000,
                help="Use a pseudonymous case reference; avoid names, diagnoses, and unnecessary sensitive details.",
            )
            save_event = st.form_submit_button("Save staff event")
        if save_event:
            add_support_event(
                SUPPORT_DB_PATH,
                case_ref,
                attendance if attendance_known else None,
                concerns,
                intervention,
                status,
                notes,
            )
            st.success("The staff event was saved to the local monitoring store.")
            st.rerun()


def render_monitoring() -> None:
    st.subheader("Monitoring history and intervention tracking")
    st.caption(
        "Staff-entered monitoring events are stored in a local SQLite file next "
        "to the app. Use pseudonymous references. This local file is not "
        "encrypted; do not enter identifiable or sensitive details unless your "
        "institution has approved the environment and retention process."
    )
    all_events = get_support_events(SUPPORT_DB_PATH)
    if all_events.empty:
        st.info("No staff monitoring events have been saved yet.")
    else:
        st.dataframe(all_events, width="stretch", hide_index=True)

    st.markdown("#### Attendance history for a case")
    case_ref = st.text_input("Case reference to view", key="history_case_ref")
    if case_ref.strip():
        history = get_support_events(SUPPORT_DB_PATH, case_ref)
        if history.empty:
            st.info("No saved monitoring events match that case reference.")
        else:
            history = history.copy()
            history["recorded_at"] = pd.to_datetime(history["recorded_at"])
            st.line_chart(
                history.set_index("recorded_at")[["attendance_pct"]].rename(
                    columns={"attendance_pct": "Staff-entered attendance (%)"}
                )
            )
            st.dataframe(history, width="stretch", hide_index=True)

    st.markdown("#### Synthetic demonstration history")
    st.caption(
        "The following records are synthetic examples for demonstrating the "
        "history view only. They are not real students, are not stored in the "
        "monitoring database, and are never used for training or prediction."
    )
    demo = pd.DataFrame(
        {
            "Demo period": ["Week 1", "Week 3", "Week 5", "Week 7"],
            "Synthetic attendance (%)": [61, 58, 72, 81],
            "Synthetic support status": [
                "Initial check-in",
                "Advising offered",
                "Follow-up scheduled",
                "Follow-up completed",
            ],
        }
    )
    st.dataframe(demo, width="stretch", hide_index=True)
    st.line_chart(
        demo.set_index("Demo period")[["Synthetic attendance (%)"]]
    )


st.set_page_config(
    page_title="Student Success Support",
    page_icon="🎓",
    layout="wide",
)
st.title("Student Success Support")
st.caption("Staff-facing decision support | student-level use requires institutional review")
st.info(
    "The ML engine, policy flags, and staff-entered context are separate. "
    "Predictions are uncertain and must not be used alone for high-stakes "
    "decisions, access restrictions, or disciplinary action."
)

try:
    dataset, data_quality = get_dataset(DATASET_CACHE_VERSION)
    evaluation = get_model_evaluation(dataset)
    initialize_store(SUPPORT_DB_PATH)
except (OSError, sqlite3.Error, ValueError, pd.errors.ParserError) as error:
    st.error(f"The application could not initialize: {error}")
    st.stop()

page = st.sidebar.radio(
    "Application area",
    (
        "Data profile & limitations",
        "ML evaluation & scoring",
        "Staff assessment & counselling",
        "Monitoring & interventions",
    ),
)

if page == "Data profile & limitations":
    render_overview(dataset, data_quality)
elif page == "ML evaluation & scoring":
    render_model_evaluation(dataset, evaluation)
elif page == "Staff assessment & counselling":
    render_staff_assessment()
else:
    render_monitoring()
