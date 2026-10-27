import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ml_engine import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    TARGET_LABELS,
    evaluate_models,
    load_dataset,
    prepare_scoring_data,
)
from support_engine import (
    add_support_event,
    attendance_policy_flag,
    get_support_events,
    initialize_store,
    suggested_support,
)


ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.quality = load_dataset(ROOT / "student_dropout_dataset_v3.csv")

    def test_primary_dataset_is_parsed_and_profiled(self):
        self.assertEqual(self.data.shape, (10000, 19))
        self.assertEqual(self.quality["delimiter"], ",")
        self.assertEqual(self.quality["rows_in_file"], 10000)
        self.assertEqual(self.quality["columns_in_file"], 19)
        self.assertEqual(self.quality["missing_target_rows"], 0)
        self.assertEqual(
            self.quality["missing_features_by_column"],
            {
                "Family_Income": 500,
                "Study_Hours_per_Day": 500,
                "Stress_Index": 500,
            },
        )
        self.assertEqual(self.quality["duplicate_rows"], 0)
        self.assertEqual(
            self.quality["class_counts"].to_dict(),
            {"0": 7646, "1": 2354},
        )
        self.assertEqual(self.quality["literal_none_counts"]["Parental_Education"], 511)

    def test_selected_features_exclude_identifiers_sensitive_fields_and_gpa(self):
        self.assertEqual(len(MODEL_FEATURES), 8)
        self.assertNotIn("Student_ID", MODEL_FEATURES)
        self.assertFalse(any("GPA" in column for column in MODEL_FEATURES))
        self.assertNotIn("Gender", MODEL_FEATURES)
        self.assertNotIn("Stress_Index", MODEL_FEATURES)
        self.assertNotIn(TARGET_COLUMN, MODEL_FEATURES)
        self.assertEqual(set(self.data[TARGET_COLUMN]), set(TARGET_LABELS))

    def test_scoring_rejects_missing_required_features(self):
        with self.assertRaisesRegex(ValueError, "Missing model feature columns"):
            prepare_scoring_data(pd.DataFrame({"Course": [1]}))

    def test_scoring_selects_only_model_features(self):
        sample = self.data.loc[:1, list(MODEL_FEATURES)].copy()
        sample["Target"] = self.data.loc[:1, TARGET_COLUMN]
        sample["irrelevant extra field"] = "ignored"
        scored = prepare_scoring_data(sample)
        self.assertEqual(tuple(scored.columns), MODEL_FEATURES)
        self.assertEqual(len(scored), 2)

    def test_malformed_row_width_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.csv"
            path.write_text("column_a;column_b\none\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected number of fields"):
                load_dataset(path)


class ModelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, _ = load_dataset(ROOT / "student_dropout_dataset_v3.csv")
        cls.results, cls.x_test, cls.y_test = evaluate_models(data)

    def test_models_use_same_holdout_and_report_real_metrics(self):
        self.assertEqual(self.results["train_rows"] + self.results["test_rows"], 10000)
        self.assertEqual(len(self.y_test), self.results["test_rows"])
        for model_name in (
            "Majority-class baseline",
            "Logistic regression",
            "Extra Trees",
        ):
            result = self.results[model_name]
            self.assertEqual(len(result["predictions"]), len(self.y_test))
            self.assertEqual(result["confusion_matrix"].shape, (2, 2))
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
                self.assertGreaterEqual(result[metric], 0)
                self.assertLessEqual(result[metric], 1)
            for label in TARGET_LABELS:
                self.assertIn(label, result["class_report"])
                for metric in ("precision", "recall", "f1-score", "support"):
                    self.assertIn(metric, result["class_report"][label])

    def test_selected_model_is_best_non_naive_macro_f1(self):
        selected = self.results["selected_model_name"]
        self.assertIn(selected, ("Logistic regression", "Extra Trees"))
        other = (
            "Extra Trees"
            if selected == "Logistic regression"
            else "Logistic regression"
        )
        self.assertGreaterEqual(
            self.results[selected]["macro_f1"],
            self.results[other]["macro_f1"],
        )

    def test_selected_model_scores_held_out_rows(self):
        model = self.results[self.results["selected_model_name"]]["model"]
        sample = self.x_test.iloc[:5]
        predictions = model.predict(sample)
        probabilities = model.predict_proba(sample)
        self.assertEqual(len(predictions), 5)
        self.assertEqual(probabilities.shape, (5, len(TARGET_LABELS)))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertTrue(abs(probabilities[0].sum() - 1) < 1e-8)


class StaffSupportTests(unittest.TestCase):
    def test_attendance_policy_boundaries_and_missing_value(self):
        self.assertFalse(attendance_policy_flag(None, 75))
        self.assertTrue(attendance_policy_flag(74.9, 75))
        self.assertFalse(attendance_policy_flag(75, 75))
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            attendance_policy_flag(101, 75)

    def test_counselling_is_based_only_on_selected_staff_factors(self):
        suggestions = suggested_support(["Financial strain"])
        self.assertIn("financial-support service", suggestions[0])
        self.assertTrue(suggested_support([]))
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            suggested_support(["Unverified diagnosis"])

    def test_intervention_event_persists_in_monitoring_history(self):
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "support.sqlite3"
            initialize_store(db_path)
            add_support_event(
                db_path,
                "  CASE-TEST  ",
                72.5,
                ["Academic confidence"],
                "Advising appointment offered",
                "Follow-up scheduled",
                "  Recheck after advising.  ",
            )
            events = get_support_events(db_path, "CASE-TEST")
            self.assertEqual(len(events), 1)
            self.assertEqual(events.iloc[0]["case_ref"], "CASE-TEST")
            self.assertEqual(events.iloc[0]["attendance_pct"], 72.5)
            self.assertEqual(events.iloc[0]["status"], "Follow-up scheduled")
            self.assertEqual(events.iloc[0]["notes"], "Recheck after advising.")

    def test_intervention_event_validates_attendance(self):
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "support.sqlite3"
            initialize_store(db_path)
            with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                add_support_event(
                    db_path,
                    "CASE-TEST",
                    101,
                    [],
                    "No intervention recorded",
                    "Open",
                    "",
                )


if __name__ == "__main__":
    unittest.main()
