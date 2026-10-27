# Student Success Support

A local staff-facing Streamlit app for exploring the requested
`student_dropout_dataset_v3.csv`, evaluating a binary dropout model, and
keeping model outputs separate from staff policy and support workflows.

## Dataset and verified quality

The WhatsApp transfer file was copied into this project as
`student_dropout_dataset_v3.csv`; the original `data.csv` was left unchanged.
The new dataset is comma-delimited and independently parses to **10,000 rows
and 19 columns**. Its target is `Dropout`, encoded as `0` and `1`:

| Target | Records | Share |
|---|---:|---:|
| 0 — No dropout | 7,646 | 76.46% |
| 1 — Dropout | 2,354 | 23.54% |

There are no malformed-width rows, missing target labels, or duplicate rows.
There are **1,500 blank feature cells**: 500 each in `Family_Income`,
`Study_Hours_per_Day`, and `Stress_Index`. The 511 literal `None` values in
`Parental_Education` are retained as a category rather than silently treated
as blanks. The app displays parsed schema, data types, missing values, class
counts, and quality checks at runtime.

## Prediction features and leakage safeguards

The classifier uses these eight features:

- Numeric: `Study_Hours_per_Day`, `Attendance_Rate`,
  `Assignment_Delay_Days`, `Travel_Time_Minutes`
- Categorical: `Internet_Access`, `Part_Time_Job`, `Semester`, `Department`

`Student_ID` is excluded as an identifier that may encode row order. `GPA`,
`Semester_GPA`, and `CGPA` are excluded because they may be post-outcome or
outcome-proxy values and there are no timestamps to confirm their availability
before the dropout label. `Age`, `Gender`, `Family_Income`, `Scholarship`,
`Parental_Education`, and `Stress_Index` are omitted to avoid demographic,
socioeconomic, support-status, and sensitive wellbeing attributes in the
prediction features. These choices do not establish that the remaining fields
are genuinely available before dropout: the CSV has no event timestamps,
cohort key, or documented prediction horizon. The pre-outcome monitoring
checkpoint is an explicit demonstration assumption.

The app compares a majority-class baseline, class-balanced logistic
regression, and class-balanced Extra Trees on the same reproducible,
stratified 80/20 holdout split (random seed 42). Imputation, scaling, and
one-hot encoding are fit inside model pipelines on training rows only. It
calculates class-wise precision, recall, F1 and support, confusion matrix,
accuracy, balanced accuracy, macro precision/recall/F1, and weighted F1 from
the loaded file at runtime. The best non-baseline model by held-out macro F1
is used for the scoring demonstration.

The split is a random internal benchmark, not external, temporal, or
prospective validation. Dataset provenance and label-generation details are
not established by the CSV. Results should not be assumed to generalize to
other institutions or real students.

## Staff assessment and support

The dataset includes `Attendance_Rate`, study hours, assignment delays,
travel time, and `Stress_Index`; the model uses attendance, study hours,
assignment delays, and travel time, but excludes stress. The file does not
provide health assessments or complete timestamped semester histories.

Staff-entered attendance, configurable attendance threshold flags, policy-only
what-if simulations, staff-entered contextual factors, non-clinical support
suggestions, counselling referrals, and intervention/monitoring history are
separate from model scoring. Editing the policy attendance field does not
change the model prediction. Synthetic monitoring records are explicitly
labeled examples, not real records or training data.

Staff events are stored in the local `student_support.sqlite3` database. It is
not encrypted. Use pseudonymous references and do not enter identifiable or
sensitive information unless your institution has approved the environment,
access controls, and retention process.

This is an educational prototype, not a validated early-warning or clinical
system. Model outputs are uncertain and are not causal findings. Do not use
them alone for admissions, disciplinary, access, or other high-stakes
decisions. Institutional validation, privacy/security review, fairness
assessment, and human oversight are required before real deployment.

## Run on Windows

From PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If dependencies are already installed:

```powershell
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`. Keep
`student_dropout_dataset_v3.csv` beside `app.py`.

## Run validation tests

```powershell
python -m unittest discover -s tests -v
```

## Reusable build prompt

> Build and validate the staff-facing Streamlit project using the supplied
> `student_dropout_dataset_v3.csv` as its primary dataset. Independently detect
> and verify its comma-delimited format, columns, data types, target counts,
> missingness, malformed rows, duplicates, and invalid labels before fitting a
> model. The observed file has 10,000 rows, 19 columns, and binary `Dropout`
> labels `0` and `1`; verify rather than hard-code those expectations. Preserve
> literal `None` values in categorical fields unless confirmed to represent
> missingness. Train a binary classifier using only
> `Study_Hours_per_Day`, `Attendance_Rate`, `Assignment_Delay_Days`,
> `Travel_Time_Minutes`, `Internet_Access`, `Part_Time_Job`, `Semester`, and
> `Department`. Exclude `Student_ID`, GPA fields, age, gender, family income,
> scholarship, parental education, and stress index, documenting the reasons
> and the unverified prediction-time assumption. Split before fitting
> preprocessing; compare a majority baseline, class-balanced logistic
> regression, and Extra Trees on the identical stratified holdout. Calculate
> actual class-wise precision/recall/F1/support, confusion matrix, accuracy,
> balanced accuracy, macro metrics, and weighted F1; never fabricate results.
> Add CSV scoring with the selected feature schema. Keep model predictions
> separate from staff-entered attendance policy checks, what-if rule
> simulations, contextual factors, counselling suggestions, and persistent
> intervention history. Label synthetic examples clearly and do not train on
> them. Document unknown dataset provenance, missing timestamps, privacy, and
> generalization limitations. Run tests and validate all app workflows.
