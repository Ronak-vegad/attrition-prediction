# What this project does (plain-English guide)

You said you’re confused after placing the CSV in `data/`. This file explains, step-by-step:
1) **What we’re trying to do**  
2) **Why we do it**  
3) **How we do it (exactly what happens when you run each script)**  
4) **How this maps to real-life HR use cases**

---

## 1) The business problem (what are we solving?)

**Employee attrition** means an employee leaves the company (quits, resigns, etc.).  
Attrition is expensive because replacing people costs time and money (recruiting + onboarding + productivity loss).

### Our goal
Build a model that predicts:
- **Attrition = 1** → employee likely to leave (flight-risk)
- **Attrition = 0** → employee likely to stay

### Why “high recall” is important
In many businesses, the biggest cost is **missing a true leaver**.

- **Recall for Attrition=1** answers:  
  “Out of all the employees who actually left, how many did we correctly flag as high-risk?”

High recall means fewer missed flight-risk employees.

---

## 2) What is in the dataset?

The Kaggle / IBM dataset has employee information like:
- Demographics: Age, Gender, MaritalStatus
- Work: Department, JobRole, OverTime
- Compensation: MonthlyIncome, StockOptionLevel
- Satisfaction: EnvironmentSatisfaction, JobInvolvement
- Tenure: YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion

And a target column:
- `Attrition` = **Yes/No**

---

## 3) After you put the CSV in `data/`, what happens next?

You now run **two scripts**:

### Script A — `src/eda.py` (Exploratory Data Analysis)
**Command:**
```bash
python src/eda.py
```

**What it does:**
1. Loads the CSV from:
   - `data/HR-Employee-Attrition.csv` (preferred), OR
   - `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` (your current file name)
2. Creates charts that help you “tell the story” of attrition:
   - Attrition rate (shows class imbalance, usually ~16% yes)
   - Attrition by Gender / Department / JobRole / MaritalStatus
   - Boxplots for numeric variables (Age, MonthlyIncome, etc.) vs Attrition
   - OverTime vs Attrition + crosstab table
   - Correlation with Attrition (numeric only)
3. Saves outputs to:
   - `reports/figures/*.png` (images)
   - some CSV summaries like `overtime_crosstab.csv`, `corr_with_attrition.csv`

**Why we do EDA:**
- Helps you understand *patterns* and *drivers*
- Helps in interviews/resume: you can say what you discovered with visuals

---

### Script B — `src/train.py` (Training + evaluation + SHAP + exporting model)
**Command:**
```bash
python src/train.py --target-recall 0.80
```

This is the “main ML pipeline”.

#### Step-by-step: what `train.py` does

### (1) Load data
Same as EDA: loads the CSV from `data/`.

### (2) Convert target to numeric
The model needs numbers, so:
- `Attrition: Yes → 1`
- `Attrition: No → 0`

### (3) Drop useless columns
Some columns are IDs or constants and don’t help prediction:
- `EmployeeNumber` (ID)
- `EmployeeCount` (constant)
- `Over18` (constant)
- `StandardHours` (constant)

We drop them to reduce noise.

### (4) Split data (IMPORTANT)
We split to measure performance honestly:
- **Train/Valid set**: used to learn patterns and tune threshold
- **Test set**: never used during training; final “realistic” evaluation

This prevents “cheating” and gives a true estimate of performance.

### (5) Preprocessing (done safely in a pipeline)
Real datasets contain mixed data types:
- Numeric (Age, Income, etc.)
- Categorical (JobRole, Department, etc.)

We do:
- Numeric: median imputation (if missing) + standard scaling
- Categorical: most-frequent imputation + one-hot encoding

**One-hot encoding** converts categories into 0/1 columns so models can use them.

### (6) Class imbalance handling (SMOTE)
Attrition “Yes” is usually rare (~16%).
That means models can get “good accuracy” by predicting “No” most of the time, but that’s useless.

So we apply **SMOTE** (Synthetic Minority Over-sampling Technique) on training only to balance the classes.

**Important:** SMOTE is inside an imblearn pipeline, so it is applied only on training folds (prevents leakage).

### (7) Model comparison (LR vs RF vs XGBoost)
We train and compare 3 models using **Stratified K-Fold cross-validation**:
1. Logistic Regression (baseline, interpretable)
2. Random Forest (nonlinear, strong baseline)
3. XGBoost (often best performance on tabular data)

We compute:
- Recall
- Precision
- F1
- ROC-AUC

But we **choose best model primarily by recall**.

### (8) Threshold tuning (this is key for “high recall”)
Most models output a probability like:
> “Attrition risk probability = 0.37”

Default decision rule is:
- if probability ≥ 0.50 → predict attrition

But for high recall we often **lower** the threshold:
- if probability ≥ 0.30 → predict attrition (example)

Lower threshold → flags more employees → recall increases (but more false alarms).

We tune threshold on the validation set to meet:
- `--target-recall 0.80` (you can change it)

It also saves a chart:
- `reports/figures/threshold_tuning.png`

### (9) Final evaluation on test set
After threshold tuning, we evaluate on test set and save:
- confusion matrix image: `reports/figures/confusion_matrix.png`
- metrics + report: `models/metrics.json`

### (10) SHAP explainability (drivers of attrition)
SHAP answers:
- “Which features push predictions toward attrition vs staying?”

We save:
- `reports/figures/shap_summary.png`

This is what you use to say:
> “OverTime, MonthlyIncome, and StockOptionLevel were top drivers of attrition risk.”

### (11) Export model artifacts (for reuse & Streamlit)
We save files so you can predict later without retraining:
- `models/attrition_pipeline.pkl` → full pipeline (preprocessing + SMOTE + model)
- `models/threshold.json` → chosen threshold for decision-making
- `models/schema.json` → list of input columns + allowed categories + default values

---

## 4) How the Streamlit app uses these files

The app `app/streamlit_app.py`:
1. Loads:
   - `models/attrition_pipeline.pkl`
   - `models/threshold.json`
   - `models/schema.json`
2. Shows a form to input employee details
3. Runs prediction and outputs:
   - probability (risk score)
   - decision using the tuned threshold (high risk vs lower risk)

**Important:** The Streamlit app will only work after you run `train.py` (because the model files must exist).

---

## 5) Real-life use cases (how this is used in companies)

### Use case 1: HR retention prioritization (most common)
HR can’t intervene with everyone. The model helps identify a shortlist:
- Top 5–10% highest risk employees
- Then HR offers retention actions (compensation review, manager coaching, internal mobility, workload adjustments, etc.)

### Use case 2: “What-if” scenarios
HR can ask:
- “What if we reduce overtime?”
- “What if we adjust compensation / stock options?”

With SHAP, you can see which levers are driving risk.

### Use case 3: Team/department-level monitoring
Aggregate risk patterns:
- Which departments show rising risk?
- Are there hotspots related to workload, satisfaction, tenure, promotions?

---

## 6) Important caution (very important for interviews)

This is decision support, not an automatic decision-maker.
- Don’t use it to fire people or make unfair decisions.
- Add fairness checks (error rates across groups).
- Monitor drift and retrain periodically.

---

## 7) What you should do next (simple checklist)

1) Run:
```bash
python src/eda.py
```
Open `reports/figures/` and view the plots.

2) Run:
```bash
python src/train.py --target-recall 0.80
```
Check:
- `reports/figures/confusion_matrix.png`
- `reports/figures/shap_summary.png`
- `models/metrics.json`

3) Run the app:
```bash
streamlit run app/streamlit_app.py
```

---

## 8) If you want, I can personalize the explanation to your results
After you run training once, send me:
- your confusion matrix numbers, OR
- the `models/metrics.json` content

Then I can write “your exact project story” (what your model achieved, your threshold, and your top SHAP drivers).

