"""Single source of truth for the dataset schema.

The UCI file ``processed.cleveland.data`` has no header row and 14 columns
(13 attributes + the diagnosis ``num``). Missing values are encoded as ``?``.
Codes below follow the UCI ``heart-disease.names`` documentation.
"""

RAW_COLUMNS = [
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",   # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1 = yes)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",        # number of major vessels (0-3) coloured by fluoroscopy
    "thal",      # 3 normal, 6 fixed defect, 7 reversible defect
    "num",       # diagnosis of heart disease, 0 (absent) .. 4
]

RAW_TARGET = "num"
TARGET = "target"  # binary: 0 = absence (num == 0), 1 = presence (num in 1..4)

FEATURES = RAW_COLUMNS[:-1]

# Continuous / count features -> median imputation + standard scaling
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Binary flags -> most-frequent imputation, passed through unchanged (already 0/1)
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Nominal codes -> most-frequent imputation + one-hot encoding
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Engineered feature added by heartrisk.features.ClinicalFeatureBuilder
ENGINEERED_FEATURES = ["hr_pct_predicted"]

# Allowed codes for categorical / binary features (UCI documentation)
ALLOWED_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
    "num": {0, 1, 2, 3, 4},
}

# Physiologically plausible ranges used for data validation and API input validation.
# Deliberately wider than the observed Cleveland ranges so valid unseen patients pass.
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalach": (40, 250),
    "oldpeak": (0.0, 10.0),
}

TARGET_LABELS = {0: "no heart disease", 1: "heart disease present"}
