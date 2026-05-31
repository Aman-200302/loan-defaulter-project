import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

# ── 1. Load raw data ──────────────────────────────────────────────
df = pd.read_csv('data/raw/credit_risk_dataset.csv')

# ── 2. Drop obvious outliers spotted in EDA ───────────────────────
df = df[df['person_age'] < 100]
df = df[df['person_emp_length'] < 60]

# ── 3. Separate features and target ──────────────────────────────
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# ── 4. Define column groups ───────────────────────────────────────
numeric_features = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length'
]

categorical_features = [
    'person_home_ownership', 'loan_intent',
    'loan_grade', 'cb_person_default_on_file'
]

# ── 5. Build sub-pipelines ────────────────────────────────────────
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # fills missing with median
    ('scaler', StandardScaler())                    # zero mean, unit variance
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# ── 6. Combine into one ColumnTransformer ─────────────────────────
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# ── 7. Fit and transform ──────────────────────────────────────────
X_processed = preprocessor.fit_transform(X)

# ── 8. Save outputs ───────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Rebuild as a clean DataFrame with column names
all_columns = numeric_features + categorical_features
X_out = pd.DataFrame(X_processed, columns=all_columns)
X_out['loan_status'] = y.reset_index(drop=True)

# Feast requires a timestamp column
X_out['event_timestamp'] = pd.Timestamp('2024-01-01')
X_out['applicant_id'] = range(len(X_out))

X_out.to_parquet('data/processed/features.parquet', index=False)
joblib.dump(preprocessor, 'models/preprocessor.pkl')

print("✓ Preprocessing done")
print(f"  Input shape:  {X.shape}")
print(f"  Output shape: {X_out.shape}")
print(f"  Saved to: data/processed/features.csv")
print(f"  Preprocessor saved to: models/preprocessor.pkl")