from feast import FeatureStore
import pandas as pd

# Point to our feature store
store = FeatureStore(repo_path="feature_store/loan_features/feature_repo")

# Simulate fetching features for 5 applicants at serving time
entity_df = pd.DataFrame({
    "applicant_id": [0, 1, 2, 3, 4],
    "event_timestamp": pd.Timestamp("2024-01-01")
})

features = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "loan_applicant_features:person_income",
        "loan_applicant_features:loan_amnt",
        "loan_applicant_features:loan_grade",
        "loan_applicant_features:loan_percent_income",
    ]
).to_df()

print("Features fetched from Feast:")
print(features)