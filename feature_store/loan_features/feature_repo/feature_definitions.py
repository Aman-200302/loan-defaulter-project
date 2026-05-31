from datetime import timedelta
import pandas as pd
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# ── Data source — points to our processed CSV ─────────────────────
loan_source = FileSource(
    path="../../../data/processed/features.parquet",   # relative to this file
    timestamp_field="event_timestamp",
)

# ── Entity — what uniquely identifies a record ────────────────────
applicant = Entity(
    name="applicant_id",
    description="Unique loan applicant identifier"
)

# ── Feature view — a named group of features ─────────────────────
loan_features_view = FeatureView(
    name="loan_applicant_features",
    entities=[applicant],
    ttl=timedelta(days=365),
    schema=[
        Field(name="person_age",               dtype=Float32),
        Field(name="person_income",            dtype=Float32),
        Field(name="person_emp_length",        dtype=Float32),
        Field(name="loan_amnt",                dtype=Float32),
        Field(name="loan_int_rate",            dtype=Float32),
        Field(name="loan_percent_income",      dtype=Float32),
        Field(name="cb_person_cred_hist_length", dtype=Float32),
        Field(name="person_home_ownership",    dtype=Float32),
        Field(name="loan_intent",              dtype=Float32),
        Field(name="loan_grade",               dtype=Float32),
        Field(name="cb_person_default_on_file", dtype=Float32),
    ],
    source=loan_source,
)