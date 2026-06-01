import pandas as pd
import numpy as np
import sklearn
import xgboost
import mlflow
import fastapi
import evidently

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("mlflow:", mlflow.__version__)
print("fastapi:", fastapi.__version__)
print("evidently:", evidently.__version__)
print("\nAll good! Environment is ready.")