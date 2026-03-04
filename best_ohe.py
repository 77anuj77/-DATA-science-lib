# ============================================================
# BEST PRACTICE – One Hot Encoding (No Data Leakage)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("Telco_Customer.csv")   # Change filename if needed

# ------------------------------------------------------------
# Separate Features and Target
# ------------------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})   # Convert target to numeric

# ------------------------------------------------------------
# Train-Test Split (BEFORE encoding)
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Identify Categorical Columns
# ------------------------------------------------------------

categorical_cols = X_train.select_dtypes(include='object').columns

# ------------------------------------------------------------
# Create OneHotEncoder
# drop='first' → avoids dummy variable trap
# handle_unknown='ignore' → prevents error if new category appears
# ------------------------------------------------------------

ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

# ------------------------------------------------------------
# Column Transformer
# Applies encoding only to categorical columns
# Keeps numerical columns as they are
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', ohe, categorical_cols)
    ],
    remainder='passthrough'
)

# ------------------------------------------------------------
# Create Full Pipeline
# Encoding + Model together
# ------------------------------------------------------------

model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000))
])

# ------------------------------------------------------------
# Train Model (Only on training data)
# ------------------------------------------------------------

model.fit(X_train, y_train)

# ------------------------------------------------------------
# Predict on Test Data
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("One-Hot Encoding Accuracy:", accuracy)