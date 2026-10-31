"""
Predicting Airline Passenger Satisfaction
Kaggle Playground Series - Season 6 Episode 10

Description:
End-to-end machine learning pipeline to predict passenger satisfaction
using survey ratings, flight details, and customer demographics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve


def load_and_inspect_data():
    # Here I am loading the train and test datasets from the data folder
    print("Loading data from data/ folder...")
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Train dataset shape: {train.shape}")
    print(f"Test dataset shape:  {test.shape}")

    # Checking for missing values
    missing_train = train.isnull().sum()
    missing_test = test.isnull().sum()
    print("\nMissing values in train set:")
    print(missing_train[missing_train > 0])
    print("\nMissing values in test set:")
    print(missing_test[missing_test > 0])

    return train, test


def preprocess_data(train_df, test_df):
    # Copying dataframes so I don't accidentally modify the originals
    train = train_df.copy()
    test = test_df.copy()

    # Handling missing values in 'Arrival Delay in Minutes':
    # Only 292 rows in train and 130 in test have missing arrival delay
    # Since flight departure and arrival delays are closely linked,
    # I fill missing arrival delay with departure delay of the same flight
    train["Arrival Delay in Minutes"] = train["Arrival Delay in Minutes"].fillna(
        train["Departure Delay in Minutes"]
    )
    test["Arrival Delay in Minutes"] = test["Arrival Delay in Minutes"].fillna(
        test["Departure Delay in Minutes"]
    )

    # Convert the boolean target satisfaction into integers (0 and 1)
    # 0 = neutral or dissatisfied, 1 = satisfied
    train["satisfaction"] = train["satisfaction"].astype(int)

    # Encoding categorical columns:
    # Gender: Female -> 0, Male -> 1
    # Customer Type: disloyal Customer -> 0, Loyal Customer -> 1
    # Type of Travel: Personal Travel -> 0, Business travel -> 1
    # Class: Eco -> 0, Eco Plus -> 1, Business -> 2 (ordinal order)
    gender_map = {"Female": 0, "Male": 1}
    customer_map = {"disloyal Customer": 0, "Loyal Customer": 1}
    travel_map = {"Personal Travel": 0, "Business travel": 1}
    class_map = {"Eco": 0, "Eco Plus": 1, "Business": 2}

    for df in [train, test]:
        df["Gender"] = df["Gender"].map(gender_map)
        df["Customer Type"] = df["Customer Type"].map(customer_map)
        df["Type of Travel"] = df["Type of Travel"].map(travel_map)
        df["Class"] = df["Class"].map(class_map)

    # Drop id from features since it's just a row counter
    feature_columns = [col for col in train.columns if col not in ["id", "satisfaction"]]
    
    X = train[feature_columns]
    y = train["satisfaction"]
    X_test = test[feature_columns]
    test_ids = test["id"]

    print(f"\nFeatures selected ({len(feature_columns)} total):")
    print(feature_columns)

    return X, y, X_test, test_ids, feature_columns


def train_and_evaluate_models(X, y):
    # Splitting into 80% train and 20% validation
    # Using stratify=y so that the class ratio (satisfied vs dissatisfied) remains the same
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining set size:   {X_train.shape[0]} rows")
    print(f"Validation set size: {X_val.shape[0]} rows")

    # Scaling features specifically for Logistic Regression
    # Gradient descent converges better and weights don't get dominated by large scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Model 1: Logistic Regression (simple linear baseline)
    print("\n--- Training Model: Logistic Regression ---")
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_probs)
    lr_acc = accuracy_score(y_val, (lr_probs >= 0.5).astype(int))
    print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Model 2: Decision Tree Classifier (simple non-linear model)
    # Using max_depth=10 to keep it from overfitting too deeply on the train set
    print("\n--- Training Model: Decision Tree Classifier ---")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_probs = dt.predict_proba(X_val)[:, 1]
    dt_auc = roc_auc_score(y_val, dt_probs)
    dt_acc = accuracy_score(y_val, (dt_probs >= 0.5).astype(int))
    print(f"Decision Tree ROC-AUC: {dt_auc:.4f}")
    print(f"Decision Tree Accuracy: {dt_acc:.4f}")

    # Model 3: LightGBM (gradient boosted trees)
    # LightGBM handles tabular features quickly and often gets strong results
    print("\n--- Training Model: LightGBM Classifier ---")
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        random_state=42,
        verbose=-1
    )
    lgb.fit(X_train, y_train)
    lgb_probs = lgb.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_probs)
    lgb_acc = accuracy_score(y_val, (lgb_probs >= 0.5).astype(int))
    print(f"LightGBM ROC-AUC: {lgb_auc:.4f}")
    print(f"LightGBM Accuracy: {lgb_acc:.4f}")

    print("\nSummary of Model Validation Scores:")
    print(f"Logistic Regression -> ROC-AUC: {lr_auc:.4f} | Accuracy: {lr_acc:.4f}")
    print(f"Decision Tree       -> ROC-AUC: {dt_auc:.4f} | Accuracy: {dt_acc:.4f}")
    print(f"LightGBM            -> ROC-AUC: {lgb_auc:.4f} | Accuracy: {lgb_acc:.4f}")

    print("\nClassification Report for Best Model (LightGBM):")
    print(classification_report(y_val, (lgb_probs >= 0.5).astype(int), digits=4))

    return lgb, X_train, X_val, y_train, y_val


def make_submission(model, X_test, test_ids, output_path="submission.csv"):
    # Predicting probability of satisfaction (class 1) on the test data
    print(f"\nGenerating predictions for test set ({len(test_ids)} rows)...")
    test_probs = model.predict_proba(X_test)[:, 1]

    sub = pd.DataFrame({
        "id": test_ids,
        "satisfaction": test_probs
    })
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("Sample rows:")
    print(sub.head())


if __name__ == "__main__":
    # Running the full workflow step by step
    raw_train, raw_test = load_and_inspect_data()
    X, y, X_test, test_ids, feature_names = preprocess_data(raw_train, raw_test)
    best_model, X_tr, X_val, y_tr, y_val = train_and_evaluate_models(X, y)
    make_submission(best_model, X_test, test_ids, output_path="submission.csv")
    print("\nAll done!")

