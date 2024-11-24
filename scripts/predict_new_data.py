# 1. Load the new dataset.

import pickle

# Load the trained model from the file
model_path = "../models/final_model.pkl"
with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully")


# 2. Preprocess the new data to match the format of the training data (e.g., same features, transformations, etc.).

import sys

sys.path.append("../src")  # Adjust path if necessary
from preprocessing import preprocess_data

import pandas as pd

# Update the path to the new data to test here:
new_data = pd.read_csv("../data/raw/Modelo_Clasificacion_Dataset.csv", index_col=[0])

new_data = preprocess_data(new_data)


# 3. Use the trained model to make predictions on the new data.

X_new = new_data.drop(columns=["clase_binaria"])
y_true = new_data["clase_binaria"]

y_new_pred = loaded_model.predict(X_new)
print("Predictions for the new data:", y_new_pred)


# 4. Evaluate the New Predictions.

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_true, y_new_pred)
print(f"Accuracy: {accuracy:.4f}")

# Precision, Recall, F1 Score
precision = precision_score(y_true, y_new_pred, pos_label=1)
recall = recall_score(y_true, y_new_pred, pos_label=1)
f1 = f1_score(y_true, y_new_pred, pos_label=1)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ROC-AUC Score
roc_auc = roc_auc_score(y_true, y_new_pred)
print(f"ROC AUC: {roc_auc:.4f}")
