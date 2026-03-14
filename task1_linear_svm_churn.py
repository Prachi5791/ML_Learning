import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--data", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--C", nargs="+", type=float, required=True)

args = parser.parse_args()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(args.data)

# -----------------------------
# Data Cleaning
# -----------------------------

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove rows with missing values
df = df.dropna()

# Extract Customer IDs AFTER cleaning
customer_ids = df["customerID"]

# Drop ID column from dataset
df = df.drop(columns=["customerID"])

# -----------------------------
# Encode Target
# -----------------------------
label = LabelEncoder()
df[args.target] = label.fit_transform(df[args.target])

# -----------------------------
# Encode categorical features
# -----------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------
# Split features and labels
# -----------------------------
X = df.drop(columns=[args.target])
y = df[args.target]

# -----------------------------
# Train Test Split (Stratified)
# -----------------------------
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    customer_ids,
    test_size=args.test_size,
    stratify=y,
    random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Training Loop for multiple C
# -----------------------------
results = []

for c in args.C:

    model = SVC(kernel="linear", C=c, probability=True)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    num_support_vectors = len(model.support_)

    print("\n==============================")
    print(f"C Value: {c}")
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)
    print("Support Vectors:", num_support_vectors)

    results.append([
        c,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        num_support_vectors
    ])

    # Save predictions (last model)
    predictions_df = pd.DataFrame({
        "CustomerID": id_test,
        "Actual": y_test,
        "Predicted": predictions,
        "Score": probabilities
    })

# -----------------------------
# Save Results CSV
# -----------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "C",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Support_Vectors"
    ]
)

results_df.to_csv("svm_linear_results.csv", index=False)

# Save predictions
predictions_df.to_csv("test_predictions.csv", index=False)

print("\n==============================")
print("Summary Table")
print(results_df)

print("\nFiles Generated:")
print("svm_linear_results.csv")
print("test_predictions.csv")

print("\n==============================")
print("INTERPRETATION")

best_row = results_df.loc[results_df["F1"].idxmax()]
best_c = best_row["C"]

print("\nEffect of Regularization Parameter (C):")

print("""
C controls the trade-off between margin width and classification error.

Smaller C values allow a wider margin but tolerate more misclassification,
which may lead to underfitting.

Larger C values reduce the margin width and attempt to classify training
points more strictly, which may increase the risk of overfitting.
""")

print("\nObserved Trend:")

for i in range(len(results_df)):
    row = results_df.iloc[i]
    print(
        f"C = {row['C']} → "
        f"Accuracy: {row['Accuracy']:.4f}, "
        f"F1: {row['F1']:.4f}, "
        f"Support Vectors: {int(row['Support_Vectors'])}"
    )

print("""
Support vectors are the training samples closest to the decision boundary.
They determine the position of the separating hyperplane in SVM.
A higher number of support vectors generally indicates a wider margin and
stronger regularization.
""")

print("\n==============================")
print("CONCLUSION")

print(f"""
Among the tested values, C = {best_c} achieved the best balance between
precision and recall as reflected by the highest F1 score.

This indicates that the model with this C value provides the best
generalization performance on unseen data while maintaining a stable margin.

Therefore, C = {best_c} is considered the optimal regularization parameter
for this Linear SVM churn prediction model.
""")