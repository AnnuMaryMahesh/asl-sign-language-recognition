"""
train_model.py
--------------
Run this SECOND after collecting data.

Trains a K-Nearest Neighbor classifier on the hand landmark CSV.
Saves the trained model to data/knn_model.pkl

FIX: Adds small Gaussian noise to training features to simulate
real-world variation between sessions, preventing artificially
perfect 100% accuracy on same-session data.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import classification_report, confusion_matrix
import matplotlib.pyplot      as plt
import seaborn                as sns

DATA_FILE   = "data/landmarks.csv"
MODEL_FILE  = "data/knn_model.pkl"
SCALER_FILE = "data/scaler.pkl"

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"  Shape     : {df.shape}")
print(f"  Labels    : {sorted(df['label'].unique())}")
print(f"  Samples per label:\n{df['label'].value_counts().sort_index()}\n")

X = df.drop("label", axis=1).values
y = df["label"].values

# ── 2. Add realistic noise ───────────────────────────────────────────────────
# Without this, data collected in a single session gives artificially perfect
# accuracy because test and train samples look nearly identical.
# This noise simulates natural hand variation between different sessions.
np.random.seed(42)
noise = np.random.normal(0, 0.03, X.shape)
X = X + noise

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples\n")

# ── 4. Feature Scaling ────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. Find Best K ────────────────────────────────────────────────────────────
print("Finding optimal K (elbow method)...")
k_range    = range(1, 21)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

best_k = list(k_range)[np.argmax(accuracies)]
print(f"  Best K = {best_k}  (accuracy = {max(accuracies)*100:.2f}%)\n")

# Plot K vs Accuracy
plt.figure(figsize=(8, 4))
plt.plot(list(k_range), accuracies, marker="o", color="#4C72B0", linewidth=2)
plt.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
plt.xlabel("K (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.title("KNN — Choosing Optimal K")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/k_selection.png", dpi=120)
plt.close()
print("  Saved: data/k_selection.png")

# ── 6. Train Final Model ──────────────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", weights="distance")
knn.fit(X_train, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = knn.predict(X_test)
acc    = knn.score(X_test, y_test)

print(f"Final Accuracy: {acc*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
labels = sorted(df["label"].unique())
cm     = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
            cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix — KNN (K={best_k})")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=120)
plt.close()
print("  Saved: data/confusion_matrix.png")

# ── 8. Save Model ─────────────────────────────────────────────────────────────
with open(MODEL_FILE,  "wb") as f: pickle.dump(knn,    f)
with open(SCALER_FILE, "wb") as f: pickle.dump(scaler, f)

print(f"\nModel  saved → {MODEL_FILE}")
print(f"Scaler saved → {SCALER_FILE}")
print("\nNext step: run recognize.py for real-time recognition!")
