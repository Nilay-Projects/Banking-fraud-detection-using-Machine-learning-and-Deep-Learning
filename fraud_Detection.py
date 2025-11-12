"""
FINANCIAL FRAUD DETECTION (Optimized Version)
---------------------------------------------
Lightweight version of the fraud detection model that avoids
memory errors by sampling and optimizing training.
"""

# ===============================
# STEP 1: IMPORT LIBRARIES
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras import models, layers
import joblib
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===============================
# STEP 2: LOAD DATA (WITH SAMPLING)
# ===============================

print("📥 Loading dataset...")
df = pd.read_csv("creditcard.csv")

print("Original shape:", df.shape)

# ---- Optional: Downsample normal transactions to save memory ----
# Keep all fraud cases, but only a small fraction of normal transactions
fraud_df = df[df["Class"] == 1]
nonfraud_df = df[df["Class"] == 0].sample(frac=0.1, random_state=RANDOM_STATE)  # 10% of normal data
df = pd.concat([fraud_df, nonfraud_df]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("After sampling:", df.shape)
print("Fraud ratio:", df["Class"].mean())

# ===============================
# STEP 3: PREPROCESSING
# ===============================

scaler = StandardScaler()
df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

joblib.dump(scaler, "scaler.joblib")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ===============================
# STEP 4: AUTOENCODER (OPTIMIZED)
# ===============================

# Train only on normal transactions
X_train_legit = X_train[y_train == 0]

input_dim = X_train_legit.shape[1]

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(12, activation='relu'),
    layers.Dense(6, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

print("\n🧠 Training Autoencoder (this may take 1–2 minutes)...")

# Convert to numpy array for training
X_legit_array = X_train_legit.values.astype("float32")

# Reduce epochs and batch size for smaller memory footprint
history = autoencoder.fit(
    X_legit_array, X_legit_array,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Plot training loss (optional)
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Autoencoder Training Loss")
plt.show()

# Evaluate reconstruction error
X_test_array = X_test.values.astype("float32")
recons = autoencoder.predict(X_test_array, batch_size=256)
mse = np.mean(np.power(X_test_array - recons, 2), axis=1)

threshold = np.percentile(mse[y_test == 0], 99)
print("Threshold for fraud detection:", threshold)

y_pred_ae = (mse > threshold).astype(int)

print("\n📊 Autoencoder Results:")
print(classification_report(y_test, y_pred_ae, digits=4))
print("ROC AUC:", roc_auc_score(y_test, mse))

# ===============================
# STEP 5: RANDOM FOREST (LIGHTWEIGHT)
# ===============================

print("\n🧠 Training Random Forest (with light SMOTE)...")

# Use SMOTE with small subset for balance
smote = SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

# Evaluate
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n📊 Random Forest Results:")
print(classification_report(y_test, y_pred_rf, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_proba_rf))

# ===============================
# STEP 6: ENSEMBLE (COMBINE RESULTS)
# ===============================

y_pred_ensemble = ((y_pred_ae == 1) | (y_pred_rf == 1)).astype(int)

print("\n📊 Ensemble Results (Autoencoder + Random Forest):")
print(classification_report(y_test, y_pred_ensemble, digits=4))

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba_rf)
plt.step(recall, precision, where="post")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Random Forest)")
plt.show()

# ===============================
# STEP 7: SAVE MODELS
# ===============================

autoencoder.save("autoencoder_small.h5")
joblib.dump(rf, "random_forest_small.pkl")
joblib.dump({"threshold": threshold}, "ae_meta.pkl")

print("\n✅ All models saved successfully")
