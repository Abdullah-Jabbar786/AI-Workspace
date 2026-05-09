import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

for df in [train, test]:
    df["Temp_Humidity"]     = df["Temperature_C"] * df["Humidity"]
    df["Moisture_Rainfall"] = df["Soil_Moisture"] * df["Rainfall_mm"]
    df["pH_Conductivity"]   = df["Soil_pH"] * df["Electrical_Conductivity"]

cat_cols = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

encoders = {}
for col in cat_cols:
    le_col = LabelEncoder()
    train[col] = le_col.fit_transform(train[col])
    test[col]  = le_col.transform(test[col])
    encoders[col] = le_col

le = LabelEncoder()
y_encoded = le.fit_transform(train["Irrigation_Need"])
print("Classes:", le.classes_)

X = train.drop(columns=["id", "Irrigation_Need"])
X_test_raw = test.drop(columns=["id"])

scaler = StandardScaler()
X_scaled      = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

# Helper function
def evaluate(name, model, X_tr, y_tr, X_v, y_v):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_v)
    acc = accuracy_score(y_v, preds)
    print(f"{name}: {acc:.4f}")
    cm = confusion_matrix(y_v, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"cm_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()
    return model, acc

print("\n--- Model Training ---")
results = {}
dt_model,   results["Decision Tree"]       = evaluate("Decision Tree", DecisionTreeClassifier(random_state=42), X_train, y_train, X_val, y_val)
nb_model,   results["Naive Bayes"]         = evaluate("Naive Bayes", GaussianNB(), X_train, y_train, X_val, y_val)
lr_model,   results["Logistic Regression"] = evaluate("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced"), X_train, y_train, X_val, y_val)
rf_model,   results["Random Forest"]       = evaluate("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1), X_train, y_train, X_val, y_val)
xgb_model,  results["XGBoost"]             = evaluate("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, eval_metric="mlogloss", random_state=42, n_jobs=-1), X_train, y_train, X_val, y_val)
lgbm_model, results["LightGBM"]            = evaluate("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1), X_train, y_train, X_val, y_val)

def kmeans_classifier(X_tr, y_tr, X_v, y_v):
    n_classes = len(np.unique(y_tr))
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    km.fit(X_tr)
    cluster_labels = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        mask = km.labels_ == c
        if mask.sum() > 0:
            cluster_labels[c] = np.bincount(y_tr[mask]).argmax()
    preds = cluster_labels[km.predict(X_v)]
    acc = accuracy_score(y_v, preds)
    print(f"K-Means (as classifier): {acc:.4f}")
    cm = confusion_matrix(y_v, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix - KMeans")
    plt.tight_layout()
    plt.savefig("cm_KMeans.png", dpi=150)
    plt.close()
    return acc

results["K-Means"] = kmeans_classifier(X_train, y_train, X_val, y_val)

print("\n--- 5-Fold Cross Validation ---")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Naive Bayes":         GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=300, learning_rate=0.1, eval_metric="mlogloss", random_state=42, n_jobs=-1),
    "LightGBM":            LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_scaled, y_encoded, cv=kf, scoring="accuracy")
    print(f"{name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

# LOOCV on small subset
print("\n--- LOOCV on 200 samples ---")
X_small = X_scaled[:200]
y_small = y_encoded[:200]
loo = LeaveOneOut()
loo_scores = cross_val_score(LogisticRegression(max_iter=1000), X_small, y_small, cv=loo, scoring="accuracy")
print(f"LOOCV Accuracy (Logistic Regression, 200 samples): {loo_scores.mean():.4f}")

print("\n--- Final Results Summary ---")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")

print("\n--- Generating Submission with LightGBM ---")
best_model = LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                             class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
best_model.fit(X_scaled, y_encoded)

test_preds  = best_model.predict(X_test_scaled)
test_labels = le.inverse_transform(test_preds)

submission = pd.DataFrame({"id": test["id"], "Irrigation_Need": test_labels})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
print("All confusion matrix PNGs saved in your folder.")