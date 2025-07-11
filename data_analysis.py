# import pandas as pd
# import numpy as np
# import os
# np.complex = complex

# import librosa
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.ensemble import RandomForestClassifier
# # from imblearn.over_sampling import SMOTE  # SMOTE commented out for now

# # 1) Load your labeled file
# df = pd.read_csv('audio_file_labels_asof.csv')

# # 2) Drop rows missing the target
# df = df.dropna(subset=['queen_status'])

# # 3) Binary encode: QR → 1, else → 0
# df['label'] = (df['queen_status'] == 'QR').astype(int)

# # 4) Feature extraction: Mel + MFCC + delta MFCC
# def extract_features(path,
#                      sr=16000,
#                      n_mels=128,
#                      n_fft=2048,
#                      hop_length=512,
#                      n_mfcc=13):
#     y, _ = librosa.load(path, sr=sr)

#     # Mel-spectrogram → dB
#     S = librosa.feature.melspectrogram(
#         y=y, sr=sr,
#         n_mels=n_mels,
#         n_fft=n_fft,
#         hop_length=hop_length
#     )
#     S_db = librosa.power_to_db(S, ref=np.max)
#     mel_feats = np.hstack([S_db.mean(axis=1), S_db.std(axis=1)])  

#     # MFCC → mean+std
#     mfcc = librosa.feature.mfcc(
#         y=y, sr=sr,
#         n_mfcc=n_mfcc,
#         n_fft=n_fft,
#         hop_length=hop_length
#     )
#     mfcc_feats = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])  # 2*n_mfcc

#     # Delta MFCC → mean+std
#     delta = librosa.feature.delta(mfcc)
#     delta_feats = np.hstack([delta.mean(axis=1), delta.std(axis=1)])  # 2*n_mfcc

#     # --- ACI code commented out ---
#     # # Acoustic Complexity Index (ACI) on Mel bands
#     # aci = np.sum(np.abs(np.diff(S_db, axis=1)), axis=1)  # length n_mels
#     # aci_feats = np.array([aci.mean(), aci.std()])        # 2 dims

#     # concatenate all features (without ACI)
#     return np.concatenate([mel_feats, mfcc_feats, delta_feats])

# # 5) Build feature matrix
# features = np.vstack([extract_features(fp) for fp in df['audio_path']])
# labels   = df['label'].values

# # 6) Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     features, labels,
#     test_size=0.2,
#     stratify=labels,
#     random_state=42 #change here for diff train/test split of random data
# )

# # 7) SMOTE block (commented out for now)
# # sm = SMOTE(random_state=42)
# # X_train, y_train = sm.fit_resample(X_train, y_train)
# # print("After SMOTE:", np.bincount(y_train))

# # 8) Fit a 5‑NN classifier on the (un‑SMOTEd) training data
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# rf = RandomForestClassifier(
#     n_estimators=100,
#     class_weight='balanced',
#     random_state=42
# )
# rf.fit(X_train, y_train)

# # 9) Evaluate on the original test set
# y_pred = rf.predict(X_test)
# print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
# print(classification_report(y_test, y_pred))

# # 9) Evaluate on original test set
# y_pred = knn.predict(X_test)
# print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}\n")
# print(classification_report(y_test, y_pred))


import os
import numpy as np
import pandas as pd
import joblib
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler




np.complex = complex  # to fix librosa issue on older NumPy

# 1) Load labeled data
df = pd.read_csv('audio_file_labels_asof.csv')
df = df.dropna(subset=['queen_status'])
df['label'] = (df['queen_status'] == 'QR').astype(int)

# 2) Define feature extraction
def extract_features(path, sr=16000, n_mels=128, n_fft=2048, hop_length=512, n_mfcc=13):
    y, _ = librosa.load(path, sr=sr)
    
    # Mel-spectrogram → dB
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    mel_feats = np.hstack([S_db.mean(axis=1), S_db.std(axis=1)])
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_feats = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])
    
    # Delta MFCC
    delta = librosa.feature.delta(mfcc)
    delta_feats = np.hstack([delta.mean(axis=1), delta.std(axis=1)])

    return np.concatenate([mel_feats, mfcc_feats, delta_feats])

# 3) Caching setup
features_file = 'features_cached.pkl'
labels_file = 'labels_cached.pkl'

if os.path.exists(features_file) and os.path.exists(labels_file):
    print("Loading cached features...")
    features = joblib.load(features_file)
    labels = joblib.load(labels_file)
else:
    print("Extracting features from audio files...")
    features = np.vstack([extract_features(fp) for fp in df['audio_path']])
    labels = df['label'].values
    joblib.dump(features, features_file)
    joblib.dump(labels, labels_file)
    print("Features cached.")

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 5) Classifiers
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

#XGBOOST
# Optional: Compute scale_pos_weight to balance classes
class_counts = np.bincount(y_train)
scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1

# Initialize and train XGBoost model
xgb_clf = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_clf.predict(X_test)


# Initialize Logistic Regression
logreg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # handles QR vs non-QR imbalance
    random_state=42
)

# Train
logreg.fit(X_train, y_train)

# Predict
y_pred_logreg = logreg.predict(X_test)


lgb = LGBMClassifier(class_weight='balanced', random_state=42)
lgb.fit(X_train, y_train)
y_pred_lgb = lgb.predict(X_test)




mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

print("\n--- MLP (Neural Net) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.3f}")
print(classification_report(y_test, y_pred_mlp))


print("\n--- LightGBM ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.3f}")
print(classification_report(y_test, y_pred_lgb))


# Evaluate
print("\n--- Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.3f}")
print(classification_report(y_test, y_pred_logreg))


print("\n--- XGBoost ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}")
print(classification_report(y_test, y_pred_xgb))

# 6) Evaluation
print("\n--- Random Forest ---")
y_pred_rf = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(classification_report(y_test, y_pred_rf))

print("\n--- KNN ---")
y_pred_knn = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(classification_report(y_test, y_pred_knn))


import matplotlib.pyplot as plt

# 1) Train/Test Split Pie Chart
train_pct = 80
test_pct = 20

plt.figure()
plt.pie([train_pct, test_pct],
        labels=["Train (80%)", "Test (20%)"],
        autopct="%1.0f%%")
plt.title("Train/Test Split Distribution")
plt.show()

# 2) Model Accuracy Bar Chart
model_accuracies = {
    "MLP": 0.955,
    "LightGBM": 0.984,
    "Logistic Regression": 0.968,
    "XGBoost": 0.968,
    "Random Forest": 0.933,
    "KNN": 0.946
}

plt.figure()
plt.bar(model_accuracies.keys(), model_accuracies.values())
plt.ylim(0.80, 1.00)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison on Test Set")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

models = {
    "KNN": knn,
    "Random Forest": rf,
    "XGBoost": xgb_clf,
    "Logistic Regression": logreg,
    "LightGBM": lgb,
    "MLP": mlp
}

metrics = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics.append({
        "Model": name,
        "Accuracy": round(accuracy, 3),
        "Precision (QR)": round(report["1"]["precision"], 2),
        "Recall (QR)": round(report["1"]["recall"], 2),
        "F1-score (QR)": round(report["1"]["f1-score"], 2),
        "Precision (QL)": round(report["0"]["precision"], 2),
        "Recall (QL)": round(report["0"]["recall"], 2),
        "F1-score (QL)": round(report["0"]["f1-score"], 2)
    })

all_metrics = pd.DataFrame(metrics)

# Display nicely in Jupyter or print in terminal
try:
    from IPython.display import display
    display(all_metrics)
except ImportError:
    print(all_metrics.to_string(index=False))
