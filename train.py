import os
import pickle
# feature/add-model-training

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

print("==> Chargement du dataset Mushroom...")
df = pd.read_csv("mushrooms.csv")
print(f"    Taille : {df.shape[0]} lignes, {df.shape[1]} colonnes")

if "class" not in df.columns:
    raise ValueError("La colonne cible 'class' est absente du fichier mushrooms.csv")

df = df.drop_duplicates()

X = df.drop(columns=["class"])
y = df["class"].map({"e": "edible", "p": "poisonous"})

if y.isna().any():
    raise ValueError("La cible contient des valeurs inattendues. Attendu : e ou p.")

categorical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

print("\n==> Séparation train / test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n==> Entraînement du modèle...")
model.fit(X_train, y_train)

print("\n==> Évaluation du modèle...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label="poisonous")

print(f"\n    Accuracy : {acc:.4f}")
print(f"    F1-score : {f1:.4f}")
print("\n    Rapport détaillé :")
print(classification_report(y_test, y_pred))

os.makedirs("artifacts", exist_ok=True)

artifact = {
    "model": model,
    "feature_names": list(X.columns),
    "target_names": ["edible", "poisonous"],
}

with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n==> Artefact sauvegardé dans artifacts/model.pkl")
print("==> Entraînement terminé.")