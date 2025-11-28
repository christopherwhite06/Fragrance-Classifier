import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Age group binning rules
def assign_age_bin(age):
    if age <= 18: return "teen"
    if age <= 25: return "early_adult"
    if age <= 35: return "adult"
    if age <= 50: return "mid_adult"
    if age <= 65: return "mature"
    return "senior"


def train_classifiers(
    dataset_path="data/fragrance_dataset.csv",
    embeddings_path="embeddings/desc_embeddings.npy",
    models_dir="models"
):
    os.makedirs(models_dir, exist_ok=True)

    print("Loading dataset and embeddings...")
    df = pd.read_csv(dataset_path)
    X = np.load(embeddings_path)

    # Ensure dataset and embeddings match size
    if len(df) != len(X):
        raise ValueError(
            f"ERROR: Dataset rows = {len(df)}, embeddings = {len(X)}.\n"
            f"Run generate_dataset.py and embedder.py again."
        )

    df["age_bin"] = df["age"].apply(assign_age_bin)

    classification_targets = {
        "gender": df["gender"],
        "mood": df["mood"],
        "country": df["country"],
        "product_fit": df["product_fit"],   # NEW
        "age_bin": df["age_bin"]
    }

    for target_name, y in classification_targets.items():
        print(f"\nTraining classifier for: {target_name}")

        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial"
        )

        clf.fit(X, y)
        joblib.dump(clf, f"{models_dir}/{target_name}_clf.pkl")

        print(f"Saved classifier: {target_name}_clf.pkl")

    print("\n✓ Training complete. All classifiers updated.")
    print("New model added: product_fit_clf.pkl")


if __name__ == "__main__":
    train_classifiers()
