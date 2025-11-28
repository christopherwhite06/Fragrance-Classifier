import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Define age bins
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

    df["age_bin"] = df["age"].apply(assign_age_bin)

    targets = {
        "gender": df["gender"],
        "mood": df["mood"],
        "country": df["country"],
        "age_bin": df["age_bin"]
    }

    for target_name, y in targets.items():
        print(f"\nTraining probabilistic classifier for: {target_name}")

        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial"
        )

        clf.fit(X, y)
        joblib.dump(clf, f"{models_dir}/{target_name}_clf.pkl")

        print(f"Saved {target_name} classifier.")

    print("\nTraining complete!")


if __name__ == "__main__":
    train_classifiers()