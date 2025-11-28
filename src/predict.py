import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import os


AGE_BIN_MIDPOINTS = {
    "teen": 16,
    "early_adult": 22,
    "adult": 30,
    "mid_adult": 43,
    "mature": 58,
    "senior": 72
}

class FragrancePredictor:
    def __init__(self, model_dir="models"):

        self.embedder = SentenceTransformer("all-mpnet-base-v2")

        self.gender_clf = joblib.load(os.path.join(model_dir, "gender_clf.pkl"))
        self.country_clf = joblib.load(os.path.join(model_dir, "country_clf.pkl"))
        self.mood_clf = joblib.load(os.path.join(model_dir, "mood_clf.pkl"))
        self.age_clf = joblib.load(os.path.join(model_dir, "age_bin_clf.pkl"))

    def embed(self, text):
        return self.embedder.encode([text])

    def predict(self, description):
        emb = self.embed(description)

        # get probability distributions
        gender_probs = self.gender_clf.predict_proba(emb)[0]
        mood_probs = self.mood_clf.predict_proba(emb)[0]
        country_probs = self.country_clf.predict_proba(emb)[0]
        age_bin_probs = self.age_clf.predict_proba(emb)[0]

        # map age bin distribution
        age_prob_dict = {
            cls: float(prob)
            for cls, prob in zip(self.age_clf.classes_, age_bin_probs)
        }

        # compute weighted average age
        avg_age = sum(
            AGE_BIN_MIDPOINTS[cls] * prob
            for cls, prob in age_prob_dict.items()
        )

        # most likely country
        top_country = self.country_clf.classes_[np.argmax(country_probs)]

        # most likely age bin
        top_age_bin = self.age_clf.classes_[np.argmax(age_bin_probs)]

        return {
            "gender_probs": dict(zip(self.gender_clf.classes_, gender_probs)),
            "mood_probs": dict(zip(self.mood_clf.classes_, mood_probs)),
            "country_probs": dict(zip(self.country_clf.classes_, country_probs)),
            "age_bin_probs": age_prob_dict,
            "average_age": int(avg_age),
            "top_age_bin": top_age_bin,
            "top_country": top_country
        }


if __name__ == "__main__":
    model = FragrancePredictor()
    print(model.predict("Fresh lemon with icy mint and clean eucalyptus"))
