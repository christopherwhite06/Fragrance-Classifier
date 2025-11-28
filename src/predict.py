import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import os


# Age bin midpoint mapping for average age calculation
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
        print("Loading Sentence-BERT model...")
        self.embedder = SentenceTransformer("all-mpnet-base-v2")

        print("Loading ML models...")
        self.gender_clf = joblib.load(os.path.join(model_dir, "gender_clf.pkl"))
        self.mood_clf = joblib.load(os.path.join(model_dir, "mood_clf.pkl"))
        self.country_clf = joblib.load(os.path.join(model_dir, "country_clf.pkl"))
        self.product_fit_clf = joblib.load(os.path.join(model_dir, "product_fit_clf.pkl"))
        self.age_clf = joblib.load(os.path.join(model_dir, "age_bin_clf.pkl"))

        print("All models loaded successfully.")

    # Embed description → BERT embedding
    def embed(self, text):
        return self.embedder.encode([text])

    def predict(self, description):
        emb = self.embed(description)

        # Probability distributions for each classifier
        gender_probs = self.gender_clf.predict_proba(emb)[0]
        mood_probs = self.mood_clf.predict_proba(emb)[0]
        country_probs = self.country_clf.predict_proba(emb)[0]
        product_fit_probs = self.product_fit_clf.predict_proba(emb)[0]
        age_bin_probs = self.age_clf.predict_proba(emb)[0]

        # Convert to dicts
        gender_dict = dict(zip(self.gender_clf.classes_, gender_probs))
        mood_dict = dict(zip(self.mood_clf.classes_, mood_probs))
        country_dict = dict(zip(self.country_clf.classes_, country_probs))
        product_fit_dict = dict(zip(self.product_fit_clf.classes_, product_fit_probs))
        age_prob_dict = dict(zip(self.age_clf.classes_, age_bin_probs))

        # Compute average age from age-bin expected value
        avg_age = int(sum(
            AGE_BIN_MIDPOINTS[b] * p for b, p in age_prob_dict.items()
        ))

        # Most likely predictions
        top_gender = max(gender_dict, key=gender_dict.get)
        top_mood = max(mood_dict, key=mood_dict.get)
        top_country = max(country_dict, key=country_dict.get)
        top_product_fit = max(product_fit_dict, key=product_fit_dict.get)
        top_age_bin = max(age_prob_dict, key=age_prob_dict.get)

        return {
            "gender_probs": gender_dict,
            "mood_probs": mood_dict,
            "country_probs": country_dict,
            "product_fit_probs": product_fit_dict,
            "age_bin_probs": age_prob_dict,
            "average_age": avg_age,

            # Top predictions
            "top_gender": top_gender,
            "top_mood": top_mood,
            "top_country": top_country,
            "top_product_fit": top_product_fit,
            "top_age_bin": top_age_bin
        }


if __name__ == "__main__":
    model = FragrancePredictor()
    print(model.predict("Fresh lemon zest, icy mint, and clean eucalyptus."))
