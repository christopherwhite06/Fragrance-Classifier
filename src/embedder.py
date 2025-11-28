import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_embeddings(
    dataset_path="data/fragrance_dataset.csv",
    embeddings_out="embeddings/desc_embeddings.npy",
    texts_out="embeddings/desc_texts.json"
):
    """
    Loads fragrance descriptions, generates embeddings using Sentence-BERT,
    and saves the embeddings + text list.
    """

    os.makedirs("embeddings", exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    if "description" not in df.columns:
        raise ValueError("Dataset must contain a 'description' column.")

    descriptions = df["description"].tolist()

    # Load embedding model
    print("Loading Sentence-BERT model (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(descriptions, show_progress_bar=True)

    # Save outputs
    np.save(embeddings_out, embeddings)
    with open(texts_out, "w") as f:
        json.dump(descriptions, f)

    print(f"Embeddings saved to {embeddings_out}")
    print(f"Description list saved to {texts_out}")
    print("Embedding generation complete.")

if __name__ == "__main__":
    generate_embeddings()
