import pandas as pd
import random
import os


def generate_synthetic_dataset(
    output_path="data/fragrance_dataset.csv",
    num_samples=10000
):
    os.makedirs("data", exist_ok=True)

    # Scent blocks
    citrus_notes = ["lemon zest", "bergamot", "orange peel", "grapefruit", "lime", "mandarin"]
    floral_notes = ["rose petals", "jasmine", "lavender", "chamomile", "peony", "orchid"]
    fresh_notes = ["clean eucalyptus", "green tea", "fresh linen", "icy mint", "ocean breeze"]
    sweet_notes = ["vanilla", "tonka bean", "caramel", "sugar musk", "coconut cream"]
    woody_notes = ["sandalwood", "cedarwood", "smoky spice", "amberwood", "vetiver"]
    herbal_notes = ["rosemary", "sage", "basil", "thyme", "mint herb", "lemongrass"]
    powdery_notes = ["baby powder", "soft cotton", "white musk", "powdery floral"]

    all_note_groups = [
        citrus_notes, floral_notes, fresh_notes,
        sweet_notes, woody_notes, herbal_notes, powdery_notes
    ]

    countries = [
        "United Kingdom", "United States", "Canada", "Brazil", "Mexico",
        "France", "Germany", "Spain", "Italy", "Sweden",
        "Australia", "New Zealand", "Japan", "South Korea", "China",
        "India", "Pakistan", "Bangladesh", "Nigeria", "South Africa",
        "Saudi Arabia", "UAE", "Turkey", "Egypt", "Vietnam",
        "Malaysia", "Singapore", "Philippines", "Indonesia", "Thailand"
    ]

    genders = ["male", "female", "unisex"]
    moods = ["energising", "calming", "warm", "fresh", "sporty", "gentle"]

    # NEW: P&G Product Categories
    product_categories = [
        "Laundry Fresh (Ariel/Tide)",
        "Baby Care (Pampers)",
        "Men’s Grooming (Gillette)",
        "Fresh Menthol Shampoo (Head & Shoulders)",
        "Warm Floral Skin Care (Olay)",
        "Home Freshening (Febreze)",
        "Dishwashing Fresh Lemon (Fairy)"
    ]

    rows = []

    for i in range(num_samples):

        # Pick 3 scent clusters
        chosen_groups = random.sample(all_note_groups, 3)
        notes = [random.choice(group) for group in chosen_groups]
        description = ", ".join(notes)

        # Assign product fit based on scents
        if any(n in description for n in citrus_notes):
            product_fit = "Laundry Fresh (Ariel/Tide)"
            if "lemon" in description or "lime" in description:
                product_fit = "Dishwashing Fresh Lemon (Fairy)"
        elif any(n in description for n in powdery_notes):
            product_fit = "Baby Care (Pampers)"
        elif any(n in description for n in fresh_notes):
            product_fit = "Fresh Menthol Shampoo (Head & Shoulders)"
        elif any(n in description for n in floral_notes):
            product_fit = "Warm Floral Skin Care (Olay)"
        elif any(n in description for n in woody_notes):
            product_fit = "Men’s Grooming (Gillette)"
        else:
            product_fit = random.choice(product_categories)

        rows.append({
            "id": i,
            "description": description,
            "gender": random.choice(genders),
            "age": random.randint(13, 80),
            "country": random.choice(countries),
            "mood": random.choice(moods),
            "product_fit": product_fit
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Generated enhanced dataset with {num_samples} rows → {output_path}")


if __name__ == "__main__":
    generate_synthetic_dataset()
