import pandas as pd
import random
import os


def generate_synthetic_dataset(
    output_path="data/fragrance_dataset.csv",
    num_samples=10000
):
    os.makedirs("data", exist_ok=True)

    # Scent building blocks — realistic components
    citrus_notes = [
        "lemon zest", "bergamot", "orange peel", "grapefruit", "lime", "mandarin"
    ]
    floral_notes = [
        "rose petals", "jasmine", "lavender", "chamomile", "peony", "orchid"
    ]
    fresh_notes = [
        "clean eucalyptus", "green tea", "fresh linen", "icy mint", "ocean breeze"
    ]
    sweet_notes = [
        "vanilla", "tonka bean", "caramel", "sugar musk", "coconut cream"
    ]
    woody_notes = [
        "sandalwood", "cedarwood", "smoky spice", "amberwood", "vetiver"
    ]
    herbal_notes = [
        "rosemary", "sage", "basil", "thyme", "mint herb", "lemongrass"
    ]
    powdery_notes = [
        "baby powder", "soft cotton", "white musk", "powdery floral"
    ]

    # Combine all families for random selection
    all_note_groups = [
        citrus_notes, floral_notes, fresh_notes,
        sweet_notes, woody_notes, herbal_notes, powdery_notes
    ]

    # Expanded global country list
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
    
    rows = []

    for i in range(num_samples):

        # Pick 3 note families and random notes inside them
        chosen_groups = random.sample(all_note_groups, 3)
        notes = [random.choice(group) for group in chosen_groups]

        description = ", ".join(notes)

        rows.append({
            "id": i,
            "description": description,

            # Single gender value
            "gender": random.choice(genders),

            # Specific age
            "age": random.randint(13, 80),

            # Specific country
            "country": random.choice(countries),

            # Mood/Emotion
            "mood": random.choice(moods)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Generated dataset with {num_samples} rows -> {output_path}")


if __name__ == "__main__":
    generate_synthetic_dataset()