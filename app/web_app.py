import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
from src.predict import FragrancePredictor

# Age bin → age range
AGE_BIN_RANGES = {
    "teen": "13-18",
    "early_adult": "19-25",
    "adult": "26-35",
    "mid_adult": "36-50",
    "mature": "51-65",
    "senior": "66-80"
}

AGE_BIN_LABELS = {
    "teen": "Teen",
    "early_adult": "Young Adult",
    "adult": "Adult",
    "mid_adult": "Mid Adult",
    "mature": "Mature Adult",
    "senior": "Senior"
}


# Load predictor once
@st.cache_resource
def load_predictor():
    return FragrancePredictor()

predictor = load_predictor()


# Basic bar chart visualiser
def prob_bar_chart(prob_dict, title):
    labels = list(prob_dict.keys())
    values = list(prob_dict.values())

    fig = go.Figure([
        go.Bar(
            x=labels,
            y=values,
            marker_color="#4b7bec"
        )
    ])
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )
    return fig


# Age gauge chart
def age_gauge(avg_age):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_age,
            title={"text": "Average Predicted Age"},
            gauge={"axis": {"range": [13, 80]}}
        )
    )
    fig.update_layout(template="plotly_white")
    return fig


# UI layout
st.set_page_config(page_title="FragranceMatch AI", layout="wide")

st.title("FragranceMatch AI")
st.write("Predict consumer personas and product-fit for fragrance descriptions.")


description = st.text_area(
    "Enter a fragrance description: Some examples: Men Dandruff Shampoo Menthol, peppermint, green tea and a clean aquatic freshness.",
    "Bright lemon zest, fresh linen, crisp green apple and clean cotton undertones.",
    height=120
)

if st.button("Predict"):

    results = predictor.predict(description)

    # Extract top variables:
    top_gender = results["top_gender"]
    top_country = results["top_country"]
    top_mood = results["top_mood"]
    top_product_fit = results["top_product_fit"]
    top_age_bin = results["top_age_bin"]

    top_age_range = AGE_BIN_RANGES[top_age_bin]
    top_age_label = AGE_BIN_LABELS[top_age_bin]

    # ───────────────────────────────────────────────
    # Persona Summary Panel
    # ───────────────────────────────────────────────
    st.markdown("## Persona Summary")

    persona_text = (
        f"This scent is most likely to appeal to **{top_age_label} consumers ({top_age_range})**, "
        f"primarily **{top_gender.lower()}** in **{top_country}**, who prefer **{top_mood.lower()}** fragrances. "
        f"This profile aligns strongly with **{top_product_fit}** product categories."
    )

    st.write(persona_text)
    st.write(f"Estimated average age: **{results['average_age']}** years")

    # Top summary metrics
    colA, colB, colC, colD = st.columns(4)

    with colA:
        st.metric("Gender", top_gender.title())

    with colB:
        st.metric("Age Group", top_age_range)

    with colC:
        st.metric("Country", top_country)

    with colD:
        st.metric("Product Fit", top_product_fit)

    st.markdown("---")

    # ───────────────────────────────────────────────
    # Gender Chart
    # ───────────────────────────────────────────────
    st.subheader(f"Gender — Most Likely: {top_gender.title()}")
    st.plotly_chart(
        prob_bar_chart(results["gender_probs"], "Gender Probability Distribution"),
        use_container_width=True
    )

    # Mood Chart
    st.subheader(f"Mood — Most Likely: {top_mood.title()}")
    st.plotly_chart(
        prob_bar_chart(results["mood_probs"], "Mood Probability Distribution"),
        use_container_width=True
    )

    # Country Chart
    st.subheader(f"Country — Most Likely: {top_country}")
    st.plotly_chart(
        prob_bar_chart(results["country_probs"], "Country Probability Distribution"),
        use_container_width=True
    )

    # Product Fit Chart (NEW)
    st.subheader(f"Product Fit — Most Likely: {top_product_fit}")
    st.plotly_chart(
        prob_bar_chart(results["product_fit_probs"], "Product Fit Probability Distribution"),
        use_container_width=True
    )

    # Age Group Chart
    st.subheader(f"Age Group — Most Likely: {top_age_range} ({top_age_label})")
    age_prob_dict_ranges = {
        AGE_BIN_RANGES[k]: v for k, v in results["age_bin_probs"].items()
    }
    st.plotly_chart(
        prob_bar_chart(age_prob_dict_ranges, "Age Group Probability Distribution"),
        use_container_width=True
    )

    # Average age gauge
    st.subheader("Average Age Estimate")
    st.plotly_chart(
        age_gauge(results["average_age"]),
        use_container_width=True
    )

    st.success("Prediction complete.")
