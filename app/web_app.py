import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
from src.predict import FragrancePredictor


AGE_BIN_RANGES = {
    "teen": "13–18",
    "early_adult": "19–25",
    "adult": "26–35",
    "mid_adult": "36–50",
    "mature": "51–65",
    "senior": "66–80"
}

@st.cache_resource
def load_predictor():
    return FragrancePredictor()

predictor = load_predictor()

def prob_bar_chart(prob_dict, title):
    labels = list(prob_dict.keys())
    values = list(prob_dict.values())

    fig = go.Figure([go.Bar(
        x=labels,
        y=values,
        marker_color="#4b7bec"
    )])
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )
    return fig


def age_gauge(avg_age):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_age,
        title={"text": "Average Predicted Age"},
        gauge={"axis": {"range": [13, 80]}}
    ))
    fig.update_layout(template="plotly_white")
    return fig


# === UI ===

st.set_page_config(page_title="FragranceMatch AI", layout="wide")

st.title("🧪 FragranceMatch AI")
st.write("Predict consumer demographics and scent personas using fragrance descriptions.")


description = st.text_area(
    "Enter a fragrance description:",
    "Fresh lemon zest, icy mint, and clean eucalyptus.",
    height=120
)


if st.button("Predict"):

    results = predictor.predict(description)

    st.markdown("## Results Overview")

    colA, colB, colC = st.columns(3)

    with colA:
        top_gender = max(results["gender_probs"], key=results["gender_probs"].get)
        st.metric("Top Gender", top_gender.title())

    with colB:
        top_country = results["top_country"]
        st.metric("Top Country", top_country)

    with colC:
        top_age_bin = results["top_age_bin"]
        st.metric("Top Age Group", AGE_BIN_RANGES[top_age_bin])

    st.markdown("---")

    st.subheader(f"Gender — Most Likely: **{top_gender.title()}**")
    st.plotly_chart(
        prob_bar_chart(results["gender_probs"], "Gender Distribution"),
        use_container_width=True
    )

    # === Mood ===
    top_mood = max(results["mood_probs"], key=results["mood_probs"].get)
    st.subheader(f"Mood — Most Likely: **{top_mood.title()}**")
    st.plotly_chart(
        prob_bar_chart(results["mood_probs"], "Mood Distribution"),
        use_container_width=True
    )

    # === Country ===
    st.subheader(f"Country — Most Likely: **{top_country}**")
    st.plotly_chart(
        prob_bar_chart(results["country_probs"], "Country Distribution"),
        use_container_width=True
    )

    st.subheader(f"Age Group — Most Likely: **{AGE_BIN_RANGES[top_age_bin]}**")

    age_prob_dict_ranges = {
        AGE_BIN_RANGES[k]: v for k, v in results["age_bin_probs"].items()
    }

    st.plotly_chart(
        prob_bar_chart(age_prob_dict_ranges, "Age Group Probability Distribution"),
        use_container_width=True
    )

    st.subheader("Average Estimated Age")
    st.plotly_chart(
        age_gauge(results["average_age"]),
        use_container_width=True
    )

    st.success("Prediction complete!")
