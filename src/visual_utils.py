import plotly.graph_objects as go


def one_hot_bar(selected_label: str, all_labels: list, title: str):
    values = [1 if lbl == selected_label else 0 for lbl in all_labels]

    fig = go.Figure([go.Bar(
        x=all_labels,
        y=values,
        marker_color=["#2e86de" if v else "#ced6e0" for v in values]
    )])

    fig.update_layout(title=title, yaxis=dict(range=[0, 1]), template="plotly_white")
    return fig


def gender_pie(selected_label: str):
    labels = ["male", "female", "unisex"]
    values = [1 if lbl == selected_label else 0 for lbl in labels]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title="Gender Prediction", template="plotly_white")
    return fig


def mood_radar(selected: str, labels: list):
    values = [1 if lbl == selected else 0 for lbl in labels]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor="rgba(46,134,222,0.3)",
        line_color="rgba(46,134,222,1)"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Mood Profile: {selected}",
        template="plotly_white"
    )
    return fig


def age_gauge(age_value: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=age_value,
        gauge={"axis": {"range": [13, 80]}},
        title={"text": "Predicted Age"}
    ))
    fig.update_layout(template="plotly_white")
    return fig
