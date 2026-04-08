import pathlib
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Pinterest Review Analyzer",
    page_icon="📌",
    layout="wide",
)

# Custom font injection — Syne for headings, DM Sans for body
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stDataFrame, .stMetric label {
    font-family: 'DM Sans', sans-serif !important;
}
h1, h2, h3, .stMetric .metric-label {
    font-family: 'Syne', sans-serif !important;
}
.block-container { padding-top: 2rem; }
div[data-testid="metric-container"] {
    background-color: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 8px;
    padding: 1rem 1.5rem;
}
</style>
""", unsafe_allow_html=True)


_DATA_PATH = pathlib.Path(__file__).parent / "data" / "analyzed.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH, parse_dates=["date"])
    return df


def main():
    st.title("📌 Pinterest Review Analyzer")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("No analyzed data found. Run `python scrape.py` then `python analyze.py` first.")
        return

    st.caption(f"{len(df):,} reviews loaded · {df['store'].nunique()} stores · {df['date'].min().date()} – {df['date'].max().date()}")

    # ── Sidebar filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        stores = ["All"] + sorted(df["store"].unique().tolist())
        selected_store = st.selectbox("Store", stores)

        ratings = sorted(df["rating"].unique().tolist())
        selected_ratings = st.multiselect("Star Rating", ratings, default=ratings)

        sentiments = ["All", "positive", "neutral", "negative"]
        selected_sentiment = st.selectbox("Sentiment", sentiments)

        topics = ["All"] + sorted(df["topic"].unique().tolist())
        selected_topic = st.selectbox("Topic", topics)

        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # ── Apply filters ────────────────────────────────────────────────────────
    filtered = df.copy()
    if selected_store != "All":
        filtered = filtered[filtered["store"] == selected_store]
    if selected_ratings:
        filtered = filtered[filtered["rating"].isin(selected_ratings)]
    if selected_sentiment != "All":
        filtered = filtered[filtered["sentiment_label"] == selected_sentiment]
    if selected_topic != "All":
        filtered = filtered[filtered["topic"] == selected_topic]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["date"].dt.date >= date_range[0]) &
            (filtered["date"].dt.date <= date_range[1])
        ]

    if filtered.empty:
        st.warning("No reviews match the current filters.")
        return

    # ── Panels (imported from render functions below) ─────────────────────────
    render_metrics(filtered)
    st.divider()
    render_sentiment_over_time(filtered)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        render_rating_distribution(filtered)
    with col2:
        render_topic_breakdown(filtered)
    st.divider()
    render_review_table(filtered)


def render_metrics(df: pd.DataFrame):
    total = len(df)
    avg_rating = df["rating"].mean()
    pct_pos = (df["sentiment_label"] == "positive").sum() / total * 100
    pct_neg = (df["sentiment_label"] == "negative").sum() / total * 100
    pct_neu = (df["sentiment_label"] == "neutral").sum() / total * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Avg Rating", f"{avg_rating:.2f} ★")
    c3.metric("Positive", f"{pct_pos:.1f}%")
    c4.metric("Neutral", f"{pct_neu:.1f}%")
    c5.metric("Negative", f"{pct_neg:.1f}%")


def render_sentiment_over_time(df: pd.DataFrame):
    st.subheader("Sentiment Over Time")
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )
    color_map = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#E60023"}
    fig = px.line(
        monthly,
        x="month",
        y="count",
        color="sentiment_label",
        color_discrete_map=color_map,
        labels={"month": "Month", "count": "Reviews", "sentiment_label": "Sentiment"},
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        legend_title_text="Sentiment",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)


def render_rating_distribution(df: pd.DataFrame):
    st.subheader("Rating Distribution")
    counts = df.groupby(["rating", "store"]).size().reset_index(name="count")
    fig = px.bar(
        counts,
        x="rating",
        y="count",
        color="store",
        barmode="group",
        color_discrete_map={"App Store": "#4A90D9", "Google Play": "#E60023"},
        labels={"rating": "Stars", "count": "Reviews", "store": "Store"},
        template="plotly_dark",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(tickmode="array", tickvals=[1, 2, 3, 4, 5]),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_topic_breakdown(df: pd.DataFrame):
    st.subheader("Topic Breakdown")
    dominant = (
        df.groupby("topic")["sentiment_label"]
        .agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else "neutral")
        .reset_index(name="dominant_sentiment")
    )
    counts = df.groupby("topic").size().reset_index(name="count")
    topic_df = counts.merge(dominant, on="topic").sort_values("count", ascending=True)
    color_map = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#E60023"}
    fig = px.bar(
        topic_df,
        x="count",
        y="topic",
        color="dominant_sentiment",
        color_discrete_map=color_map,
        orientation="h",
        labels={"count": "Reviews", "topic": "Topic", "dominant_sentiment": "Dominant Sentiment"},
        template="plotly_dark",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_review_table(df: pd.DataFrame):
    st.subheader("Reviews")
    display_cols = ["date", "store", "rating", "sentiment_label", "sentiment_score", "topic", "title", "review"]
    st.dataframe(
        df[display_cols].sort_values("date", ascending=False),
        use_container_width=True,
        height=400,
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "rating": st.column_config.NumberColumn("★", format="%d ★"),
            "sentiment_score": st.column_config.NumberColumn("Score", format="%.3f"),
            "sentiment_label": "Sentiment",
            "topic": "Topic",
            "store": "Store",
            "title": "Title",
            "review": st.column_config.TextColumn("Review", width="large"),
        },
    )


if __name__ == "__main__":
    main()
