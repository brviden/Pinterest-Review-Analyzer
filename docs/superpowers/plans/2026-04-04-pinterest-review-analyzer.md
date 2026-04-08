# Pinterest Review Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-script Python pipeline that scrapes Pinterest reviews from both app stores, runs VADER sentiment + TF-IDF topic clustering, and displays results in a dark-themed interactive Streamlit dashboard.

**Architecture:** Three sequential scripts — `scrape.py` writes raw reviews to `data/reviews.csv`, `analyze.py` enriches them into `data/analyzed.csv`, and `app.py` reads the analyzed CSV and renders a filterable Streamlit dashboard. No database; re-running `scrape.py` replaces the CSV.

**Tech Stack:** Python 3.10+, `google-play-scraper`, `app-store-scraper`, `vaderSentiment`, `scikit-learn`, `streamlit`, `pandas`, `plotly`, `pytest`

---

## File Map

| File | Responsibility |
|------|---------------|
| `scrape.py` | Fetch reviews from App Store + Google Play, normalize to common schema, write `data/reviews.csv` |
| `analyze.py` | Read `data/reviews.csv`, add VADER sentiment + TF-IDF/KMeans topic columns, write `data/analyzed.csv` |
| `app.py` | Streamlit dashboard — load `data/analyzed.csv`, sidebar filters, 5 panels + review table |
| `.streamlit/config.toml` | Dark theme: Pinterest red accent, DM Sans font |
| `requirements.txt` | Pinned runtime dependencies |
| `.gitignore` | Ignore `data/` CSVs |
| `tests/test_analyze.py` | Unit tests for sentiment scoring and topic labeling functions |
| `tests/test_scrape.py` | Unit tests for review normalization functions |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `.gitignore`
- Create: `data/.gitkeep`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
google-play-scraper==1.2.7
app-store-scraper==0.3.5
vaderSentiment==3.3.2
scikit-learn==1.4.2
streamlit==1.35.0
pandas==2.2.2
plotly==5.22.0
pytest==8.2.2
```

- [ ] **Step 2: Create `.streamlit/config.toml`**

```toml
[theme]
base = "dark"
backgroundColor = "#0F0F0F"
secondaryBackgroundColor = "#1A1A1A"
primaryColor = "#E60023"
textColor = "#F5F5F5"
font = "sans serif"
```

- [ ] **Step 3: Create `.gitignore`**

```
data/reviews.csv
data/analyzed.csv
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create empty `data/.gitkeep` and `tests/__init__.py`**

Create `data/.gitkeep` (empty file so the `data/` directory is tracked).
Create `tests/__init__.py` (empty).

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without error. Verify with:
```bash
python -c "import streamlit, pandas, plotly, vaderSentiment, sklearn; print('OK')"
```
Expected output: `OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .streamlit/config.toml .gitignore data/.gitkeep tests/__init__.py
git commit -m "chore: project setup — deps, theme, gitignore"
```

---

## Task 2: Review Normalization (scrape.py foundation + tests)

**Files:**
- Create: `scrape.py`
- Create: `tests/test_scrape.py`

The two stores return differently-shaped dicts. This task extracts the normalization logic into two pure functions that can be unit-tested independently of network calls.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scrape.py`:

```python
from scrape import normalize_appstore_review, normalize_play_review
import datetime

def test_normalize_appstore_review():
    raw = {
        "id": "abc123",
        "date": datetime.datetime(2024, 3, 1, 12, 0, 0),
        "rating": 4,
        "title": "Great app",
        "review": "Love it",
    }
    result = normalize_appstore_review(raw)
    assert result == {
        "id": "abc123",
        "store": "App Store",
        "date": "2024-03-01",
        "rating": 4,
        "title": "Great app",
        "review": "Love it",
    }

def test_normalize_play_review():
    raw = {
        "reviewId": "xyz789",
        "at": datetime.datetime(2024, 6, 15, 9, 0, 0),
        "score": 2,
        "thumbsUpCount": 5,
        "content": "Crashes constantly",
    }
    result = normalize_play_review(raw)
    assert result == {
        "id": "xyz789",
        "store": "Google Play",
        "date": "2024-06-15",
        "rating": 2,
        "title": "",
        "review": "Crashes constantly",
    }

def test_normalize_appstore_review_missing_title():
    raw = {
        "id": "no_title",
        "date": datetime.datetime(2024, 1, 1),
        "rating": 3,
        "title": None,
        "review": "It's okay",
    }
    result = normalize_appstore_review(raw)
    assert result["title"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scrape.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `scrape.py` doesn't exist yet.

- [ ] **Step 3: Create `scrape.py` with normalization functions**

```python
import datetime
import pandas as pd

REVIEW_COUNT = 500
APPSTORE_APP_ID = "429047995"
PLAY_PACKAGE_ID = "com.pinterest"


def normalize_appstore_review(raw: dict) -> dict:
    return {
        "id": str(raw["id"]),
        "store": "App Store",
        "date": raw["date"].strftime("%Y-%m-%d") if isinstance(raw["date"], datetime.datetime) else str(raw["date"])[:10],
        "rating": int(raw["rating"]),
        "title": raw.get("title") or "",
        "review": raw.get("review") or "",
    }


def normalize_play_review(raw: dict) -> dict:
    return {
        "id": str(raw["reviewId"]),
        "store": "Google Play",
        "date": raw["at"].strftime("%Y-%m-%d") if isinstance(raw["at"], datetime.datetime) else str(raw["at"])[:10],
        "rating": int(raw["score"]),
        "title": "",
        "review": raw.get("content") or "",
    }


def fetch_appstore_reviews() -> list[dict]:
    from app_store_scraper import AppStore
    app = AppStore(country="us", app_name="pinterest", app_id=APPSTORE_APP_ID)
    app.review(how_many=REVIEW_COUNT)
    return [normalize_appstore_review(r) for r in app.reviews]


def fetch_play_reviews() -> list[dict]:
    from google_play_scraper import reviews, Sort
    result, _ = reviews(
        PLAY_PACKAGE_ID,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=REVIEW_COUNT,
    )
    return [normalize_play_review(r) for r in result]


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    print("Fetching App Store reviews...")
    appstore = fetch_appstore_reviews()
    print(f"  Got {len(appstore)} reviews")
    print("Fetching Google Play reviews...")
    play = fetch_play_reviews()
    print(f"  Got {len(play)} reviews")
    df = pd.DataFrame(appstore + play)
    df.to_csv("data/reviews.csv", index=False)
    print(f"Saved {len(df)} total reviews to data/reviews.csv")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scrape.py -v
```

Expected:
```
tests/test_scrape.py::test_normalize_appstore_review PASSED
tests/test_scrape.py::test_normalize_play_review PASSED
tests/test_scrape.py::test_normalize_appstore_review_missing_title PASSED
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add scrape.py tests/test_scrape.py
git commit -m "feat: scrape.py normalization functions with tests"
```

---

## Task 3: Live Scraping Verification

**Files:**
- No new files — verify `scrape.py` end-to-end

- [ ] **Step 1: Run the scraper**

```bash
python scrape.py
```

Expected output:
```
Fetching App Store reviews...
  Got 500 reviews
Fetching Google Play reviews...
  Got 500 reviews
Saved 1000 total reviews to data/reviews.csv
```

Note: Counts may be slightly lower if fewer reviews are available. Any output above 200 per store is acceptable.

- [ ] **Step 2: Verify the CSV schema**

```bash
python -c "import pandas as pd; df = pd.read_csv('data/reviews.csv'); print(df.dtypes); print(df.head(2))"
```

Expected: Columns `id, store, date, rating, title, review` all present. `rating` is int64. Both `App Store` and `Google Play` appear in the `store` column.

- [ ] **Step 3: Commit**

```bash
git add scrape.py
git commit -m "feat: scrape.py live fetch verified"
```

---

## Task 4: Sentiment Analysis (analyze.py foundation + tests)

**Files:**
- Create: `analyze.py`
- Create: `tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analyze.py`:

```python
from analyze import score_sentiment

def test_positive_review():
    label, score = score_sentiment("I absolutely love this app, it's amazing!")
    assert label == "positive"
    assert score >= 0.05

def test_negative_review():
    label, score = score_sentiment("This app crashes constantly and is broken garbage.")
    assert label == "negative"
    assert score <= -0.05

def test_neutral_review():
    label, score = score_sentiment("The app.")
    assert label == "neutral"
    assert -0.05 < score < 0.05

def test_empty_review():
    label, score = score_sentiment("")
    assert label == "neutral"
    assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_analyze.py -v
```

Expected: `ImportError` — `analyze.py` doesn't exist yet.

- [ ] **Step 3: Create `analyze.py` with sentiment function**

```python
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

_analyzer = SentimentIntensityAnalyzer()
N_CLUSTERS = 8


def score_sentiment(text: str) -> tuple[str, float]:
    """Return (label, compound_score) for a review string."""
    if not text or not text.strip():
        return "neutral", 0.0
    compound = _analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, round(compound, 4)


def get_topic_label(cluster_center, feature_names: list[str]) -> str:
    """Return top-3 keywords joined by ' · ' for a KMeans cluster center."""
    top_indices = cluster_center.argsort()[-3:][::-1]
    return " · ".join(feature_names[i] for i in top_indices)


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    results = df["review"].fillna("").apply(score_sentiment)
    df = df.copy()
    df["sentiment_label"] = results.apply(lambda x: x[0])
    df["sentiment_score"] = results.apply(lambda x: x[1])
    return df


def add_topics(df: pd.DataFrame) -> pd.DataFrame:
    texts = df["review"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", min_df=2)
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    cluster_labels = {
        i: get_topic_label(km.cluster_centers_[i], feature_names)
        for i in range(N_CLUSTERS)
    }
    df = df.copy()
    df["topic"] = [cluster_labels[label] for label in km.labels_]
    return df


if __name__ == "__main__":
    import os
    df = pd.read_csv("data/reviews.csv")
    print(f"Loaded {len(df)} reviews")
    print("Running sentiment analysis...")
    df = add_sentiment(df)
    print("Running topic clustering...")
    df = add_topics(df)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/analyzed.csv", index=False)
    print(f"Saved to data/analyzed.csv")
    print(df[["store", "rating", "sentiment_label", "topic"]].head(5))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_analyze.py -v
```

Expected:
```
tests/test_analyze.py::test_positive_review PASSED
tests/test_analyze.py::test_negative_review PASSED
tests/test_analyze.py::test_neutral_review PASSED
tests/test_analyze.py::test_empty_review PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "feat: analyze.py sentiment scoring with tests"
```

---

## Task 5: Topic Clustering Tests + Full Pipeline Verification

**Files:**
- Modify: `tests/test_analyze.py`

- [ ] **Step 1: Add topic labeling tests to `tests/test_analyze.py`**

Append to the existing file:

```python
import numpy as np
from analyze import get_topic_label

def test_get_topic_label_returns_three_words():
    feature_names = np.array(["crash", "load", "freeze", "ads", "login", "slow"])
    center = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7])
    label = get_topic_label(center, feature_names)
    assert label == "load · freeze · slow"

def test_get_topic_label_format():
    feature_names = np.array(["alpha", "beta", "gamma"])
    center = np.array([0.3, 0.1, 0.9])
    label = get_topic_label(center, feature_names)
    parts = label.split(" · ")
    assert len(parts) == 3
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: All 6 tests pass.

- [ ] **Step 3: Run the full pipeline end-to-end**

```bash
python analyze.py
```

Expected:
```
Loaded 1000 reviews
Running sentiment analysis...
Running topic clustering...
Saved to data/analyzed.csv
```

Followed by a 5-row preview with `store`, `rating`, `sentiment_label`, and `topic` columns populated.

- [ ] **Step 4: Verify analyzed CSV**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/analyzed.csv')
print('Columns:', df.columns.tolist())
print('Sentiment counts:', df['sentiment_label'].value_counts().to_dict())
print('Topic sample:', df['topic'].unique()[:4])
"
```

Expected: 8 unique topic strings in `topic · word · word` format.

- [ ] **Step 5: Commit**

```bash
git add tests/test_analyze.py
git commit -m "feat: topic clustering tests + full pipeline verified"
```

---

## Task 6: Streamlit App — Theme, Fonts, and Data Loading

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create `app.py` with theme injection and data loading**

```python
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


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/analyzed.csv", parse_dates=["date"])
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
        color_discrete_sequence=["#E60023", "#4A90D9"],
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
        .agg(lambda x: x.value_counts().index[0])
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
```

- [ ] **Step 2: Launch the app to verify it loads**

```bash
streamlit run app.py
```

Expected: Browser opens to `http://localhost:8501`. Dashboard title "📌 Pinterest Review Analyzer" visible. All 5 metric cards, charts, and review table render. Sidebar filters update all panels when changed.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: streamlit dashboard — filters, metrics, charts, table"
```

---

## Task 7: Run Full Test Suite

**Files:** No changes — verification only.

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected:
```
tests/test_analyze.py::test_positive_review PASSED
tests/test_analyze.py::test_negative_review PASSED
tests/test_analyze.py::test_neutral_review PASSED
tests/test_analyze.py::test_empty_review PASSED
tests/test_analyze.py::test_get_topic_label_returns_three_words PASSED
tests/test_analyze.py::test_get_topic_label_format PASSED
tests/test_scrape.py::test_normalize_appstore_review PASSED
tests/test_scrape.py::test_normalize_play_review PASSED
tests/test_scrape.py::test_normalize_appstore_review_missing_title PASSED
9 passed
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "chore: all 9 tests passing — pipeline complete"
```

---

## Usage Reference

```bash
# Refresh data (overwrites existing CSVs)
python scrape.py

# Re-run analysis (after scrape or to change cluster count)
python analyze.py

# Launch dashboard
streamlit run app.py
```

To change the number of reviews fetched, edit `REVIEW_COUNT` at the top of `scrape.py`.
To change the number of topic clusters, edit `N_CLUSTERS` at the top of `analyze.py`.
