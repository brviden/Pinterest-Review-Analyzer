import os
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
    n_clusters = min(N_CLUSTERS, len(df))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    cluster_labels = {
        i: get_topic_label(km.cluster_centers_[i], feature_names)
        for i in range(n_clusters)
    }
    df = df.copy()
    df["topic"] = [cluster_labels[label] for label in km.labels_]
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/reviews.csv")
    print(f"Loaded {len(df)} reviews")
    print("Running sentiment analysis...")
    df = add_sentiment(df)
    print("Running topic clustering...")
    df = add_topics(df)
    df.to_csv("data/analyzed.csv", index=False)
    print(f"Saved to data/analyzed.csv")
    print(df[["store", "rating", "sentiment_label", "topic"]].head(5))
