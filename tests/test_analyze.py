import numpy as np
from analyze import score_sentiment, get_topic_label

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

def test_get_topic_label_returns_three_words():
    feature_names = np.array(["crash", "load", "freeze", "ads", "login", "slow"])
    center = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7])
    label = get_topic_label(center, feature_names)
    assert label == "load · freeze · slow"

def test_get_topic_label_format():
    feature_names = np.array(["alpha", "beta", "gamma"])
    center = np.array([0.3, 0.1, 0.9])
    label = get_topic_label(center, feature_names)
    assert label == "gamma · alpha · beta"
