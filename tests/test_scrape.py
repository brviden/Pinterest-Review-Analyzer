import datetime
from scrape import normalize_appstore_review, normalize_play_review

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
