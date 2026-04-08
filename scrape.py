import datetime
import requests

REVIEW_COUNT = 500
APPSTORE_APP_ID = "429047995"
PLAY_PACKAGE_ID = "com.pinterest"


def normalize_appstore_review(raw: dict) -> dict:
    return {
        "id": str(raw["id"]),
        "store": "App Store",
        "date": raw["date"].strftime("%Y-%m-%d"),
        "rating": int(raw["rating"]),
        "title": raw.get("title") or "",
        "review": raw.get("review") or "",
    }


def normalize_play_review(raw: dict) -> dict:
    return {
        "id": str(raw["reviewId"]),
        "store": "Google Play",
        "date": raw["at"].strftime("%Y-%m-%d"),
        "rating": int(raw["score"]),
        "title": "",
        "review": raw.get("content") or "",
    }


def _fetch_appstore_itunes_rss() -> list[dict]:
    """Fallback: fetch App Store reviews via the public iTunes RSS feed."""
    results = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, 11):  # up to 10 pages × 50 = 500 reviews
        url = (
            f"https://itunes.apple.com/us/rss/customerreviews"
            f"/page={page}/id={APPSTORE_APP_ID}/sortby=mostrecent/json"
        )
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        entries = r.json()["feed"].get("entry", [])
        for e in entries:
            if "im:rating" not in e:   # skip the app-level entry on page 1
                continue
            results.append({
                "id": str(e["id"]["label"]),
                "store": "App Store",
                "date": e["updated"]["label"][:10],
                "rating": int(e["im:rating"]["label"]),
                "title": e.get("title", {}).get("label", ""),
                "review": e.get("content", {}).get("label", ""),
            })
        if len(entries) < 50:
            break
        if len(results) >= REVIEW_COUNT:
            break
    return results[:REVIEW_COUNT]


def fetch_appstore_reviews() -> list[dict]:
    from app_store_scraper import AppStore
    import logging

    app = AppStore(country="us", app_name="pinterest", app_id=APPSTORE_APP_ID)
    app.review(how_many=REVIEW_COUNT)
    if app.reviews:
        return [normalize_appstore_review(r) for r in app.reviews]

    # app_store_scraper returned nothing (Apple API token scraping broken) —
    # fall back to the public iTunes RSS feed which needs no auth.
    logging.getLogger(__name__).warning(
        "app_store_scraper returned 0 reviews; falling back to iTunes RSS API"
    )
    return _fetch_appstore_itunes_rss()


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
    import pandas as pd
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
