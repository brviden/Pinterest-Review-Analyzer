# Pinterest Review Analyzer

**Live Demo: [pinterest-review-analyzer.streamlit.app](https://pinterest-review-analyzer.streamlit.app/)**

A data pipeline and interactive dashboard that scrapes, analyzes, and visualizes Pinterest app store reviews. Built to show how product managers can use data to find user pain points and make better roadmap decisions.

---

## Why I Built This

App store reviews are one of the most underused sources of product feedback. They contain real, unfiltered user opinions, but at scale, reading them manually is not practical.

I built this tool to show how a PM can turn thousands of reviews into clear insights: which topics drive negative sentiment, how user satisfaction changes over time, and where the experience is breaking down across platforms.

This is the kind of analysis I would run before a quarterly planning cycle or after a major release.

---

## What It Does

- Scrapes up to 1,000 Pinterest reviews from the App Store and Google Play
- Runs VADER sentiment analysis to classify each review as positive, neutral, or negative
- Uses TF-IDF and KMeans clustering to group reviews into 8 topic buckets
- Shows everything in a dark-themed dashboard with filters for store, rating, sentiment, topic, and date range

---

## Key Product Insights

**Sentiment breakdown across 1,000 reviews:**
- Positive: 67%
- Negative: 20%
- Neutral: 13%
- Average rating: 3.54 out of 5

**Top complaints from negative reviews:**
1. Too many ads, users say real content is buried under promoted pins
2. Recent UI changes made the app harder to use
3. Bugs and crashes, especially on new devices or after updates
4. Account bans with no explanation or way to appeal
5. Performance issues like freezing and slow scrolling

**Most common topics across all reviews:**
- General app experience (love, like, Pinterest) dominated most reviews
- Ads were the second biggest topic, mostly negative
- App quality and performance came up frequently in both positive and negative reviews

---

## Product Recommendations

1. **Fix the ads experience.** The volume of ads is the single biggest driver of negative reviews. A better ad-to-content ratio or a cleaner ad format would directly improve satisfaction.

2. **Be more careful with UI changes.** Several negative reviews mentioned that updates made the app worse. New features should be tested with a subset of users before rolling out broadly.

3. **Improve account recovery.** Users who get banned feel frustrated because there is no clear path to appeal. A simple support flow here could recover users who are otherwise lost.

4. **Address performance on new devices.** Multiple reviews mention the app breaking after upgrading phones. This points to a compatibility issue worth investigating before the next major release.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Scraping | google-play-scraper, iTunes RSS API |
| Analysis | vaderSentiment, scikit-learn (TF-IDF + KMeans) |
| Dashboard | Streamlit, Plotly |
| Data | pandas, CSV |
| Testing | pytest (9 unit tests) |
