# Pinterest App Store Review Analyzer — Design Spec

**Date:** 2026-04-04  
**Status:** Approved

---

## Overview

A Python pipeline that scrapes Pinterest reviews from both the Apple App Store and Google Play Store, runs sentiment analysis and topic clustering on them, and displays results in an interactive Streamlit dashboard. Intended for periodic manual refresh to track trends over time.

---

## Project Structure

```
pinterest-review-analyzer/
├── scrape.py          # Pulls reviews from App Store + Google Play → data/reviews.csv
├── analyze.py         # Enriches reviews with sentiment + topic → data/analyzed.csv
├── app.py             # Streamlit dashboard reads data/analyzed.csv
├── requirements.txt
└── data/
    ├── reviews.csv
    └── analyzed.csv
```

---

## Data Flow

1. Run `scrape.py` → fetches latest Pinterest reviews from both stores, saves to `data/reviews.csv`
2. Run `analyze.py` → adds sentiment score/label and topic cluster label to each review, saves to `data/analyzed.csv`
3. Run `streamlit run app.py` → loads `analyzed.csv` and renders the interactive dashboard

Re-running `scrape.py` overwrites the CSV (simple refresh, no deduplication).

---

## Scraping (`scrape.py`)

- **Apple App Store:** `app-store-scraper`, Pinterest app ID `429047995`
- **Google Play:** `google-play-scraper`, package ID `com.pinterest`
- Fetches most recent N reviews (constant `REVIEW_COUNT = 500` at top of file, easy to change)
- Normalizes both sources into a common schema:
  ```
  id | store | date | rating | title | review
  ```
- Saves combined output to `data/reviews.csv`

---

## Analysis (`analyze.py`)

Reads `data/reviews.csv`, adds two new columns, saves to `data/analyzed.csv`.

### Sentiment (VADER)
- Uses `vaderSentiment` to compute a `compound` score per review
- Labels:
  - `positive`: compound ≥ 0.05
  - `neutral`: −0.05 < compound < 0.05
  - `negative`: compound ≤ −0.05
- Output columns: `sentiment_score`, `sentiment_label`

### Topic Clustering (TF-IDF + KMeans)
- TF-IDF vectorization of review text (stopwords removed)
- KMeans clustering into 8 clusters (tunable)
- Each cluster is auto-labeled by joining its top 3 TF-IDF keywords (e.g., `"crash · load · freeze"`), shown as-is in the dashboard — no manual labeling step required
- Output column: `topic`

---

## Dashboard (`app.py`)

Built with Streamlit + Plotly.

### Sidebar Filters
- Store: All / App Store / Google Play
- Star rating: multiselect 1–5
- Sentiment: All / Positive / Neutral / Negative
- Topic: All + the 8 cluster labels
- Date range: slider

### Main Panels

1. **Summary metrics row** — total reviews, avg rating, % positive / neutral / negative (live with filters)
2. **Sentiment over time** — Plotly line chart of sentiment breakdown by month
3. **Rating distribution** — Plotly bar chart split by store
4. **Topic breakdown** — horizontal bar chart, review count per topic, colored by dominant sentiment
5. **Review table** — paginated, sortable table with all columns including sentiment + topic labels

---

## Design System

### Aesthetic Direction
Dark-mode data dashboard. Pinterest brand red (`#E60023`) as primary accent. Editorial, data-forward, not decorative.

### Theme (`.streamlit/config.toml`)
- Background: `#0F0F0F`
- Card/surface: `#1A1A1A`
- Primary accent: `#E60023`
- Text: `#F5F5F5`
- Muted text: `#888888`

### Typography (injected via `st.markdown` custom CSS)
- Headings: `Syne` (Google Fonts)
- Body: `DM Sans` (Google Fonts)
- Base size: 16px, line-height 1.5

### Chart Principles (ui-ux-pro-max)
- All Plotly charts use a colorblind-safe sequential/diverging palette
- Every chart has legends + hover tooltips with exact values
- No chart conveys meaning by color alone — text labels added where needed
- Consistent card elevation: charts sit inside `#1A1A1A` containers with a subtle `1px #2A2A2A` border

---

## Dependencies

```
google-play-scraper
app-store-scraper
vaderSentiment
scikit-learn
streamlit
pandas
plotly
```

---

## Out of Scope

- Scheduled/automated scraping
- Database storage or deduplication across runs
- Claude API or transformer-based NLP
- Export to CSV/PDF
- Deployment (runs locally only)
