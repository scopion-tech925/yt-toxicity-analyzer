# YouTube Comment Toxicity Analyzer

An end-to-end ETL + EDA pipeline that collects YouTube comments through the YouTube Data API v3, scores each comment for toxicity, stores the results in SQLite/CSV, and explores how toxicity varies by **channel**, **topic**, and **video popularity**. A Streamlit dashboard presents the results as a live web app.

**Live app:** https://yt-toxicity-analyzer.streamlit.app/

---

## Features

- **Extract:** pulls top-level comments and video metadata (views, likes, title, channel) for given video IDs or channel handles
- **Transform:** validates records, removes duplicates and empty comments, cleans text, scores toxicity, and infers a topic from each video title
- **Load:** writes a normalized SQLite database (`videos` 1-N `comments`) and a flat CSV
- **EDA:** toxicity rate by channel and topic, plus a Spearman correlation between video views and toxicity
- **Dashboard:** interactive Streamlit app with filters, charts, the most toxic comments, and a box to score your own text

## Pipeline

```
YouTube Data API v3
        |
   extract.py  -->  data/raw_comments.jsonl
        |
  transform.py  (validate -> clean -> toxicity.py -> topic)
        |
     load.py    -->  data/comments_clean.csv + data/youtube_toxicity.db
        |
      eda.py    -->  data/reports/*.csv + toxicity_eda.png
        |
      app.py    -->  Streamlit dashboard
```

## Project structure

```
.
├── app.py              # Streamlit web app
├── main.py             # pipeline runner
├── config.py           # settings (videos, channels, limits, paths, backend)
├── extract.py          # YouTube API extraction
├── transform.py        # validation, cleaning, topic inference
├── toxicity.py         # toxicity classifier (lexicon / transformer backends)
├── load.py             # CSV + SQLite loading
├── eda.py              # analysis and charts
├── requirements.txt
└── data/
    ├── youtube_toxicity.db
    ├── comments_clean.csv
    └── reports/        # CSV summaries + toxicity_eda.png
```

## Toxicity backends

Set with the `TOXICITY_BACKEND` environment variable.

| Backend | Description |
|---|---|
| `lexicon` (default) | Offline and dependency-free. Whole-word matching against weighted category lexicons (profanity, insult, threat, hate, spam), plus ALL-CAPS and exclamation signals. Hits are combined so one threat or slur is toxic on its own, while a single mild insult is not. |
| `transformer` | Uses the [Detoxify](https://github.com/unitaryai/detoxify) `original` model. More accurate but slower. Needs `pip install torch detoxify`. |

A comment is labeled toxic when its score is at least `TOXIC_THRESHOLD` (default 0.5).

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/scopion-tech925/yt-toxicity-analyzer.git
cd yt-toxicity-analyzer
python -m venv venv
# Windows PowerShell:  venv\Scripts\Activate.ps1
# Mac/Linux:           source venv/bin/activate
pip install -U -r requirements.txt
```

> On Windows PowerShell you may first need:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 2. Get a YouTube API key

1. Create a project at [Google Cloud Console](https://console.cloud.google.com/).
2. Enable **YouTube Data API v3**.
3. Create an API key under **Credentials**.
4. Set it as an environment variable. **Never commit it to the repo.**

```powershell
# Windows PowerShell
$env:YOUTUBE_API_KEY="your_key_here"
```
```bash
# Mac/Linux
export YOUTUBE_API_KEY="your_key_here"
```

### 3. Configure what to analyze

Edit `config.py`:

```python
VIDEO_IDS = ["dQw4w9WgXcQ"]        # individual videos
CHANNELS  = ["@mkbhd"]             # channel handles or IDs (can be empty)
MAX_COMMENTS_PER_VIDEO = 500
MAX_VIDEOS_PER_CHANNEL = 25
```

The free API quota is about 10,000 units per day, so start small.

### 4. Run the pipeline

```bash
python main.py            # extract -> transform -> load -> eda
python main.py extract    # or run a single step
python main.py load
python main.py eda
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The dashboard reads `data/youtube_toxicity.db`, so it works without an API key.

## Deployment

The app is hosted on [Streamlit Community Cloud](https://streamlit.io/cloud), deployed directly from this repository (`main` branch, entry file `app.py`). It only reads the committed database and never calls the YouTube API, so no secrets are required.

## Analysis notes and limitations

- The lexicon backend is a **rough baseline**. It can miss sarcasm, misspellings, and non-English comments, and may flag quoted or discussed language.
- Topics are inferred from **keywords in video titles**, so they are approximate.
- The views-vs-toxicity correlation uses few videos, and channel or topic can confound it. It shows association, not causation.
- The comment-level correlation treats comments as independent even though they share a video's view count, so its p-value is optimistic. The video-level result is the more reliable one.
- Only top-level comments are collected, not replies.

## Tech stack

Python, pandas, NumPy, SciPy, Matplotlib, SQLite, Google API Client, Streamlit (optional: PyTorch + Detoxify)

## Security

API keys are read from the `YOUTUBE_API_KEY` environment variable and are not stored in the repository. Do not commit keys or `.env` files.

## Author

**USHNA FATIMA**, UET Lahore,2025-AI-252 /BS AI  
GitHub: [@scopion-tech925](https://github.com/scopion-tech925)

