"""Central configuration for the YouTube Comment Toxicity Analyzer."""
import os

# --- YouTube Data API ---
# Never hard-code keys. Set it in your shell:  export YOUTUBE_API_KEY="..."
# (Windows PowerShell:  $env:YOUTUBE_API_KEY="...")
API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Either a list of video IDs, or channel handles (@name) / channel IDs to pull videos from.
VIDEO_IDS = ["dQw4w9WgXcQ"]                      # example
CHANNELS  = ["@mkbhd", "@MrBeast"]               # example; leave empty to skip

MAX_COMMENTS_PER_VIDEO = 500                     # safety cap per video
MAX_VIDEOS_PER_CHANNEL = 25                      # most recent N videos per channel

# --- Storage ---
DATA_DIR   = "data"
RAW_PATH   = os.path.join(DATA_DIR, "raw_comments.jsonl")      # extract -> transform
CLEAN_CSV  = os.path.join(DATA_DIR, "comments_clean.csv")      # load output
DB_PATH    = os.path.join(DATA_DIR, "youtube_toxicity.db")     # load output
REPORT_DIR = os.path.join(DATA_DIR, "reports")

# --- Toxicity model ---
# "lexicon" (offline, zero deps) or "transformer" (detoxify, needs torch)
TOXICITY_BACKEND = os.getenv("TOXICITY_BACKEND", "lexicon")
TOXIC_THRESHOLD  = 0.5
