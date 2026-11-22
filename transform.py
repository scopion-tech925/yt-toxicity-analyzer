"""TRANSFORM: validate raw fields, clean text, run the toxicity classifier.

Input : data/raw_comments.jsonl   (from extract.py, or your own source)
Output: in-memory pandas DataFrame handed to load.py
"""
import json, html, re
from collections import Counter
import pandas as pd
import config, toxicity

REQUIRED = ["comment_id", "video_id", "text", "author", "like_count", "published_at"]
URL_RE  = re.compile(r"https?://\S+|www\.\S+")
WS_RE   = re.compile(r"\s+")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing required fields; coerce dtypes; drop exact dupes."""
    before = len(df)
    df = df.dropna(subset=REQUIRED).copy()
    df = df[df["text"].astype(str).str.strip().str.len() > 0]   # empty comments
    df = df.drop_duplicates(subset=["comment_id"])               # API double-pull
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    print(f"VALIDATE: {before} -> {len(df)} rows "
          f"({before - len(df)} invalid/duplicated removed)")
    return df.reset_index(drop=True)


def clean_text(s: str) -> str:
    s = html.unescape(str(s))
    s = URL_RE.sub("[URL]", s)
    return WS_RE.sub(" ", s).strip()


# ----------------------------- topic inference ------------------------------
# Broad topic buckets: the first keyword found in the title decides the topic.
TOPIC_KEYWORDS = {
    "tech":            ["iphone", "apple", "samsung", "android", "macbook", "app",
                         " ai ", "robot", "gadget", "tech", "camera", "phone",
                         "laptop", "unbox", "headphone"],
    "gaming":          ["minecraft", "fortnite", "gaming", "game", "gta",
                         "speedrun", "among us", "squid game"],
    "money":           ["$", "money", "cash", "bitcoin", "crypto", "lottery",
                         "prize", "jackpot", "giveaway", "dollar"],
    "food":            ["cooking", "recipe", "food", "pizza", "eating",
                         "restaurant", "chef", "burger", "chocolate"],
    "sports":          ["football", "soccer", "nba", "basketball", "match",
                         "cricket", "ufc", "boxing", "olympics"],
    "music":           ["song", "music", "album", "concert", "remix", "cover"],
    "news-politics":   ["news", "breaking", "election", "trump", "biden",
                         "president", "politics", "war", "debate"],
    "prank-challenge": ["prank", "challenge", "last to", "survive", "survival"],
}

STOPWORDS = {"the","a","an","and","or","of","to","in","on","for","with","my","our",
             "your","his","her","their","this","that","is","are","was","were","i",
             "you","we","they","he","she","it","at","by","from","as","be","has",
             "have","had","not","but","so","if","when","than","then","what","how",
             "official","video","episode","part","new","best","top","ft"}

def _kw_regex(k: str) -> str:
    """Whole-word match (so 'app' doesn't hit 'happy', 'war' doesn't hit 'software')."""
    k = k.strip()
    pre  = r"(?<!\w)" if k[0].isalnum() else ""
    post = r"(?!\w)"  if k[-1].isalnum() else ""
    return pre + re.escape(k) + post

_TOPIC_RES = {topic: re.compile("|".join(_kw_regex(k) for k in kws))
              for topic, kws in TOPIC_KEYWORDS.items()}


def infer_topic(title) -> str:
    if not isinstance(title, str) or not title.strip():
        return "general"
    t = title.lower()
    # 1) hashtag takes priority, mapped through a bucket when possible
    tags = re.findall(r"#(\w+)", t)
    if tags:
        for topic, rx in _TOPIC_RES.items():
            if rx.search(tags[0]):
                return topic
        return tags[0]                            # unknown hashtag -> keep as-is
    # 2) keyword buckets over the whole title
    for topic, rx in _TOPIC_RES.items():
        if rx.search(t):
            return topic
    # 3) fallback: most frequent meaningful word (deterministic tie-break: first seen)
    words = [w for w in re.findall(r"[a-z']{4,}", t) if w not in STOPWORDS]
    return Counter(words).most_common(1)[0][0] if words else "general"


def run(raw_path: str = config.RAW_PATH) -> pd.DataFrame:
    records = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = validate(pd.DataFrame(records))

    df["text_clean"] = df["text"].map(clean_text)
    df["text_len"]   = df["text_clean"].str.len()

    print(f"CLASSIFY: scoring {len(df)} comments "
          f"(backend={config.TOXICITY_BACKEND}) ...")
    preds = df["text_clean"].map(toxicity.predict)
    df["toxicity_score"] = preds.map(lambda p: p["toxicity"])
    df["is_toxic"]       = preds.map(lambda p: p["is_toxic"])
    for cat in ["profanity", "insult", "threat", "hate", "spam"]:
        df[f"cat_{cat}"] = preds.map(
            lambda p, c=cat: p["categories"].get(c, p["categories"].get(c.replace("_", " "), 0.0)))

    df["topic"] = (df["video_title"] if "video_title" in df.columns
                   else pd.Series("", index=df.index)).map(infer_topic)   # keyword-based topics
    print(f"TRANSFORM done: toxicity rate = {df['is_toxic'].mean():.2%}")
    return df


if __name__ == "__main__":
    run()
