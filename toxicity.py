"""Toxicity classifier with two swappable backends.

1. lexicon     - zero-dependency, offline: weighted category lexicons + text signals.
2. transformer - detoxify "original" model (pip install detoxify torch).

Both expose: predict(text) -> {"toxicity": float in [0,1], "is_toxic": bool,
                               "categories": {cat: float}}
"""
import re, html
import config

# ----------------------------- lexicon backend ------------------------------
# Whole-word / whole-phrase matching only (the old substring match flagged
# "crypto" via "cry", "diet" via "die", etc.).
LEXICONS = {
    "profanity": ["fuck", "shit", "bitch", "asshole", "dick", "cunt", "piss", "bastard"],
    "insult":    ["idiot", "stupid", "dumb", "moron", "loser", "retard", "retarded",
                  "trash", "pathetic", "clown", "useless", "worthless", "ugly",
                  "hate you", "shut up"],
    "threat":    ["kill yourself", "kys", "go die", "hope you die", "kill you",
                  "hurt you", "beat you", "destroy you", "come find"],
    "hate":      ["nigger", "faggot", "nazi", "kkk"],
    "spam":      ["subscribe to my channel", "check my channel", "free money",
                  "click the link", "crypto giveaway", "www.", "http://", "https://",
                  "buy now", "promo code"],
}

# Severity of one hit in each category. Hits are combined as 1 - prod(1 - w),
# so one threat/slur is toxic on its own, while a single mild insult is not.
WEIGHTS = {"profanity": 0.35, "insult": 0.40, "threat": 0.80, "hate": 0.90, "spam": 0.10}


def _term_regex(term: str) -> str:
    pre  = r"(?<!\w)" if term[0].isalnum() else ""
    post = r"(?!\w)"  if term[-1].isalnum() else ""
    return pre + re.escape(term) + post

_PATTERNS = {cat: [(t, re.compile(_term_regex(t))) for t in terms]
             for cat, terms in LEXICONS.items()}


def _lexicon_score(text: str) -> dict:
    raw = html.unescape(text)
    t = raw.lower()
    cats, keep = {}, 1.0
    for cat, pats in _PATTERNS.items():
        h = sum(1 for _, rx in pats if rx.search(t))
        cats[cat] = round(min(h / 3.0, 1.0), 4)
        keep *= (1 - WEIGHTS[cat]) ** h
    score = 1 - keep
    # text-level signals (computed on the ORIGINAL text: the old code lower-cased
    # first, so the CAPS rule could never fire). Only boost if something was hit.
    if score > 0:
        alpha = [c for c in raw if c.isalpha()]
        if len(alpha) > 10 and sum(c.isupper() for c in alpha) / len(alpha) > 0.6:
            score += 0.15
        if raw.count("!") > 4:
            score += 0.10
    return {"toxicity": round(min(score, 1.0), 4), "categories": cats}


class LexiconBackend:
    def predict(self, text: str) -> dict:
        return _lexicon_score(text)

# --------------------------- transformer backend ----------------------------
# detoxify label -> our category names (so transform.py's cat_* columns are filled)
_DETOX_MAP = {"obscene": "profanity", "insult": "insult", "threat": "threat",
              "identity_attack": "hate"}

class TransformerBackend:
    def __init__(self):
        from detoxify import Detoxify           # lazy import: only if selected
        self._model = Detoxify("original")

    def predict(self, text: str) -> dict:
        scores = self._model.predict(text)
        cats = {_DETOX_MAP.get(k, k.replace("_", " ")): round(float(v), 4)
                for k, v in scores.items() if k != "toxicity"}
        return {"toxicity": round(float(scores["toxicity"]), 4), "categories": cats}

# ------------------------------- dispatcher ---------------------------------
_backend = None

def get_backend():
    global _backend
    if _backend is None:
        _backend = (TransformerBackend() if config.TOXICITY_BACKEND == "transformer"
                    else LexiconBackend())
    return _backend

def predict(text: str) -> dict:
    r = get_backend().predict(text or "")
    r["is_toxic"] = bool(r["toxicity"] >= config.TOXIC_THRESHOLD)
    return r
