"""
AspectIQ prediction engine (no Streamlit code in here, so it can be unit-tested).

review -> aspects -> clause around each aspect -> sentiment model
       -> emotion model restricted to emotions that fit the sentiment
"""
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion

from text_utils import (ASPECT_CATEGORIES, ALLOWED_EMOTIONS, build_input, aspect_clause,
                        categorize_aspect)

EMOTION_SCORE = {"anger": -3, "frustration": -2, "disappointment": -1,
                 "neutral": 0, "satisfaction": 1, "joy": 2}

# dataset "aspect terms" that are verbs/filler rather than product parts
JUNK_TERMS = {"quality", "look", "looks", "use", "works", "work", "run", "runs", "carry", "navigate", "feel", "built", "loads",
              "pages", "word", "case", "edges", "space", "functions", "operation", "shipped",
              "features", "feature", "office", "set up", "specs", "specifications", "buttons"}

# category keywords that are really OPINION words (slow, fast, expensive...)
OPINION_TRIGGERS = {"slow", "fast", "lag", "laggy", "heating", "overheating", "expensive", "cheap",
                    "worth", "deal", "help"}
# never a product aspect on its own (usually a verb: "photos look grainy")
NEVER_ASPECT = {"look", "looks", "quality"}


# ------------------------------------------------------------ model checks
def is_current_model(model):
    """True if the pipeline was produced by train_models.py (has lexicon features)."""
    try:
        tf = model.named_steps["tfidf"]
        return isinstance(tf, FeatureUnion) and any(n == "lex" for n, _ in tf.transformer_list)
    except Exception:
        return False


# ------------------------------------------------------------ aspect terms
def dataset_terms_from(data):
    """Frequent, non-junk aspect terms from the training data."""
    if data is None or "aspect" not in data.columns:
        return []
    counts = data["aspect"].dropna().astype(str).str.strip().str.lower().value_counts()
    terms = [t for t, c in counts.items()
             if c >= 3 and len(t) >= 3 and t not in JUNK_TERMS and re.search(r"[a-z]", t)]
    return terms


def detect_aspects(review, dataset_terms=()):
    text = str(review)
    low = text.lower()
    keywords = {}
    for category, words in ASPECT_CATEGORIES.items():
        for w in words:
            if w.lower() not in NEVER_ASPECT:
                keywords[w.lower()] = category
    for t in dataset_terms:
        keywords.setdefault(t, categorize_aspect(t))

    found = []                               # (start, end, term)
    for term in sorted(keywords, key=len, reverse=True):
        for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", low):
            s, e = m.span()
            if any(not (e <= fs or s >= fe) for fs, fe, _ in found):   # overlaps a longer match
                continue
            found.append((s, e, term))
    found.sort()

    # keep one entry per term (first mention)
    seen, items = set(), []
    for s, e, term in found:
        if term in seen:
            continue
        seen.add(term)
        items.append({"term": term, "start": s, "end": e,
                      "category": keywords.get(term) or categorize_aspect(term)})

    # drop opinion-word "aspects" when a real term of the same category exists
    real_cats = {i["category"] for i in items if i["term"] not in OPINION_TRIGGERS}
    items = [i for i in items if i["term"] not in OPINION_TRIGGERS or i["category"] not in real_cats]
    # one row per (category, clause): "photos" + "webcam" in one clause is one Camera aspect
    out, used = [], set()
    for i in items:
        clause = aspect_clause(text, i["start"], i["end"])
        key = (i["category"], clause)
        if i["category"] != "Other" and key in used:
            continue
        used.add(key)
        out.append(i)
    return out[:12]


# ------------------------------------------------------------- prediction
def _proba(model, x):
    p = model.predict_proba([x])[0]
    return dict(zip(model.classes_, p))


def predict_aspect(sentiment_model, emotion_model, review, item):
    x = build_input(review, item["term"], item["start"], item["end"])
    clause = aspect_clause(review, item["start"], item["end"])

    sp = _proba(sentiment_model, x)
    # 'conflict' is ~2% of the data and the model is very unreliable on it,
    # so only show it if the model is actually confident.
    ranked = sorted(sp.items(), key=lambda kv: kv[1], reverse=True)
    sentiment, conf = ranked[0]
    if sentiment == "conflict" and conf < 0.5:
        sentiment, conf = next((k, v) for k, v in ranked if k != "conflict")

    ep = _proba(emotion_model, x)
    allowed = [e for e in ALLOWED_EMOTIONS.get(sentiment, ["Neutral"]) if e in ep] or \
              ALLOWED_EMOTIONS.get(sentiment, ["Neutral"])
    emotion = max(allowed, key=lambda e: ep.get(e, 0.0))
    return clause, x, sentiment, emotion, float(conf), sp, ep


def analyze_review(review, sentiment_model, emotion_model, dataset_terms=()):
    rows = []
    for item in detect_aspects(review, dataset_terms):
        try:
            clause, x, sentiment, emotion, conf, sp, ep = predict_aspect(
                sentiment_model, emotion_model, review, item)
            rows.append({
                "Aspect": item["term"].title(),
                "Category": item["category"],
                "Context": clause,
                "Sentiment": sentiment.title(),
                "Emotion": emotion,
                "Emotion Score": EMOTION_SCORE.get(emotion.lower(), 0),
                "Confidence": round(conf, 2),
                "Model Input": x,
                "Sentiment Probabilities": ", ".join(f"{k}: {v:.2f}" for k, v in
                                                    sorted(sp.items(), key=lambda kv: -kv[1])),
            })
        except Exception as exc:                       # never hide errors
            rows.append({
                "Aspect": item["term"].title(), "Category": item["category"], "Context": "",
                "Sentiment": "Prediction Error", "Emotion": "Prediction Error", "Emotion Score": 0,
                "Confidence": 0.0, "Model Input": "", "Sentiment Probabilities": str(exc),
            })
    return pd.DataFrame(rows)
