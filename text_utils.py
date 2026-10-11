"""
Shared text utilities for AspectIQ.

IMPORTANT: this file is imported by BOTH train_models.py and app.py.
Because the saved models call these functions internally, training-time and
prediction-time preprocessing can never drift apart again.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------- negation
NEGATORS = {
    "not", "no", "never", "cannot", "without", "hardly", "barely", "nothing",
    "neither", "nor", "nobody", "none", "lacks", "lack", "lacking", "n't",
}
# words that carry meaning for opinions and must never be dropped as stopwords
KEEP = NEGATORS | {
    "too", "very", "enough", "more", "most", "less", "least", "much", "few",
    "little", "well", "only", "again", "over", "off", "down", "out", "against",
    "so", "quite", "really", "extremely", "just", "even", "nor", "up",
}
STOP = set(ENGLISH_STOP_WORDS) - KEEP

CLAUSE_BREAK = re.compile(
    r"[.;!?:\n]+|\s[-–—]+\s|"
    r"\b(?:but|however|although|though|whereas|while|yet|except|despite|unfortunately)\b|"
    r"\band\b(?=\s+(?:the|my|its|their|this|that|these|those|it|i|they|there|customer|battery|screen|keyboard)\b)",
    re.I,
)

def _expand(text):
    t = str(text).lower()
    t = t.replace("can't", "cannot").replace("won't", "will not")
    t = re.sub(r"n't\b", " not", t)
    return t

def normalize(text):
    """lowercase -> expand n't -> letters only -> drop stopwords (keep negators)
    -> mark words inside a negation window with NOT_ -> light plural stemming."""
    t = _expand(text)
    t = re.sub(r"[^a-z\s,]", " ", t)          # keep commas as negation-scope breakers
    out, neg_left = [], 0
    for tok in re.findall(r"[a-z]+|,", t):
        if tok == ",":
            neg_left = 0
            continue
        if tok in NEGATORS:
            neg_left = 3
            out.append("not")
            continue
        if tok in STOP or len(tok) < 2:
            continue
        if len(tok) > 3 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(("not_" + tok) if neg_left > 0 else tok)
        if neg_left > 0:
            neg_left -= 1
    return " ".join(out)

def prep_for_model(text):
    """Used as the TfidfVectorizer preprocessor inside the saved pipelines."""
    return normalize(text)

# --------------------------------------------------------- clause extraction
def _spans(sentence, pattern):
    spans, last = [], 0
    for m in pattern.finditer(sentence):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(sentence)))
    return [(a, b) for a, b in spans if sentence[a:b].strip()]

def _find_span(spans, pos):
    for a, b in spans:
        if a <= pos <= b:
            return a, b
    return spans[-1] if spans else (0, 0)

COMMA = re.compile(r",")

def aspect_clause(sentence, start, end):
    """Return the part of `sentence` that talks about the aspect at [start,end).

    1. split on . ; ! ? and contrast words (but/however/although...)
    2. inside that segment also split on commas and take the comma-part holding
       the aspect - but only if that part contains an opinion word;
       otherwise fall back to the whole segment.
    """
    sentence = str(sentence)
    start = max(0, min(int(start), len(sentence)))
    seg_a, seg_b = _find_span(_spans(sentence, CLAUSE_BREAK), start)
    seg = sentence[seg_a:seg_b]
    rel = start - seg_a
    parts = _spans(seg, COMMA)
    if len(parts) > 1:
        pa, pb = _find_span(parts, rel)
        part = seg[pa:pb]
        prof = opinion_profile(part.replace("aspecttarget", " "))
        # use the comma-part only if it carries its own opinion; otherwise the
        # opinion is in a neighbouring part ("Battery lasts all day, which is
        # impressive") so keep the whole segment.
        if prof["p"] + prof["n"] > 0:
            return part.strip(" ,")
    return seg.strip(" ,")

def locate(sentence, term):
    """Find the aspect term inside free text (used at prediction time)."""
    m = re.search(r"(?<!\w)" + re.escape(str(term)) + r"(?!\w)", str(sentence), re.I)
    if m:
        return m.start(), m.end()
    i = str(sentence).lower().find(str(term).lower())
    return (i, i + len(term)) if i >= 0 else (0, 0)

def mark_aspect(clause, term):
    """Replace the aspect words by one marker token so the model knows WHICH
    words are the target, then append the aspect words for topic information."""
    # If the "aspect" is itself an opinion word (slow, fast, expensive...) keep it,
    # otherwise masking it would erase the opinion.
    if _stem_lookup(str(term).lower(), POS_WORDS | NEG_WORDS):
        return str(clause)
    pat = re.compile(r"(?<!\w)" + re.escape(str(term)) + r"(?!\w)", re.I)
    return pat.sub(" aspecttarget ", str(clause), count=1)

def build_input(sentence, term, start=None, end=None):
    """The ONE function that creates model input, for training and prediction."""
    if start is None or end is None:
        start, end = locate(sentence, term)
    clause = aspect_clause(sentence, start, end)
    return mark_aspect(clause, term) + " || " + str(term)


# ======================================================================
# OPINION LEXICON (laptop-review domain) + lexicon features
# ======================================================================
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

POS_WORDS = set("""
good great excellent amazing awesome fantastic wonderful perfect love loved loves lovely beautiful gorgeous
fast quick quickly smooth snappy responsive bright clear crisp sharp vivid sturdy solid durable reliable
comfortable easy simple intuitive nice best better superb superior stunning impressive incredible outstanding
fabulous flawless happy pleased satisfied satisfying recommend recommended worth affordable cheap inexpensive
lightweight light portable slim sleek elegant premium responsive quiet cool helpful friendly polite professional
prompt seamless stable efficient powerful robust lasting enjoy enjoyed enjoyable pleasant decent fine ok okay
handy useful convenient improved improvement plus pro breeze charm bargain steal glad thrilled delighted
blazing lightning flawlessly nicely well perfectly fantastic gem winner phenomenal remarkable terrific
""".split())
NEG_WORDS = set("""
bad poor terrible horrible awful worst hate hated hates useless slow sluggish laggy lag lags crash crashes crashed
crashing freeze freezes froze freezing hang hangs broken break breaks broke fail fails failed failure failing dead
drain drains drained draining hot hotter overheat overheats overheating noisy loud flimsy cheap_feeling weak
dim dark blurry fuzzy grainy cramped uncomfortable difficult hard confusing complicated annoying annoyed annoy
frustrating frustrated disappointing disappointed disappoint disappointment problem problems issue issues
defective faulty flaw flawed glitch glitchy buggy bug bugs unreliable unresponsive unusable worthless junk garbage
trash rubbish pathetic ridiculous unacceptable overpriced expensive costly pricey rude unhelpful incompetent
scam ripoff regret regretted waste wasted horrible mediocre underwhelming lacking lacks lack missing
poorly badly sucks suck sucked disgusting nightmare hassle mess messy stuck heavy bulky clunky outdated obsolete
inferior unstable stupid ugly cumbersome irritating tedious painful frustrating unfortunately damage damaged
cracked scratched scratch loose wobbly squeaky rattle rattling stopped stops refuse refused wrong error errors
dull washed unbearable limited jumpy die dies died dying sticks sticky flicker flickers flickering
""".split())
STRONG_POS = set("love loved loves amazing awesome fantastic wonderful perfect excellent superb stunning incredible outstanding fabulous flawless gorgeous best phenomenal terrific thrilled delighted blazing lightning breeze".split())
ANGER_WORDS = set("terrible horrible awful worst hate hated hates useless garbage trash rubbish pathetic ridiculous unacceptable scam ripoff worthless disgusting rude incompetent nightmare junk sucks suck sucked stupid furious angry".split())
DISAPPOINT_WORDS = set("disappointing disappointed disappoint disappointment poor mediocre underwhelming overpriced expensive costly pricey lacking lacks lack regret regretted flimsy weak unfortunately waste wasted inferior outdated missing cheap_feeling".split())
INTENSIFY = {"very", "really", "extremely", "incredibly", "so", "too", "super", "absolutely", "totally", "highly", "quite", "especially"}
DAMPEN = {"slightly", "somewhat", "bit", "little", "fairly", "kinda", "sort"}

def _lex_tokens(text):
    t = _expand(text)
    t = re.sub(r"[^a-z\s,]", " ", t)
    return re.findall(r"[a-z]+|,", t)

def _stem_lookup(tok, vocab):
    if tok in vocab:
        return True
    for suf in ("ing", "ed", "es", "s", "ly"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3 and tok[: -len(suf)] in vocab:
            return True
    return False

POS_PHRASES = ["all day", "lasts long", "long lasting", "long battery", "works well", "worth it", "worth every",
               "highly recommend", "good value", "great value", "no complaints", "no problem", "no issues",
               "stays cool", "runs cool", "holds up", "up to date", "state of the art", "top notch", "second to none",
               "cannot be beat", "can't be beat", "loud and clear", "value for money", "worth the money", "worth the price"]
NEG_PHRASES = ["too high", "too expensive", "too small", "too big", "too heavy", "too dim", "too loud", "too slow",
               "too hot", "too long", "not worth", "at all times", "gets hot", "get hot", "runs hot", "waste of",
               "fell apart", "falls apart", "stopped working", "stops working", "does not work", "did not work",
               "never worked", "keeps crashing", "keeps freezing", "keeps disconnecting", "no support", "hard to",
               "difficult to", "drains quickly", "drains fast", "dies quickly", "dies fast", "die fast", "does not last", "doesn't last"]

def _phrase_hits(text, phrases):
    low = " " + _expand(text) + " "
    low = re.sub(r"[^a-z\s]", " ", low)
    low = re.sub(r"\s+", " ", low)
    return sum(1 for p in phrases if (" " + re.sub(r"[^a-z\s]", "", _expand(p)) + " ") in low)

def opinion_profile(text):
    """Return dict with negation-aware polarity evidence for a clause."""
    toks = _lex_tokens(text)
    pos = neg = npos = nneg = 0.0
    neg_left = 0
    mult = 1.0
    strong_pos = anger = disappoint = 0
    for i, tok in enumerate(toks):
        if tok == ",":
            neg_left = 0; mult = 1.0; continue
        if tok in NEGATORS:
            neg_left = 3; continue
        if tok in INTENSIFY:
            mult = 1.5; continue
        if tok in DAMPEN:
            mult = 0.6; continue
        is_p = _stem_lookup(tok, POS_WORDS)
        is_n = _stem_lookup(tok, NEG_WORDS)
        if is_p or is_n:
            negated = neg_left > 0
            if is_p:
                if negated: nneg += mult
                else: pos += mult
            if is_n:
                if negated: npos += mult * 0.6      # "not slow" is only mildly positive
                else: neg += mult
            if not negated:
                if _stem_lookup(tok, STRONG_POS): strong_pos += 1
                if _stem_lookup(tok, ANGER_WORDS): anger += 1
                if _stem_lookup(tok, DISAPPOINT_WORDS): disappoint += 1
            neg_left = max(0, neg_left - 1)
            mult = 1.0
        elif neg_left > 0:
            neg_left -= 1
    ph_pos = _phrase_hits(text, POS_PHRASES)
    ph_neg = _phrase_hits(text, NEG_PHRASES)
    # a phrase like "not worth" also contains a lexicon word; only add phrases once
    pos += ph_pos
    neg += ph_neg
    p_total = pos + npos
    n_total = neg + nneg
    return dict(pos=pos, neg=neg, neg_pos=npos, neg_neg=nneg, p=p_total, n=n_total,
                net=p_total - n_total, strong_pos=strong_pos, anger=anger, disappoint=disappoint)

class LexiconFeatures(BaseEstimator, TransformerMixin):
    """Dense opinion-lexicon features that sit next to TF-IDF in the pipeline."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rows = []
        for x in X:
            clause = str(x).split("||")[0].replace("aspecttarget", " ")
            o = opinion_profile(clause)
            net = o["net"]
            rows.append([
                np.tanh(net), float(o["p"] > 0), float(o["n"] > 0),
                float(o["p"] > 0 and o["n"] > 0), float(o["p"] == 0 and o["n"] == 0),
                np.tanh(o["neg_pos"]), np.tanh(o["neg_neg"]),
                float(o["strong_pos"] > 0), float(o["anger"] > 0), float(o["disappoint"] > 0),
            ])
        return np.asarray(rows) * 2.0


# ======================================================================
# ASPECT CATEGORIES + EMOTION LABELS (shared by training and app)
# ======================================================================
ASPECT_CATEGORIES = {
    "Battery": ["battery life", "battery", "charging", "charger", "charge", "power supply", "power"],
    "Display": ["display", "screen", "lcd", "resolution", "brightness", "monitor", "graphics", "viewing angle"],
    "Keyboard": ["keyboard", "keys", "key", "touchpad", "trackpad"],
    "Performance": ["performance", "speed", "processor", "cpu", "ram", "memory", "hard drive", "drive", "boot",
                    "slow", "fast", "lag", "laggy", "heating", "overheating", "fan"],
    "Software": ["software", "program", "programs", "application", "applications", "windows", "operating system",
                 "os", "microsoft office"],
    "Design": ["design", "look", "appearance", "style", "size", "weight", "thickness", "enclosure", "build", "build quality"],
    "Sound": ["sound", "speaker", "speakers", "audio", "volume", "microphone", "mic"],
    "Camera": ["camera", "webcam", "photo", "photos", "picture", "pictures"],
    "Connectivity": ["wifi", "wi-fi", "wireless", "bluetooth", "usb", "ethernet", "network", "internet", "port", "ports", "hdmi"],
    "Price": ["price", "cost", "value", "expensive", "cheap", "money", "worth", "deal"],
    "Service & Support": ["service", "support", "customer service", "sales team", "warranty", "repair", "exchange", "help"],
    "Gaming": ["gaming", "game", "games"],
}

def _has_word(text, word):
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None

def categorize_aspect(term):
    """Word-boundary version of the notebook's categorize_aspect
    (the notebook used substring matching, so 'os' matched 'photos')."""
    t = str(term).lower().strip()
    for category, words in ASPECT_CATEGORIES.items():
        for w in sorted(words, key=len, reverse=True):
            if _has_word(t, w):
                return category
    return "Other"

def emotion_from_evidence(polarity, clause):
    """Emotion label = f(sentiment, opinion words in THE ASPECT'S CLAUSE).
    Whole-word + negation aware, and always consistent with the polarity."""
    o = opinion_profile(str(clause).replace("aspecttarget", " ").split("||")[0])
    if polarity == "positive":
        return "Joy" if o["strong_pos"] > 0 else "Satisfaction"
    if polarity == "negative":
        if o["anger"] > 0:
            return "Anger"
        if o["disappoint"] > 0:
            return "Disappointment"
        return "Frustration"
    return "Neutral"           # neutral + conflict (mixed) -> Neutral

ALLOWED_EMOTIONS = {
    "positive": ["Joy", "Satisfaction"],
    "negative": ["Anger", "Frustration", "Disappointment"],
    "neutral": ["Neutral"],
    "conflict": ["Neutral"],
}
