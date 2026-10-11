"""
AspectIQ - training script (replaces the model-training cells of the notebook)

Run once:   python train_models.py
Needs:      Laptop_Train_v2.csv  (next to this file)
Creates:    models/sentiment_model.pkl, models/emotion_model.pkl,
            emotion_dataset.csv, model_comparison.csv, models/metrics.json

What changed compared with the notebook, and why
  1. Input = the CLAUSE around the aspect (found with the dataset's from/to
     offsets), not the whole sentence -> "display amazing but battery drains"
     no longer gives both aspects the same answer.
  2. Negation is kept and marked (not good -> not not_good) and an opinion
     lexicon is added as features -> "not good" is no longer Positive.
  3. Preprocessing lives INSIDE the saved pipeline (text_utils.py), so the app
     can never preprocess differently from training.
  4. Emotion labels are rebuilt with whole-word, negation-aware rules that are
     always consistent with polarity (old rule: "didn't like" -> Satisfaction).
  5. Evaluation groups rows by sentence id. The notebook's random split put
     the same sentence in train AND test (52% of test rows), inflating scores.
"""
import json, os, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from text_utils import (LexiconFeatures, prep_for_model, build_input, categorize_aspect,
                        emotion_from_evidence)

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
SRC = next((p for p in [BASE / "Laptop_Train_v2.csv", Path("Laptop_Train_v2.csv")] if p.exists()), None)
if SRC is None:
    raise SystemExit("Laptop_Train_v2.csv not found next to train_models.py")
(BASE / "models").mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(SRC).drop_duplicates().dropna(subset=["Sentence", "Aspect Term", "polarity"])
df["polarity"] = df["polarity"].astype(str).str.lower().str.strip()
df = df[df["polarity"].isin(["positive", "negative", "neutral", "conflict"])].reset_index(drop=True)
df["Model Input"] = [build_input(s, t, a, b) for s, t, a, b in
                     zip(df["Sentence"], df["Aspect Term"], df["from"], df["to"])]
df["Aspect Category"] = df["Aspect Term"].apply(categorize_aspect)
df["Emotion"] = [emotion_from_evidence(p, x) for p, x in zip(df["polarity"], df["Model Input"])]
print("rows:", len(df)); print(df["polarity"].value_counts().to_string()); print(df["Emotion"].value_counts().to_string())

# ---------------------------------------------------------------- models
def features(with_lex=True):
    parts = [("w", TfidfVectorizer(preprocessor=prep_for_model, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
             ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                   preprocessor=prep_for_model))]
    if with_lex:
        parts.append(("lex", LexiconFeatures()))
    return FeatureUnion(parts)

def lr():  return LogisticRegression(max_iter=3000, class_weight="balanced", C=3, random_state=42)
def svm(): return LinearSVC(class_weight="balanced", C=0.3, random_state=42)
def nb():  return ComplementNB(alpha=0.5)

def make(name):
    if name == "Logistic Regression": return Pipeline([("tfidf", features()), ("model", lr())])
    if name == "Linear SVM":          return Pipeline([("tfidf", features()), ("model", svm())])
    # Naive Bayes cannot take the (signed) lexicon features -> TF-IDF only
    return Pipeline([("tfidf", features(False)), ("model", nb())])

gkf = GroupKFold(n_splits=5)
rows = []
for task, target in [("Sentiment", "polarity"), ("Emotion", "Emotion")]:
    y = df[target]
    for name in ["Logistic Regression", "Linear SVM", "Multinomial Naive Bayes"]:
        pred = np.empty(len(df), dtype=object)
        for tr, te in gkf.split(df, y, df["id"]):
            pred[te] = make(name).fit(df["Model Input"].iloc[tr], y.iloc[tr]).predict(df["Model Input"].iloc[te])
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
        rows.append({"Task": task, "Model": name, "Accuracy": round(accuracy_score(y, pred) * 100, 2),
                     "Precision": round(p * 100, 2), "Recall": round(r * 100, 2), "F1 Score": round(f * 100, 2)})
        print(f"{task:9s} {name:24s} acc={rows[-1]['Accuracy']:.2f}  F1={rows[-1]['F1 Score']:.2f}")
comparison = pd.DataFrame(rows)
comparison.to_csv(BASE / "model_comparison.csv", index=False, encoding="utf-8-sig")

# --------------------------------------------------------- final models
final = {}
for task, target in [("sentiment", "polarity"), ("emotion", "Emotion")]:
    sub = comparison[comparison["Task"] == task.title()].sort_values("F1 Score", ascending=False)
    best = sub.iloc[0]["Model"]
    if task == "emotion" or best == "Linear SVM" or best == "Multinomial Naive Bayes":
        best = "Logistic Regression"          # app needs class probabilities
    model = make(best).fit(df["Model Input"], df[target])
    joblib.dump(model, BASE / "models" / f"{task}_model.pkl")
    final[task] = best
    print(f"saved models/{task}_model.pkl  ({best})")

# per-class report from the grouped cross-validation (sentiment)
pred = np.empty(len(df), dtype=object)
for tr, te in gkf.split(df, df["polarity"], df["id"]):
    pred[te] = make("Logistic Regression").fit(df["Model Input"].iloc[tr], df["polarity"].iloc[tr]).predict(df["Model Input"].iloc[te])
print("\nSentiment (grouped CV):\n", classification_report(df["polarity"], pred, zero_division=0))

# ------------------------------------------------ files the dashboard reads
df[["id", "Sentence", "Aspect Term", "Aspect Category", "polarity", "Emotion", "from", "to"]].to_csv(
    BASE / "emotion_dataset.csv", index=False, encoding="utf-8-sig")
json.dump({"final_models": final, "evaluation": "5-fold GroupKFold by sentence id"},
          open(BASE / "models" / "metrics.json", "w"), indent=2)
print("\nDone. Now run:  streamlit run app.py")
