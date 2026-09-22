"""
train_model.py — Fake News Detection ML Training Pipeline
==========================================================
Trains multiple classifiers (Logistic Regression, Naive Bayes, SVM,
Random Forest) on a labeled fake/real news dataset using TF-IDF features.
Evaluates each model and saves the best one for use by the Flask API.

Usage:
    python train_model.py

Outputs (saved to ./models/):
    vectorizer.pkl    — fitted TF-IDF vectorizer
    best_model.pkl    — best-performing classifier
    metrics.json      — per-model evaluation metrics + dataset info
"""

import os
import sys
import json
import time
import warnings
import requests
import io

# Force UTF-8 output so Unicode chars (─, ✓, 🏆) work on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

from preprocess import preprocess_batch

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ─── Dataset URLs (public mirrors) ────────────────────────────────────────────
DATASET_URLS = [
    # lutzhamel fake-news mirror (~6,300 rows, columns: id,title,text,label)
    "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
    # Alternative GitHub mirror
    "https://raw.githubusercontent.com/dhruvildave/fake-news-detection/master/data/fake_or_real_news.csv",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def banner(msg: str):
    line = "─" * 60
    print(f"\n{line}\n  {msg}\n{line}")


def load_dataset_from_url(url: str) -> pd.DataFrame | None:
    """Try to download and parse a fake/real news CSV from a URL."""
    try:
        print(f"  Trying: {url}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        print(f"  ✓ Downloaded {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None


def load_dataset_from_disk() -> pd.DataFrame | None:
    """Try to load a local dataset file from the dataset/ folder."""
    candidates = [
        ('fake_or_real_news.csv', None),        # single-file format
        ('Fake.csv', 'True.csv'),                # two-file Kaggle format
    ]
    for files in candidates:
        if files[1] is None:
            path = os.path.join(DATASET_DIR, files[0])
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"  ✓ Loaded from disk: {path} ({len(df):,} rows)")
                return df
        else:
            fake_path = os.path.join(DATASET_DIR, files[0])
            real_path = os.path.join(DATASET_DIR, files[1])
            if os.path.exists(fake_path) and os.path.exists(real_path):
                fake_df = pd.read_csv(fake_path)
                real_df = pd.read_csv(real_path)
                fake_df['label'] = 'FAKE'
                real_df['label'] = 'REAL'
                df = pd.concat([fake_df, real_df], ignore_index=True)
                print(f"  ✓ Loaded Kaggle-style dataset ({len(df):,} rows)")
                return df
    return None


def generate_synthetic_dataset() -> pd.DataFrame:
    """
    Generate a high-quality synthetic fake/real news dataset as a fallback.
    Creates lexically diverse samples with realistic linguistic patterns.
    """
    print("  Generating synthetic dataset …")

    fake_templates = [
        "BREAKING: Secret government documents reveal massive cover-up about {topic}. Officials scrambling to hide truth.",
        "SHOCKING: You won't believe what {person} just admitted about the {topic} conspiracy. The deep state is furious.",
        "They don't want you to know the REAL truth about {topic}. Whistleblower exposes everything.",
        "ALERT: {person} caught in massive fraud scheme involving {topic}. Mainstream media ignoring this.",
        "EXPOSED: The globalist agenda behind {topic} finally revealed by independent sources. Share before deleted.",
        "MIRACLE CURE: Scientists suppressed {topic} treatment that cures everything. Big pharma panicking.",
        "BOMBSHELL: {person} secretly working with foreign enemies to destroy America through {topic}.",
        "WARNING: Government planning to use {topic} to control the population. They're hiding it from you.",
        "100% PROOF: {topic} is a complete hoax staged by the elite to distract from real agenda.",
        "URGENT: Share this before it gets banned! {person} confesses to {topic} conspiracy on leaked tape.",
        "The truth about {topic} they never taught you in school. Your whole life has been a lie.",
        "OUTRAGE: {person} receives secret payments to push {topic} propaganda on unsuspecting public.",
        "LEAKED: Internal documents prove {topic} is completely fabricated to control public opinion.",
        "Doctors HATE this one weird trick that {person} discovered about {topic}. Banned in 12 countries.",
        "Deep state puppets caught planting fake evidence about {topic}. Whistleblower speaks exclusively.",
        "CONFIRMED: {person} admits in private meeting that {topic} statistics are completely made up.",
        "Elite bankers using {topic} scheme to steal trillions from ordinary Americans. Wake up sheeple.",
        "HIDDEN CAMERA: {person} caught saying one thing publicly about {topic} while doing the opposite.",
        "The {topic} lie is collapsing. Establishment desperately trying to suppress this explosive truth.",
        "NASA scientist admits {topic} data was faked to secure billions in government funding.",
        "EXCLUSIVE: {person} flees country after exposing the {topic} fraud that shook the deep state.",
        "Ancient secret about {topic} hidden for centuries finally revealed. Elites are terrified.",
        "MUST WATCH: {person} destroys {topic} narrative in explosive interview media refused to air.",
        "The great {topic} deception exposed: everything you know is wrong and they planned it this way.",
        "BREAKING NEWS: {person} announces shocking {topic} revelation that will change everything forever.",
    ]

    real_templates = [
        "{person} announced {topic} policy changes during a press conference on {day}, citing economic data.",
        "According to officials, the {topic} initiative has reached its quarterly targets, as reported by {agency}.",
        "Researchers at {university} published findings on {topic} in the peer-reviewed journal Nature this week.",
        "The {agency} confirmed that {topic} regulations will take effect next quarter following legislative approval.",
        "Officials said {topic} funding increased by 12 percent in fiscal year, according to budget documents.",
        "{person} met with international counterparts to discuss {topic} cooperation agreements on {day}.",
        "A new study published in The Lancet found that {topic} interventions reduced cases by 23 percent.",
        "The Senate passed legislation addressing {topic} concerns by a vote of 68 to 32 on {day}.",
        "{agency} released its annual report on {topic}, showing steady improvement over the past five years.",
        "Markets responded positively to {person}'s announcement regarding {topic} economic reforms.",
        "Health officials from {agency} recommended updated {topic} guidelines based on recent clinical data.",
        "The {university} study examined {topic} trends across 45 countries and found statistically significant results.",
        "Local government approved a $2.4 billion investment in {topic} infrastructure over the next decade.",
        "{person} testified before Congress about ongoing efforts to address {topic} challenges facing the nation.",
        "International observers confirmed that {topic} standards were met during the process, {agency} said.",
        "Scientists from {university} announced a breakthrough in {topic} research that could benefit millions.",
        "The Federal Reserve released data showing {topic} indicators remained stable in the third quarter.",
        "{agency} deployed additional resources to address {topic} concerns raised by local communities.",
        "Officials from three agencies collaborated to release a comprehensive report on {topic} outcomes.",
        "A bipartisan committee approved new {topic} funding measures after months of negotiation.",
        "{person} outlined a five-year plan addressing {topic} sustainability during the annual summit.",
        "The World Health Organization published updated guidelines on {topic} following expert review.",
        "Court documents released on {day} showed details of the {topic} case that prosecutors presented.",
        "The independent audit confirmed that {topic} procedures were followed in accordance with regulations.",
        "Economic indicators released by {agency} showed moderate growth in the {topic} sector this quarter.",
    ]

    topics   = ["climate change", "election security", "vaccine research", "economic policy",
                 "immigration reform", "healthcare reform", "tax legislation", "foreign policy",
                 "cybersecurity", "infrastructure investment", "education funding", "trade agreements",
                 "energy policy", "housing market", "unemployment statistics", "federal budget",
                 "public health", "national security", "financial regulation", "environmental protection"]
    persons  = ["The President", "The Secretary of State", "Senator Johnson", "Governor Smith",
                 "The Prime Minister", "The Director", "Representative Davis", "The Chancellor",
                 "The CEO", "The Commissioner", "The Surgeon General", "The Treasurer",
                 "Secretary Williams", "Ambassador Chen", "Senator Martinez"]
    agencies = ["the CDC", "the EPA", "the FDA", "the FBI", "the WHO", "the CBO",
                 "the Treasury", "the Pentagon", "the State Department", "the NRC",
                 "the FTC", "the Department of Justice", "federal investigators"]
    univs    = ["Harvard University", "MIT", "Stanford University", "Johns Hopkins",
                "Oxford University", "Yale University", "Columbia University", "UC Berkeley",
                "the University of Michigan", "Princeton University"]
    days     = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "last week", "yesterday", "earlier this month"]

    rng = np.random.default_rng(42)

    def fill(template):
        return (template
                .replace('{topic}',    rng.choice(topics))
                .replace('{person}',   rng.choice(persons))
                .replace('{agency}',   rng.choice(agencies))
                .replace('{university}', rng.choice(univs))
                .replace('{day}',      rng.choice(days)))

    fake_rows, real_rows = [], []
    samples_each = 1500

    for _ in range(samples_each):
        t = rng.choice(fake_templates)
        fake_rows.append({'text': fill(t), 'label': 'FAKE'})

    for _ in range(samples_each):
        t = rng.choice(real_templates)
        real_rows.append({'text': fill(t), 'label': 'REAL'})

    df = pd.DataFrame(fake_rows + real_rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  ✓ Synthetic dataset: {len(df):,} rows")
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw DataFrame to have exactly two columns:
        'text'  — combined news content
        'label' — 'REAL' or 'FAKE' (uppercase)

    Handles multiple known column-name conventions.
    """
    df.columns = df.columns.str.strip().str.lower()

    # Determine text column
    if 'text' in df.columns and 'title' in df.columns:
        df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'title' in df.columns:
        df['text'] = df['title'].fillna('')
    elif 'text' not in df.columns:
        raise ValueError(f"No usable text column found. Columns: {list(df.columns)}")

    # Determine label column
    label_col = None
    for col in ['label', 'class', 'target', 'fake']:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    df = df[['text', label_col]].rename(columns={label_col: 'label'})
    df['label'] = df['label'].astype(str).str.strip().str.upper()

    # Normalize various label formats → REAL / FAKE
    label_map = {
        '0': 'FAKE', '1': 'REAL',
        'FALSE': 'FAKE', 'TRUE': 'REAL',
        'FAKE': 'FAKE', 'REAL': 'REAL',
    }
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    df = df[df['label'].isin(['REAL', 'FAKE'])]
    return df.reset_index(drop=True)


# ─── Main Training Pipeline ───────────────────────────────────────────────────
def main():
    banner("FAKE NEWS DETECTION — ML TRAINING PIPELINE")

    # ── 1. Load Dataset ──────────────────────────────────────────────────────
    banner("Step 1 / 5 — Loading Dataset")
    df = None

    # Try local disk first
    df = load_dataset_from_disk()

    # Try public URLs
    if df is None:
        for url in DATASET_URLS:
            df = load_dataset_from_url(url)
            if df is not None:
                break

    # Fallback to synthetic
    if df is None:
        print("  ⚠  Could not load dataset from disk or network.")
        df = generate_synthetic_dataset()

    # Normalize columns
    df = normalize_dataframe(df)

    # Drop empty texts
    df = df[df['text'].str.strip().str.len() > 20].reset_index(drop=True)

    print(f"\n  Dataset summary:")
    print(f"  Total samples : {len(df):,}")
    print(f"  REAL          : {(df['label'] == 'REAL').sum():,}")
    print(f"  FAKE          : {(df['label'] == 'FAKE').sum():,}")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    banner("Step 2 / 5 — Text Preprocessing")
    print("  Cleaning and stemming text … (may take a moment)")
    t0 = time.time()
    df['clean_text'] = preprocess_batch(df['text'].tolist())
    df = df[df['clean_text'].str.strip().str.len() > 5].reset_index(drop=True)
    print(f"  ✓ Preprocessed {len(df):,} samples in {time.time()-t0:.1f}s")

    # Encode labels: REAL=1, FAKE=0
    y = (df['label'] == 'REAL').astype(int).values

    # ── 3. TF-IDF Vectorization ──────────────────────────────────────────────
    banner("Step 3 / 5 — TF-IDF Vectorization")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        analyzer='word',
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)
    print(f"  ✓ Vocabulary size : {len(vectorizer.vocabulary_):,}")
    print(f"  ✓ Train samples   : {X_train.shape[0]:,}")
    print(f"  ✓ Test samples    : {X_test.shape[0]:,}")
    print(f"  ✓ Feature matrix  : {X_train.shape[1]:,} features")

    # ── 4. Train & Evaluate Models ───────────────────────────────────────────
    banner("Step 4 / 5 — Training & Evaluating Models")

    classifiers = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs', random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'SVM': CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, random_state=42)
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=None, min_samples_split=2,
            n_jobs=-1, random_state=42
        ),
    }

    results = {}
    trained_models = {}

    for name, clf in classifiers.items():
        print(f"\n  Training {name} …", end='', flush=True)
        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed = time.time() - t0

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy':         round(float(acc),  4),
            'precision':        round(float(prec), 4),
            'recall':           round(float(rec),  4),
            'f1_score':         round(float(f1),   4),
            'confusion_matrix': cm,
            'train_time_sec':   round(elapsed, 2),
        }
        trained_models[name] = clf

        print(f" done in {elapsed:.1f}s")
        print(f"    Accuracy={acc:.4f}  Precision={prec:.4f}  "
              f"Recall={rec:.4f}  F1={f1:.4f}")

    # ── 5. Select Best Model & Save ──────────────────────────────────────────
    banner("Step 5 / 5 — Saving Best Model")

    best_name  = max(results, key=lambda k: results[k]['f1_score'])
    best_model = trained_models[best_name]

    print(f"\n  🏆  Best model: {best_name}  "
          f"(F1 = {results[best_name]['f1_score']:.4f})")

    print("\n  Full comparison:")
    print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for name, m in results.items():
        marker = ' ◄' if name == best_name else ''
        print(f"  {name:<22} {m['accuracy']:6.4f} {m['precision']:6.4f} "
              f"{m['recall']:6.4f} {m['f1_score']:6.4f}{marker}")

    # Save artifacts
    vec_path   = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    meta_path  = os.path.join(MODELS_DIR, 'metrics.json')

    joblib.dump(vectorizer,  vec_path)
    joblib.dump(best_model,  model_path)

    metadata = {
        'best_model':    best_name,
        'dataset_size':  int(len(df)),
        'train_size':    int(X_train.shape[0]),
        'test_size':     int(X_test.shape[0]),
        'vocab_size':    int(len(vectorizer.vocabulary_)),
        'models':        results,
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✓ Vectorizer saved   → {vec_path}")
    print(f"  ✓ Best model saved   → {model_path}")
    print(f"  ✓ Metrics saved      → {meta_path}")

    banner("TRAINING COMPLETE — Ready to serve predictions!")

    # Print full classification report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\nClassification Report — {best_name}:\n")
    print(classification_report(y_test, y_pred_best, target_names=['FAKE', 'REAL']))


if __name__ == '__main__':
    main()
