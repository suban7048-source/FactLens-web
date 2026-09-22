# 🔍 FactLens — Fake News Detection System

> An AI-powered web application that uses Machine Learning to detect fake news with high confidence. Built with Python, Flask, Scikit-learn, NLTK, and a modern responsive frontend.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip
- Internet connection (for dataset download on first training)

### 1. Install Dependencies

```bash
cd backend
python -m pip install --prefer-binary -r requirements.txt
```

### 2. Train the ML Models

```bash
cd backend
python -X utf8 train_model.py
```

This will:
- Download the fake/real news dataset automatically (~6,300 samples)
- Preprocess all text (lowercase, remove stopwords, stem)
- Train 4 classifiers with TF-IDF vectorization
- Compare all models (accuracy, precision, recall, F1)
- Save the best model to `backend/models/`

Expected output:
```
────────────────────────────────────────────────────────────
  FAKE NEWS DETECTION — ML TRAINING PIPELINE
────────────────────────────────────────────────────────────

  Step 1 / 5 — Loading Dataset
  ...
  🏆  Best model: SVM  (F1 = 0.9621)
  
  ✓ Vectorizer saved   → backend/models/vectorizer.pkl
  ✓ Best model saved   → backend/models/best_model.pkl
  ✓ Metrics saved      → backend/models/metrics.json
```

### 3. Start the Flask Server

```bash
cd backend
python -X utf8 app.py
```

### 4. Open the App

Visit: **http://localhost:5000**

---

## 📁 Project Structure

```
Fake News Web/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # ML training pipeline
│   ├── preprocess.py       # Text cleaning utilities
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # Saved model files (auto-created)
│   │   ├── vectorizer.pkl  # Fitted TF-IDF vectorizer
│   │   ├── best_model.pkl  # Best classifier
│   │   └── metrics.json    # Model comparison metrics
│   └── dataset/            # Optional: place CSV files here
│       ├── Fake.csv         # (Kaggle format, optional)
│       └── True.csv         # (Kaggle format, optional)
├── frontend/
│   ├── index.html          # Single-page app
│   ├── css/
│   │   └── style.css       # Full design system
│   └── js/
│       └── app.js          # Frontend logic
└── README.md
```

---

## 🤖 Machine Learning Pipeline

### Text Preprocessing
1. Lowercase normalization
2. HTML tag removal
3. URL and email stripping
4. Punctuation and digit removal
5. NLTK stopword removal
6. Porter Stemmer normalization

### TF-IDF Vectorization
- `max_features=15,000`
- `ngram_range=(1, 2)` — unigrams + bigrams
- `sublinear_tf=True` — log-scaled TF
- `min_df=2` — ignore rare terms

### Models Trained
| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classifier with L2 regularization |
| Naive Bayes | Multinomial NB with Laplace smoothing |
| SVM | Linear SVC with Platt calibration |
| Random Forest | 150 trees, ensemble classifier |

### Model Selection
The model with the highest **F1-Score** on the test set is automatically selected and saved.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serve frontend |
| `POST` | `/api/predict` | Classify news text |
| `GET`  | `/api/metrics` | Model comparison metrics |
| `GET`  | `/api/health` | Health check |

### POST `/api/predict`

**Request:**
```json
{
  "text": "Scientists at MIT announced a breakthrough in renewable energy..."
}
```

**Response:**
```json
{
  "label": "REAL",
  "confidence": 0.9421,
  "model": "SVM",
  "explanation": "This text exhibits hallmarks of credible journalism..."
}
```

---

## 💻 Features

- ✅ **Real-time classification** — instant results via Flask API
- 🎨 **Dark/Light mode** — persisted in localStorage
- 📊 **Model comparison** — interactive Chart.js bar charts
- 🔢 **Confusion matrix** — visual TP/TN/FP/FN breakdown
- 📱 **Fully responsive** — mobile, tablet, desktop
- ⚡ **Animated confidence ring** — SVG-based progress visualization
- 📝 **Sample texts** — built-in fake/real examples to try
- ⌨️ **Keyboard shortcut** — `Ctrl+Enter` to analyze
- 🔒 **Input validation** — both client-side and server-side

---

## 🔧 Optional: Use Kaggle Dataset

For the official Kaggle fake news dataset:

1. Download from: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
2. Place `Fake.csv` and `True.csv` in `backend/dataset/`
3. Run `python train_model.py` — it will detect and use these files automatically

---

## 📊 Expected Model Performance

> Results vary by dataset. Typical performance on the fake/real news dataset:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~95% | ~0.95 |
| Naive Bayes | ~93% | ~0.93 |
| **SVM** | **~96%** | **~0.96** |
| Random Forest | ~94% | ~0.94 |

---

## 🛠️ Troubleshooting

**Training fails on Windows (Unicode error):**
→ Use `python -X utf8 train_model.py` instead of `python train_model.py`

**"Model not loaded" error:****
→ Run `python train_model.py` first

**"Cannot connect to API":**
→ Make sure `python app.py` is running and visit http://localhost:5000

**Training fails to download dataset:**
→ The script will automatically fall back to a synthetic dataset

**NLTK data not found:**
→ Run `python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"`

---

## 📝 License

MIT License — Free for educational and personal use.

---