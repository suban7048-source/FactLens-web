"""
app.py — Fake News Detection System — Flask API
================================================
Serves the frontend SPA and exposes prediction + metrics API endpoints.

Endpoints:
    GET  /                   — Frontend HTML
    POST /api/predict        — Predict real/fake for submitted text
    GET  /api/metrics        — Model comparison metrics
    GET  /api/health         — Health check

Start the server:
    python app.py
"""

import os
import json
import math
import joblib

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

from preprocess import preprocess_text

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR  = os.path.join(BASE_DIR, '..', 'frontend')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
METRICS_PATH  = os.path.join(MODELS_DIR, 'metrics.json')
VEC_PATH      = os.path.join(MODELS_DIR, 'vectorizer.pkl')
MODEL_PATH    = os.path.join(MODELS_DIR, 'best_model.pkl')

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.abspath(FRONTEND_DIR),
    static_url_path=''
)
CORS(app)

# ─── Load Model at Startup ────────────────────────────────────────────────────
vectorizer  = None
best_model  = None
best_name   = 'Unknown'
metrics_data = {}

def load_model():
    global vectorizer, best_model, best_name, metrics_data

    if not os.path.exists(VEC_PATH) or not os.path.exists(MODEL_PATH):
        print("⚠  Model files not found. Please run train_model.py first.")
        return False

    vectorizer = joblib.load(VEC_PATH)
    best_model = joblib.load(MODEL_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
        best_name = metrics_data.get('best_model', 'Unknown')

    try:
        print(f"[+] Loaded model: {best_name}")
        print(f"[+] Vectorizer vocabulary: {len(vectorizer.vocabulary_):,} terms")
    except Exception:
        pass
    return True

# Initialize model at startup
load_model()


def get_confidence(model, features) -> float:
    """
    Extract a calibrated [0, 1] confidence score from any sklearn classifier.
    Handles predict_proba models and decision_function (LinearSVC/CalibratedCV).
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)[0]
        return float(max(probs))

    if hasattr(model, 'decision_function'):
        score = float(model.decision_function(features)[0])
        # Platt scaling approximation: sigmoid of absolute distance
        confidence = 1.0 / (1.0 + math.exp(-abs(score)))
        return min(max(float(confidence), 0.50), 0.99)

    return 0.75  # fallback


def build_explanation(label: str, confidence: float) -> str:
    """Generate a human-readable explanation based on prediction and confidence."""
    conf_pct = int(confidence * 100)

    if label == 'FAKE':
        if conf_pct >= 90:
            return (
                "This text exhibits strong indicators of misinformation: "
                "sensationalist language, emotional manipulation, absence of "
                "verifiable sources, and patterns commonly found in fabricated news."
            )
        elif conf_pct >= 75:
            return (
                "This text contains several patterns associated with fake news, "
                "including unverified claims, inflammatory phrasing, and lack of "
                "credible attribution."
            )
        else:
            return (
                "This text shows some indicators of potentially misleading content. "
                "We recommend verifying the information with trusted news sources."
            )
    else:  # REAL
        if conf_pct >= 90:
            return (
                "This text exhibits hallmarks of credible journalism: neutral tone, "
                "factual reporting style, proper source attribution, and language "
                "consistent with legitimate news outlets."
            )
        elif conf_pct >= 75:
            return (
                "This text contains patterns characteristic of authentic news reporting, "
                "including structured presentation and measured language."
            )
        else:
            return (
                "This text appears to be legitimate news, though with moderate confidence. "
                "Always cross-reference important information with multiple trusted sources."
            )


# ─── Frontend Routes ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    full = os.path.join(app.static_folder, path)
    if os.path.exists(full):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


# ─── API Routes ───────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':      'ok',
        'model_ready': best_model is not None,
        'model':       best_name,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if best_model is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    text = data.get('text', '').strip()

    # Input validation
    if not text:
        return jsonify({'error': 'Please provide a non-empty "text" field.'}), 400
    if len(text) < 15:
        return jsonify({'error': 'Text is too short. Please provide at least 15 characters.'}), 400
    if len(text) > 50_000:
        return jsonify({'error': 'Text is too long. Please limit to 50,000 characters.'}), 400

    # Preprocess
    cleaned = preprocess_text(text)
    if not cleaned or len(cleaned.split()) < 2:
        return jsonify({
            'error': 'Could not extract meaningful content from the provided text.'
        }), 422

    # Vectorize
    features = vectorizer.transform([cleaned])

    # Predict
    label_int  = int(best_model.predict(features)[0])
    label      = 'REAL' if label_int == 1 else 'FAKE'
    confidence = get_confidence(best_model, features)
    explanation = build_explanation(label, confidence)

    return jsonify({
        'label':       label,
        'confidence':  round(confidence, 4),
        'model':       best_name,
        'explanation': explanation,
    })


@app.route('/api/metrics', methods=['GET'])
def metrics():
    if not metrics_data:
        return jsonify({'error': 'Metrics not available. Run train_model.py first.'}), 503

    return jsonify(metrics_data)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Fake News Detection System — Flask API")
    print("=" * 60)

    load_model()

    print("\n  Server starting at: http://localhost:5000")
    print("  Press Ctrl+C to stop.\n")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
    )
