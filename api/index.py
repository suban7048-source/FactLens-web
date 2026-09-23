import sys
import os

# ── Serverless NLTK data path ─────────────────────────────────────────────────
# In Vercel's serverless environment, only /tmp is writable.
# Set NLTK_DATA to /tmp so downloads go there if needed.
os.environ.setdefault('NLTK_DATA', '/tmp/nltk_data')
import nltk
nltk.data.path.insert(0, '/tmp/nltk_data')

# ── Add backend directory to path ─────────────────────────────────────────────
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app import app
