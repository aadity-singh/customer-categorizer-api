import joblib
from pathlib import Path

# Absolute-safe path (prevents Windows / reload issues)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "customer_categorizer.pkl"

_model = None  # cache model (important)

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model
