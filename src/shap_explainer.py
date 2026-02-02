import shap
import pandas as pd
import numpy as np
from datetime import datetime

from src.model_loader import load_model
from src.features import build_features
from src.db import explanations_col


# ⚠️ DO NOT load model at import time
_model = None
_preprocessor = None
_classifier = None
_explainer = None


def get_explainer():
    global _model, _preprocessor, _classifier, _explainer

    if _explainer is None:
        _model = load_model()
        _preprocessor = _model.named_steps["preprocessor"]
        _classifier = _model.named_steps["model"]

        # Tree-based model explainer
        _explainer = shap.TreeExplainer(_classifier)

    return _explainer, _preprocessor, _classifier


def explain_customer(input_data: dict):
    """
    Returns top SHAP features for a single customer
    """

    explainer, preprocessor, classifier = get_explainer()

    # 1️⃣ Input → DataFrame
    df = pd.DataFrame([input_data])

    # 2️⃣ Training parity
    if "Response" not in df.columns:
        df["Response"] = 0

    # 3️⃣ Feature engineering
    df = build_features(df)

    # 4️⃣ Preprocess only
    X = preprocessor.transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    # 5️⃣ SHAP values
    shap_values = explainer.shap_values(X)

    # 6️⃣ Get predicted class
    predicted_class = classifier.predict(X)[0]
    class_index = list(classifier.classes_).index(predicted_class)

    # shape: (1, n_features, n_classes)
    shap_for_class = shap_values[0, :, class_index]

    feature_names = preprocessor.get_feature_names_out()

    # 7️⃣ Build explanation
    explanation = dict(zip(feature_names, shap_for_class))

    # 8️⃣ Sort by absolute impact
    explanation = dict(
        sorted(
            explanation.items(),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )
    )

    # 9️⃣ Keep top 10 only
    top_explanation = dict(list(explanation.items())[:10])

    # 🔟 Save to MongoDB
    explanations_col.insert_one({
        "input": input_data,
        "prediction": predicted_class,
        "explanation": {
            k: float(v) for k, v in top_explanation.items()
        },
        "model": "customer_categorizer_v1",
        "created_at": datetime.utcnow()
    })

    return top_explanation
