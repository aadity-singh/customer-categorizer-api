import shap
import pandas as pd
import numpy as np

from src.model_loader import load_model
from src.features import build_features

# Load model once
_model = load_model()
_preprocessor = _model.named_steps["preprocessor"]
_classifier = _model.named_steps["model"]

# SHAP explainer (tree-based model)
_explainer = shap.TreeExplainer(_classifier)


def explain_customer(input_data: dict):
    """
    Returns top SHAP features for a single customer
    """

    # 1️⃣ Input → DataFrame
    df = pd.DataFrame([input_data])

    # 2️⃣ Training parity
    if "Response" not in df.columns:
        df["Response"] = 0

    # 3️⃣ Feature engineering
    df = build_features(df)

    # 4️⃣ Preprocess only
    X = _preprocessor.transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    # 5️⃣ SHAP values
    shap_values = _explainer.shap_values(X)

    # 6️⃣ Get predicted class
    predicted_class = _classifier.predict(X)[0]
    class_index = list(_classifier.classes_).index(predicted_class)

    # ✅ CORRECT extraction for 3D SHAP output
    # shape: (1, n_features, n_classes)
    shap_for_class = shap_values[0, :, class_index]

    feature_names = _preprocessor.get_feature_names_out()

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

    # Optional: return only top 10
    return dict(list(explanation.items())[:10])
