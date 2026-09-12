"""
phm_pipeline.py
===============
Single shared inference path for the PHM stack (CLI demo, Streamlit
dashboard, FastAPI service). Guarantees training-serving parity: the same
physics feature layer (features.py) and the same artifact dispatch used
in training are applied at inference.

Artifacts (models/):
  scaler_ad.joblib, ad_model_type.joblib (mahalanobis|pca|iforest|autoencoder),
  ad_threshold.joblib, model_ad.joblib (kind-specific payload),
  model_fc_xgboost.joblib, label_encoder_fault.joblib, model_rul_xgboost.joblib
"""
import os
import numpy as np
import pandas as pd
import joblib

from features import FEATURES_RAW, FEATURES_AD, FEATURES_FAULT, FEATURES_RUL, engineer_features


def _load(path, fallback=None):
    try:
        return joblib.load(path)
    except Exception:
        if fallback is not None:
            return joblib.load(fallback)
        raise


def load_artifacts(models_dir="models"):
    """Load the full PHM artifact set into a dict."""
    j = lambda n: os.path.join(models_dir, n)
    ad_kind = str(joblib.load(j("ad_model_type.joblib")))
    try:
        ad_model = joblib.load(j("model_ad.joblib"))
    except Exception:
        legacy = {"iforest": "model_ad_iforest.joblib", "autoencoder": "model_ad_autoencoder.joblib",
                  "pca": "model_ad_pca.joblib", "mahalanobis": "model_ad_mahal.joblib"}
        ad_model = joblib.load(j(legacy.get(ad_kind, "model_ad_iforest.joblib")))
    try:
        thr = joblib.load(j("ad_threshold.joblib"))
        ad_thr = float(np.asarray(thr).ravel()[0])
    except Exception:
        ad_thr = float(joblib.load(j("ad_thresholds_both.joblib"))["threshold"])
    # Supervised known-fault head (v6+): primary detector when present.
    # Mahalanobis stays as the novelty watchdog + OOD gate.
    try:
        superv = joblib.load(j("model_superv_ad.joblib"))
    except Exception:
        superv = None
    try:
        superv_thr = float(np.asarray(joblib.load(j("superv_threshold.joblib"))).ravel()[0])
    except Exception:
        superv_thr = 0.5
    return {
        "scaler": joblib.load(j("scaler_ad.joblib")),
        "ad_kind": ad_kind, "ad_model": ad_model, "ad_threshold": ad_thr,
        "superv": superv, "superv_threshold": superv_thr,
        "clf": joblib.load(j("model_fc_xgboost.joblib")),
        "le": joblib.load(j("label_encoder_fault.joblib")),
        "reg": joblib.load(j("model_rul_xgboost.joblib")),
    }


def prepare(raw):
    """raw: dict or DataFrame with FEATURES_RAW (+cycle/scenario when known).
    Returns engineered DataFrame with all model columns."""
    df = pd.DataFrame([raw]) if isinstance(raw, dict) else raw.copy()
    missing = [c for c in FEATURES_RAW if c not in df.columns]
    if missing:
        raise KeyError(f"missing raw signals: {missing}")
    return engineer_features(df)


def ad_scores(art, df_eng):
    """Anomaly scores (higher = more anomalous) for engineered rows."""
    Z = art["scaler"].transform(df_eng[FEATURES_AD])
    kind, m = art["ad_kind"], art["ad_model"]
    if kind == "mahalanobis":
        d = Z - np.asarray(m["mean"])
        return np.einsum("ij,jk,ik->i", d, np.asarray(m["cov_inv"]), d)
    if kind == "pca":
        Zr = m.inverse_transform(m.transform(Z))
        return np.mean(np.square(Z - Zr), axis=1)
    if kind == "iforest":
        return -m.decision_function(Z)
    if kind == "autoencoder":
        return np.mean(np.square(Z - m.predict(Z)), axis=1)
    raise ValueError(f"unknown AD kind: {kind}")


def predict(art, df_eng):
    """Full PHM pass over engineered rows. Returns list of dicts.

    Detection: supervised known-fault head when shipped (probability vs
    operating point); otherwise the Mahalanobis novelty score vs threshold.
    """
    if art.get("superv") is not None:
        scores = np.asarray(art["superv"].predict_proba(df_eng[FEATURES_FAULT])[:, 1])
        is_anom = scores > art.get("superv_threshold", 0.5)
    else:
        scores = ad_scores(art, df_eng)
        is_anom = scores > art["ad_threshold"]
    Xc = df_eng[FEATURES_FAULT]
    enc = art["clf"].predict(Xc)
    labels = art["le"].inverse_transform(enc)
    proba = art["clf"].predict_proba(Xc) if hasattr(art["clf"], "predict_proba") else None
    if "cycle" not in df_eng.columns:
        raise KeyError("'cycle' (elapsed mission time) is required for RUL prediction")
    rul = art["reg"].predict(df_eng[FEATURES_RUL])
    out = []
    for i in range(len(df_eng)):
        out.append({
            "is_anomaly": bool(is_anom[i]),
            "ad_score": float(scores[i]),
            "fault_type": str(labels[i]),
            "confidence": float(proba[i, enc[i]] * 100.0) if proba is not None else float("nan"),
            "rul": float(rul[i]),
        })
    return out


def explain(art, df_eng_row):
    """Top driving features for the predicted fault class.
    Uses SHAP when installed, else falls back to model importances."""
    Xc = df_eng_row[FEATURES_FAULT]
    enc = int(art["clf"].predict(Xc)[0])
    try:
        import shap
        expl = shap.TreeExplainer(art["clf"])
        sv = expl.shap_values(Xc)
        if isinstance(sv, list):
            v = np.asarray(sv[enc][0])
        elif np.ndim(sv) == 3:
            v = np.asarray(sv)[0, :, enc]
        else:
            v = np.asarray(sv)[0]
        ranked = sorted(zip(FEATURES_FAULT, v), key=lambda kv: abs(kv[1]), reverse=True)
        return {"method": "shap", "class": enc, "top": [(k, float(x)) for k, x in ranked[:8]]}
    except Exception:
        imp = getattr(art["clf"], "feature_importances_", None)
        if imp is None:
            return {"method": "unavailable", "class": enc, "top": []}
        ranked = sorted(zip(FEATURES_FAULT, imp), key=lambda kv: kv[1], reverse=True)
        return {"method": "gain-importance", "class": enc, "top": [(k, float(x)) for k, x in ranked[:8]]}
