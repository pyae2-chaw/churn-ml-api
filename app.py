# app.py
import os
from datetime import timedelta
from dotenv import load_dotenv

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import joblib

load_dotenv()

app = Flask(__name__)

# ---------- Basic config ----------
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
app.permanent_session_lifetime = timedelta(days=7)

SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "session")
ENV = os.getenv("ENV", "development").lower()  # "development" or "production"
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", None)  # e.g. "onrender.com" or "ml.yourdomain.com"

# CORS: from env (comma-separated). Always include localhost for dev.
default_origins = ["http://localhost:5173"]
extra_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = list(dict.fromkeys(default_origins + extra_origins))  # dedupe, keep order

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,  # required for cookies
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Type"],
)

# Friendly 413
@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": "File too large. Max 15 MB."}), 413

# ---------- Model (optional; lazy load is fine) ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_order.pkl")

_model = None
_feature_order = None

def get_model():
    global _model, _feature_order
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    if _feature_order is None and os.path.exists(FEATURES_PATH):
        _feature_order = joblib.load(FEATURES_PATH)
    return _model, _feature_order

# ---------- Auth helpers (cookie) ----------
def _has_session_cookie() -> bool:
    return bool(request.cookies.get(SESSION_COOKIE_NAME))

def _set_session_cookie(resp, value: str):
    """
    Set cookie correctly for dev vs prod:
    - DEV (localhost): SameSite=Lax, Secure=False
    - PROD (cross-site): SameSite=None, Secure=True
    """
    if ENV == "production":
        resp.set_cookie(
            SESSION_COOKIE_NAME, value,
            max_age=7 * 24 * 3600,
            httponly=True,
            secure=True,           # required for SameSite=None on modern browsers
            samesite="None",       # allow cross-site (vercel.app ↔ onrender.com)
            domain=COOKIE_DOMAIN,  # optional; omit unless you know you need it
            path="/",
        )
    else:
        resp.set_cookie(
            SESSION_COOKIE_NAME, value,
            max_age=7 * 24 * 3600,
            httponly=True,
            secure=False,
            samesite="Lax",
            path="/",
        )
    return resp

# ---------- Basic routes / health ----------
@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify({"ok": True, "service": "ml-api"}), 200

@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"ok": True}), 200

# Mock login (useful in dev; also works in prod if you want the ML API to mint its own cookie)
@app.route("/auth/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return ("", 204)
    resp = make_response(jsonify({"success": True, "user": {"id": "cookie-user"}}), 200)
    return _set_session_cookie(resp, "session-id-xyz")

@app.route("/auth/logout", methods=["POST", "OPTIONS"])
def logout():
    if request.method == "OPTIONS":
        return ("", 204)
    resp = make_response(jsonify({"success": True}), 200)
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp

# ---------- Predict (requires cookie) ----------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    if not _has_session_cookie():
        return jsonify({"success": False, "error": "Unauthorized (no session cookie)"}), 401

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file field 'file' in form-data"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        df = pd.read_csv(f)

        model, feature_order = get_model()
        if model is None:
            demo = [{"index": int(i), "churn_pred": 0, "prob": 0.1} for i in range(len(df))]
            result = {"success": True, "predictions": demo}
        else:
            X = df[feature_order] if feature_order is not None else df
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1].tolist()
            else:
                preds_raw = model.predict(X)
                probs = [float(x) for x in preds_raw]
            preds = model.predict(X).tolist()
            rows = [{"index": int(i), "churn_pred": int(p), "prob": float(pr)}
                    for i, (p, pr) in enumerate(zip(preds, probs))]
            result = {"success": True, "predictions": rows}

        # simple memory storage for latest result
        app.last_result = result
        return jsonify(result), 200

    except pd.errors.EmptyDataError:
        return jsonify({"success": False, "error": "Uploaded file is empty or invalid CSV"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------- Latest results (requires cookie) ----------
@app.route("/results/latest", methods=["GET", "OPTIONS"])
def results_latest():
    if request.method == "OPTIONS":
        return ("", 204)

    if not _has_session_cookie():
        return jsonify({"success": False, "error": "Unauthorized (no session cookie)"}), 401

    result = getattr(app, "last_result", None)
    if not result:
        return jsonify({"success": False, "error": "No results found"}), 404

    return jsonify({"success": True, "data": result}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=(ENV != "production"))
