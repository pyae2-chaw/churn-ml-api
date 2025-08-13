# app.py
import os
from datetime import timedelta
from dotenv import load_dotenv

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import joblib
from pymongo import MongoClient

load_dotenv()

# ===========================
# Basic app & security config
# ===========================
app = Flask(__name__)

# Max upload size: 15 MB
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

# If you ever set cookies from this service (not needed for Option A)
app.secret_key = os.getenv("SECRET_KEY", "change-me")
app.permanent_session_lifetime = timedelta(days=7)

# Friendly error for 413
@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": "File too large. Max 15 MB."}), 413

# ===========================
# CORS for JWT (no cookies)
# ===========================
ALLOWED_ORIGINS = [
    "https://churn-client.vercel.app",
    "http://localhost:5173",
    # add your custom domain here if you have one, e.g. "https://yourdomain.com",
]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,  # safe to keep on; we're not sending cookies anyway
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Type"],
)

# ===========================
# DB (optional)
# ===========================
MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = mongo_client["ml_app"] if mongo_client else None
results_collection = db["results"] if db is not None else None

# ===========================
# Model (LAZY load for fast boot)
# ===========================
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_order.pkl")

_model = None
_feature_order = None

def get_model():
    """Lazy-load heavy artifacts only when /predict is actually called."""
    global _model, _feature_order
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        if os.path.exists(FEATURES_PATH):
            _feature_order = joblib.load(FEATURES_PATH)
    return _model, _feature_order

# ===========================
# Auth helper (JWT via Authorization header)
# ===========================
def _require_auth():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
        # TODO: verify token signature/expiry if you need strict security
        return True
    return False

# ===========================
# Health (keep instant)
# ===========================
@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify({"ok": True, "service": "ml-api"}), 200

@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"ok": True}), 200

# ===========================
# Predict
# ===========================
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Preflight should be fast; flask-cors handles most headers
    if request.method == "OPTIONS":
        return ("", 204)

    # --- Auth gate (JWT) ---
    if not _require_auth():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    # --- File checks ---
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file field 'file' in form-data"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        df = pd.read_csv(f)

        model, feature_order = get_model()
        if model is None:
            # Demo output to keep endpoint functional without a model
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

        # Optional: store latest result
        user_id = request.args.get("user_id") or request.form.get("user_id")
        if results_collection is not None and user_id:
            results_collection.update_one(
                {"user_id": user_id},
                {"$set": {"user_id": user_id, "result_json": result}},
                upsert=True,
            )

        return jsonify(result), 200

    except pd.errors.EmptyDataError:
        return jsonify({"success": False, "error": "Uploaded file is empty or invalid CSV"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ===========================
# Latest results
# ===========================
@app.route("/results/latest", methods=["GET", "OPTIONS"])
def results_latest():
    if request.method == "OPTIONS":
        return ("", 204)

    # Optional (recommended): protect with JWT as well
    if not _require_auth():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    user_id = request.args.get("user_id")
    if results_collection is None or not user_id:
        return jsonify({"success": False, "error": "Results not available"}), 400

    doc = results_collection.find_one({"user_id": user_id}, {"_id": 0})
    if not doc:
        return jsonify({"success": False, "error": "No results found for user"}), 404

    return jsonify({"success": True, "data": doc.get("result_json")}), 200

# ===========================
# Cookie demo (unused in Option A)
# ===========================
@app.route("/auth/mock-login", methods=["POST"])
def mock_login():
    resp = make_response(jsonify({"success": True}))
    resp.set_cookie(
        "session", "example-session-id",
        max_age=7 * 24 * 3600,
        secure=True, httponly=True, samesite="None", path="/",
    )
    return resp

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
