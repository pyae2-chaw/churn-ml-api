# app.py
import os
from datetime import timedelta
from dotenv import load_dotenv

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# --- ML / IO imports (keep your real model & DB here) ---
import pandas as pd
import joblib
from pymongo import MongoClient
# from xgboost import XGBClassifier  # if you load a trained model

load_dotenv()

# ===========================
# Basic app & security config
# ===========================
app = Flask(__name__)

# Max upload size: 15 MB
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

# If you ever set cookies from this service:
app.secret_key = os.getenv("SECRET_KEY", "change-me")
app.permanent_session_lifetime = timedelta(days=7)

# ===========================
# CORS (credentialed & strict)
# ===========================
ALLOWED_ORIGINS = [
    "https://churn-client.vercel.app",
    "http://localhost:5173",
]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True,                       # allow cookies if you need them
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["Content-Type"],
)

# ===========================
# DB (optional, keep your own)
# ===========================
MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = mongo_client["ml_app"] if mongo_client else None
results_collection = db["results"] if db is not None else None

# ===========================
# Model (example load)
# ===========================
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_order.pkl")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
feature_order = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None

# ===========================
# Helpers
# ===========================
def _is_origin_allowed(origin: str) -> bool:
    return origin in ALLOWED_ORIGINS

def _require_auth():
    """
    Minimal auth gate:
    - Prefer JWT: 'Authorization: Bearer <token>' (validate here if you want)
    - Or cookie-based: check request.cookies.get('session') or similar
    """
    auth = request.headers.get("Authorization", "")
    cookie_session = request.cookies.get("session")  # example cookie name

    # If you use JWT across domains:
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
        # TODO: validate token (e.g., signature, expiry)
        return True

    # If you rely on cookie from THIS domain (onrender.com ML API):
    if cookie_session:
        # TODO: verify cookie (if you set it on this service)
        return True

    return False

# ===========================
# Routes
# ===========================
@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "ml-api"}), 200

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # OPTIONS is handled by flask-cors, but this early return is harmless
    if request.method == "OPTIONS":
        return ("", 204)

    # --- Auth gate ---
    if not _require_auth():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    # --- Read file ---
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file field 'file' in form-data"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        # Read CSV into DataFrame
        df = pd.read_csv(f)

        # (Optional) align features for model
        if model is None:
            # Demo output so the endpoint works even without a model
            demo = [{"index": int(i), "churn_pred": 0, "prob": 0.1} for i in range(len(df))]
            result = {"success": True, "predictions": demo}

        else:
            X = df[feature_order] if feature_order is not None else df
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1].tolist()
            else:
                # Some models (e.g., SVM without prob) won’t have predict_proba
                preds_raw = model.predict(X)
                probs = [float(x) for x in preds_raw]

            preds = model.predict(X).tolist()
            result_rows = []
            for i, (p, pr) in enumerate(zip(preds, probs)):
                result_rows.append({"index": int(i), "churn_pred": int(p), "prob": float(pr)})

            result = {"success": True, "predictions": result_rows}

        # (Optional) Save latest result per user in Mongo by user_id
        user_id = request.args.get("user_id") or request.form.get("user_id")
        if results_collection is not None and user_id:
            results_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "user_id": user_id,
                    "result_json": result,
                }},
                upsert=True,
            )

        return jsonify(result), 200

    except pd.errors.EmptyDataError:
        return jsonify({"success": False, "error": "Uploaded file is empty or invalid CSV"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/results/latest", methods=["GET", "OPTIONS"])
def results_latest():
    if request.method == "OPTIONS":
        return ("", 204)

    user_id = request.args.get("user_id")
    if results_collection is None or not user_id:
        return jsonify({"success": False, "error": "Results not available"}), 400

    doc = results_collection.find_one({"user_id": user_id}, {"_id": 0})
    if not doc:
        return jsonify({"success": False, "error": "No results found for user"}), 404
    return jsonify({"success": True, "data": doc.get("result_json")}), 200

# ===========================
# Example route: set cookie (ONLY if you use cookie auth for this domain)
# ===========================
@app.route("/auth/mock-login", methods=["POST"])
def mock_login():
    # If you want this API to set a cookie for itself (onrender.com)
    resp = make_response(jsonify({"success": True}))
    # IMPORTANT: Secure + SameSite=None for cross-site cookie
    resp.set_cookie(
        "session", "example-session-id",
        max_age=7 * 24 * 3600,
        secure=True,
        httponly=True,
        samesite="None",
        path="/",
    )
    return resp

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
