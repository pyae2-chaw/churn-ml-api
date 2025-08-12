# app.py
import os
import re
import io
import uuid
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# ------------------ Setup & env ------------------
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
CLIENT_ORIGIN = os.getenv("CLIENT_ORIGIN", "https://churn-client.vercel.app")  # no trailing slash

# Mongo
client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = client["ml_app"] if client else None
results_collection = db["results"] if db is not None else None

# Quiet some model warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

# ------------------ CORS ------------------
# Allow your prod Vercel site, localhost, and preview deployments
LOCAL = "http://localhost:5173"
_preview_re = re.compile(
    r"^https://churn-client-[\w-]+-pyae-kyi-thar-chaws-projects\.vercel\.app$"
)

def origin_allowed(origin: str) -> bool:
    if not origin:
        return True
    if origin == CLIENT_ORIGIN or origin == LOCAL:
        return True
    if _preview_re.match(origin or ""):
        return True
    return False

CORS(
    app,
    origins=origin_allowed,
    supports_credentials=False,            # ML API doesn't use cookies
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

# ------------------ Paths / Folders ------------------
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
RESULT_FOLDER = os.getenv("RESULT_FOLDER", "/tmp/results")
MODEL_FOLDER  = os.getenv("MODEL_FOLDER", "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ------------------ Models ------------------
def load_models():
    components = {
        "model": joblib.load(os.path.join(MODEL_FOLDER, "model.pkl")),
        "scaler": joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl")),
        "income_bins": joblib.load(os.path.join(MODEL_FOLDER, "income_bins.pkl")),
        "balance_bins": joblib.load(os.path.join(MODEL_FOLDER, "balance_bins.pkl")),
        "outstanding_bins": joblib.load(os.path.join(MODEL_FOLDER, "outstanding_bins.pkl")),
        "income_map": joblib.load(os.path.join(MODEL_FOLDER, "income_map.pkl")),
        "balance_map": joblib.load(os.path.join(MODEL_FOLDER, "balance_map.pkl")),
        "outstanding_map": joblib.load(os.path.join(MODEL_FOLDER, "outstanding_map.pkl")),
    }

    feature_order = [
        "Credit Score",
        "Customer Tenure",
        "Balance Band Encoded",
        "NumOfProducts",
        "Outstanding Loans Band Encoded",
        "Income Band Encoded",
        "Credit History Length",
        "NumComplaints",
    ]
    return components, feature_order

model_components, FEATURE_ORDER = load_models()

# ------------------ Helpers ------------------
def normalize_column(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())

def map_columns(df: pd.DataFrame, expected_features: dict) -> pd.DataFrame:
    norm_user_cols = {normalize_column(c): c for c in df.columns}
    mapped = {}
    for key, variations in expected_features.items():
        for v in variations:
            norm_v = normalize_column(v)
            if norm_v in norm_user_cols:
                mapped[key] = norm_user_cols[norm_v]
                break
    return df.rename(columns={v: k for k, v in mapped.items()})

expected_features = {
    "credit_score": ["credit score", "creditscore", "Credit_Score", "Credit Score"],
    "balance": ["balance", "Balance"],
    "num_products": ["NumOfProducts", "num_products", "number of products"],
    "income": ["income", "annual income", "Income"],
    "customer_tenure": ["Customer Tenure", "tenure", "Tenure", "Years with Bank"],
    "outstanding_loans": ["Outstanding Loans", "outstanding loans", "loan_balance", "outstanding_debt"],
    "credit_history_length": ["Credit History Length", "history length"],
    "num_complaints": ["NumComplaints", "complaints"],
}

def safe_bin(series: pd.Series, bin_edges, labels):
    min_val, max_val = bin_edges[0], bin_edges[-1]
    series_clipped = series.clip(lower=min_val, upper=max_val)
    return pd.cut(series_clipped, bins=bin_edges, labels=labels, include_lowest=True)

# ------------------ Routes ------------------
@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are allowed"}), 400

        # Multipart form also carries user_id
        user_id = request.form.get("user_id", "anonymous")

        # Save upload to /tmp
        file_id = uuid.uuid4().hex
        upload_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
        file.save(upload_path)

        # Read and normalize
        df = pd.read_csv(upload_path)
        df = map_columns(df, expected_features)

        required_cols = [
            "income", "balance", "outstanding_loans",
            "credit_score", "customer_tenure",
            "num_products", "credit_history_length", "num_complaints",
        ]
        df = df.dropna(subset=required_cols)
        if len(df) < 50:
            return jsonify({"error": "Insufficient rows after cleaning. Minimum 50 required."}), 400

        # Binning
        df["income_band"] = safe_bin(df["income"], model_components["income_bins"], list(model_components["income_map"].keys()))
        df["balance_band"] = safe_bin(df["balance"], model_components["balance_bins"], list(model_components["balance_map"].keys()))
        df["outstanding_band"] = safe_bin(df["outstanding_loans"], model_components["outstanding_bins"], list(model_components["outstanding_map"].keys()))

        # Encoded features
        df["Income Band Encoded"] = df["income_band"].map(model_components["income_map"])
        df["Balance Band Encoded"] = df["balance_band"].map(model_components["balance_map"])
        df["Outstanding Loans Band Encoded"] = df["outstanding_band"].map(model_components["outstanding_map"])

        # Align feature names expected by the model
        df["Credit Score"] = df["credit_score"]
        df["Customer Tenure"] = df["customer_tenure"]
        df["NumOfProducts"] = df["num_products"]
        df["Credit History Length"] = df["credit_history_length"]
        df["NumComplaints"] = df["num_complaints"]

        final_df = df[FEATURE_ORDER]
        scaled = model_components["scaler"].transform(final_df)

        model = model_components["model"]
        df["Churn Prediction"] = model.predict(scaled)
        df["Probability"] = model.predict_proba(scaled)[:, 1]

        # Save raw CSV (with predictions) to Mongo as text
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_string = csv_buf.getvalue()

        # Keep the original filename for download name
        original_name = file.filename or f"upload_{file_id}.csv"

        if results_collection:
            # Replace previous results for this user_id
            results_collection.delete_many({"user_id": user_id})
            results_collection.insert_one({
                "user_id": user_id,
                "filename": original_name,
                "created_at": datetime.utcnow(),
                "raw_csv": csv_string,
            })

        # Also write a copy to /tmp (optional)
        result_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_id}.csv"
        result_path = os.path.join(RESULT_FOLDER, result_file)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(csv_string)

        return jsonify({
            "status": "success",
            "predictions_file": result_file,
            "churn_count": int(df["Churn Prediction"].sum()),
            "total_customers": int(len(df)),
            "churn_rate": float(df["Churn Prediction"].mean()),
        }), 200

    except Exception as e:
        print("----- ERROR in /predict -----")
        print(f"{type(e).__name__}: {e}")
        print("-----------------------------")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.get("/results/latest")
def latest_results():
    try:
        user_id = request.args.get("user_id", "anonymous")
        if not results_collection:
            return jsonify({"error": "DB not configured"}), 500

        result = results_collection.find_one({"user_id": user_id})
        if not result:
            return jsonify({"error": "No results found"}), 404

        df = pd.read_csv(io.StringIO(result["raw_csv"]))

        # Clean & coerce where needed
        df = df.dropna(subset=["num_complaints"])
        df["num_complaints"] = pd.to_numeric(df["num_complaints"], errors="coerce")
        df = df.dropna(subset=["num_complaints"])

        # Build summaries for charts
        complaints_chart = (
            df.groupby(["num_complaints", "Churn Prediction"])
              .size().unstack(fill_value=0).reset_index()
              .rename(columns={0: "retained", 1: "churned", "num_complaints": "complaints"})
              .sort_values("complaints")
              .to_dict(orient="records")
        )

        churn_counts = df["Churn Prediction"].value_counts().to_dict()
        pie_data = [
            {"name": "Retained", "value": churn_counts.get(0, 0)},
            {"name": "Churned",  "value": churn_counts.get(1, 0)},
        ]

        tenure_summary = (
            df.groupby(["customer_tenure", "Churn Prediction"])
              .size().unstack(fill_value=0).reset_index()
              .rename(columns={0: "retained", 1: "churned", "customer_tenure": "tenure"})
              .sort_values("tenure")
              .to_dict(orient="records")
        )

        score_summary = (
            df.groupby(["credit_score", "Churn Prediction"])
              .size().unstack(fill_value=0).reset_index()
              .rename(columns={0: "retained", 1: "churned"})
              .sort_values("credit_score")
              .to_dict(orient="records")
        )

        band_order = list(model_components["income_map"].keys())
        income_summary = (
            df.groupby(["income_band", "Churn Prediction"])
              .size().unstack(fill_value=0).reset_index()
        )
        income_summary["income_band"] = pd.Categorical(
            income_summary["income_band"], categories=band_order, ordered=True
        )
        income_summary = income_summary.sort_values("income_band").rename(columns={
            0: "retained",
            1: "churned",
            "income_band": "band",
        }).to_dict(orient="records")

        avg_credit_score = (
            df.groupby("Churn Prediction")["credit_score"]
              .mean()
              .reset_index()
              .rename(columns={"Churn Prediction": "label", "credit_score": "avg_score"})
              .replace({0: "Retained", 1: "Churned"})
              .to_dict(orient="records")
        )

        df["prob_bin"] = pd.cut(
            df["Probability"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            include_lowest=True,
        )
        prob_bins_df = df["prob_bin"].value_counts().sort_index().reset_index()
        prob_bins_df.columns = ["bin", "count"]
        prob_bins = prob_bins_df.to_dict(orient="records")

        # Balance area chart (bin by 5k)
        bin_width = 5000
        min_balance = df["balance"].min()
        max_balance = df["balance"].max()
        balance_bins = list(range(int(min_balance), int(max_balance + bin_width), bin_width))
        if len(balance_bins) < 2:
            balance_bins = [int(min_balance), int(max_balance + bin_width)]

        df["balance_range"] = pd.cut(
            df["balance"],
            bins=balance_bins,
            include_lowest=True,
            labels=[int((balance_bins[i] + balance_bins[i + 1]) / 2) for i in range(len(balance_bins) - 1)]
        )

        balance_area = (
            df.groupby(["balance_range", "Churn Prediction"])
              .size().unstack(fill_value=0)
              .reset_index()
              .rename(columns={0: "retained", 1: "churned", "balance_range": "balance"})
              .dropna()
        )
        try:
            balance_area["balance"] = balance_area["balance"].astype(int)
        except Exception:
            pass
        balance_area_chart = balance_area.sort_values("balance").to_dict(orient="records")

        return jsonify({
            "csv_url": result["filename"],
            "total_rows": int(len(df)),
            "pieData": pie_data,
            "tenureChart": tenure_summary,
            "creditScoreChart": score_summary,
            "incomeBandChart": income_summary,
            "avgCreditScoreByChurn": avg_credit_score,
            "probabilityBins": prob_bins,
            "creditScoreDistribution": (
                df.groupby(["credit_score", "Churn Prediction"])
                  .size().unstack(fill_value=0).reset_index()
                  .rename(columns={0: "retained", 1: "churned"})
                  .to_dict(orient="records")
            ),
            "balanceAreaChart": balance_area_chart,
            "complaintsLineChart": complaints_chart,
        }), 200

    except Exception as e:
        print("❌ ERROR in /results/latest:", str(e))
        return jsonify({"error": str(e)}), 500

@app.get("/results/download")
def download_csv():
    if not results_collection:
        return jsonify({"error": "DB not configured"}), 500

    user_id = request.args.get("user_id", "anonymous")
    result = results_collection.find_one({"user_id": user_id})
    if not result:
        return jsonify({"error": "No results found"}), 404

    return Response(
        result["raw_csv"],
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={result['filename']}"},
    )

# --------------- Gunicorn entrypoint uses: app ---------------
if __name__ == "__main__":
    # Local dev only
    app.run(host="0.0.0.0", port=5000, debug=False)
