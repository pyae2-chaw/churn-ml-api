from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import pandas as pd
import os
import joblib
import uuid
from datetime import datetime
import warnings
import re
from sklearn.exceptions import InconsistentVersionWarning
from dotenv import load_dotenv
from pymongo import MongoClient
import io

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)
db = client["ml_app"]
results_collection = db["results"]

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def load_models():
    components = {
        'model': joblib.load(os.path.join(MODEL_FOLDER, 'model.pkl')),
        'scaler': joblib.load(os.path.join(MODEL_FOLDER, 'scaler.pkl')),
        'income_bins': joblib.load(os.path.join(MODEL_FOLDER, 'income_bins.pkl')),
        'balance_bins': joblib.load(os.path.join(MODEL_FOLDER, 'balance_bins.pkl')),
        'outstanding_bins': joblib.load(os.path.join(MODEL_FOLDER, 'outstanding_bins.pkl')),
        'income_map': joblib.load(os.path.join(MODEL_FOLDER, 'income_map.pkl')),
        'balance_map': joblib.load(os.path.join(MODEL_FOLDER, 'balance_map.pkl')),
        'outstanding_map': joblib.load(os.path.join(MODEL_FOLDER, 'outstanding_map.pkl')),
    }

    feature_order = [
        "Credit Score", "Customer Tenure", "Balance Band Encoded",
        "NumOfProducts", "Outstanding Loans Band Encoded",
        "Income Band Encoded", "Credit History Length", "NumComplaints"
    ]

    return components, feature_order

model_components, FEATURE_ORDER = load_models()

def normalize_column(col):
    return re.sub(r'[^a-z0-9]', '', col.lower())

def map_columns(df, expected_features):
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
    'credit_score': ['credit score', 'creditscore', 'Credit_Score', 'Credit Score'],
    'balance': ['balance', 'Balance'],
    'num_products': ['NumOfProducts', 'num_products', 'number of products'],
    'income': ['income', 'annual income', 'Income'],
    'customer_tenure': ['Customer Tenure', 'tenure', 'Tenure', 'Years with Bank'],
    'outstanding_loans': ['Outstanding Loans', 'outstanding loans', 'loan_balance', 'outstanding_debt'],
    'credit_history_length': ['Credit History Length', 'history length'],
    'num_complaints': ['NumComplaints', 'complaints']
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        file_id = uuid.uuid4().hex
        upload_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
        file.save(upload_path)
        df = pd.read_csv(upload_path)
        df = map_columns(df, expected_features)

        required_cols = [
            'income', 'balance', 'outstanding_loans',
            'credit_score', 'customer_tenure',
            'num_products', 'credit_history_length', 'num_complaints'
        ]
        df = df.dropna(subset=required_cols)
        if len(df) < 50:
            return jsonify({"error": "Insufficient rows after cleaning. Minimum 50 required."}), 400

        def safe_bin(series, bin_edges, labels):
            min_val, max_val = bin_edges[0], bin_edges[-1]
            series_clipped = series.clip(lower=min_val, upper=max_val)
            return pd.cut(series_clipped, bins=bin_edges, labels=labels, include_lowest=True)

        df["income_band"] = safe_bin(df["income"], model_components['income_bins'], list(model_components['income_map'].keys()))
        df["balance_band"] = safe_bin(df["balance"], model_components['balance_bins'], list(model_components['balance_map'].keys()))
        df["outstanding_band"] = safe_bin(df["outstanding_loans"], model_components['outstanding_bins'], list(model_components['outstanding_map'].keys()))

        df["Income Band Encoded"] = df["income_band"].map(model_components['income_map'])
        df["Balance Band Encoded"] = df["balance_band"].map(model_components['balance_map'])
        df["Outstanding Loans Band Encoded"] = df["outstanding_band"].map(model_components['outstanding_map'])

        df["Credit Score"] = df["credit_score"]
        df["Customer Tenure"] = df["customer_tenure"]
        df["NumOfProducts"] = df["num_products"]
        df["Credit History Length"] = df["credit_history_length"]
        df["NumComplaints"] = df["num_complaints"]

        final_df = df[FEATURE_ORDER]
        scaled_data = model_components['scaler'].transform(final_df)

        model = model_components['model']
        df['Churn Prediction'] = model.predict(scaled_data)
        df['Probability'] = model.predict_proba(scaled_data)[:, 1]

        result_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_id}.csv"
        result_path = os.path.join(RESULT_FOLDER, result_file)

        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # ✅ Get user_id from formData
        user_id = request.form.get("user_id", "anonymous")
        results_collection.delete_many({"user_id": user_id})
        results_collection.insert_one({
            "user_id": user_id,
            "filename": file.filename,
            "created_at": datetime.utcnow(),
            "raw_csv": csv_string
        })

        return jsonify({
            'status': 'success',
            'predictions_file': result_file,
            'churn_count': int(df['Churn Prediction'].sum()),
            'total_customers': len(df),
            'churn_rate': float(df['Churn Prediction'].mean())
        })

    except Exception as e:
        print("----- ERROR in /predict -----")
        print(f"{type(e).__name__}: {e}")
        print("-----------------------------")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/results/latest', methods=['GET'])
def latest_results():
    try:
        user_id = request.args.get("user_id", "anonymous")
        result = results_collection.find_one({"user_id": user_id})

        if not result:
            return jsonify({"error": "No results found"}), 404

        df = pd.read_csv(io.StringIO(result['raw_csv']))

        df = df.dropna(subset=['num_complaints'])
        df['num_complaints'] = pd.to_numeric(df['num_complaints'], errors='coerce')
        df = df.dropna(subset=['num_complaints'])

        complaints_chart = (
            df.groupby(['num_complaints', 'Churn Prediction'])
              .size().unstack(fill_value=0).reset_index()
              .rename(columns={0: 'retained', 1: 'churned', 'num_complaints': 'complaints'})
              .sort_values('complaints')
              .to_dict(orient='records')
        )

        churn_counts = df['Churn Prediction'].value_counts().to_dict()
        pie_data = [
            {"name": "Retained", "value": churn_counts.get(0, 0)},
            {"name": "Churned", "value": churn_counts.get(1, 0)}
        ]

        tenure_summary = (
            df.groupby(['customer_tenure', 'Churn Prediction'])
              .size().unstack(fill_value=0).reset_index()
              .rename(columns={0: "retained", 1: "churned", 'customer_tenure': 'tenure'})
              .sort_values("tenure")
              .to_dict(orient='records')
        )

        score_summary = (
            df.groupby(['credit_score', 'Churn Prediction'])
              .size().unstack(fill_value=0).reset_index()
              .rename(columns={0: "retained", 1: "churned", 'credit_score': 'credit_score'})
              .sort_values("credit_score")
              .to_dict(orient='records')
        )

        band_order = list(model_components['income_map'].keys())
        income_summary = (
            df.groupby(['income_band', 'Churn Prediction'])
              .size().unstack(fill_value=0).reset_index()
        )
        income_summary['income_band'] = pd.Categorical(
            income_summary['income_band'], categories=band_order, ordered=True
        )
        income_summary = income_summary.sort_values("income_band")
        income_summary = income_summary.rename(columns={
            0: "retained",
            1: "churned",
            'income_band': 'band'
        }).to_dict(orient='records')

        avg_credit_score = (
            df.groupby('Churn Prediction')['credit_score']
              .mean()
              .reset_index()
              .rename(columns={'Churn Prediction': 'label', 'credit_score': 'avg_score'})
              .replace({0: 'Retained', 1: 'Churned'})
              .to_dict(orient='records')
        )

        df['prob_bin'] = pd.cut(
            df['Probability'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
            include_lowest=True
        )
        prob_bins_df = df['prob_bin'].value_counts().sort_index().reset_index()
        prob_bins_df.columns = ['bin', 'count']
        prob_bins = prob_bins_df.to_dict(orient='records')

        credit_score_distribution = (
            df.groupby(['credit_score', 'Churn Prediction'])
              .size().unstack(fill_value=0).reset_index()
              .rename(columns={0: 'retained', 1: 'churned'})
              .to_dict(orient='records')
        )

        bin_width = 5000
        min_balance = df['balance'].min()
        max_balance = df['balance'].max()
        balance_bins = list(range(int(min_balance), int(max_balance + bin_width), bin_width))

        df['balance_range'] = pd.cut(
            df['balance'],
            bins=balance_bins,
            include_lowest=True,
            labels=[int((balance_bins[i] + balance_bins[i + 1]) / 2) for i in range(len(balance_bins) - 1)]
        )

        balance_area = (
            df.groupby(['balance_range', 'Churn Prediction'])
              .size().unstack(fill_value=0)
              .reset_index()
              .rename(columns={0: 'retained', 1: 'churned', 'balance_range': 'balance'})
              .dropna()
        )
        balance_area['balance'] = balance_area['balance'].astype(int)
        balance_area_chart = balance_area.sort_values("balance").to_dict(orient='records')

        return jsonify({
            "csv_url": result["filename"],
            "total_rows": len(df),
            "pieData": pie_data,
            "tenureChart": tenure_summary,
            "creditScoreChart": score_summary,
            "incomeBandChart": income_summary,
            "avgCreditScoreByChurn": avg_credit_score,
            "probabilityBins": prob_bins,
            "creditScoreDistribution": credit_score_distribution,
            "balanceAreaChart": balance_area_chart,
            "complaintsLineChart": complaints_chart
        })

    except Exception as e:
        print("❌ ERROR in /results/latest:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/results/download', methods=['GET'])
def download_csv():
    user_id = request.args.get("user_id", "anonymous")
    result = results_collection.find_one({"user_id": user_id})

    if not result:
        return jsonify({"error": "No results found"}), 404

    return Response(
        result["raw_csv"],
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={result['filename']}"}
    )

if __name__ == '__main__':
    app.run(port=5000, debug=False)
