# Churn Prediction Project

End-to-end ML pipeline that predicts telecom customer churn using XGBoost, MongoDB Atlas, and a live Streamlit demo.

## Live Demo

Deploy with Streamlit Community Cloud, then add your app URL here.

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.85 |
| PR-AUC | 0.66 |
| Accuracy | 79% |
| Churn Precision | 59% |
| Churn Recall | 69% |

## Tech Stack

- Python, pandas, scikit-learn, XGBoost, SHAP
- MongoDB Atlas (feature store)
- Streamlit (deployment demo)

## Project Structure

```
churn_prediction_project/
├── app.py                  # Streamlit deployment app
├── src/
│   ├── data_ingestion.py   # Load data + upload to MongoDB
│   ├── feature_engineering.py
│   └── model_training.py
├── models/churn_model.pkl
└── data/raw/
```

## Local Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the pipeline

```powershell
python src/data_ingestion.py
python src/model_training.py
```

### Run the demo app locally

```powershell
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (make sure `.env` is **not** committed).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub.
4. Click **New app**.
5. Select your repo, branch `main`, and main file `app.py`.
6. Click **Deploy**.

Streamlit will install `requirements.txt` and host the app at a URL like:

`https://your-app-name.streamlit.app`

## Environment Variables

MongoDB is only needed for the training/ingestion pipeline locally. The Streamlit demo uses the saved model file and does **not** require `MONGO_URI`.

Create a local `.env` file for ingestion/training:

```env
MONGO_URI="your_mongodb_connection_string"
```

## Portfolio Highlights

- Full ML pipeline from raw data to deployment
- Feature engineering with domain-specific variables
- Hyperparameter tuning and threshold optimization
- MongoDB feature store integration
- Explainability with SHAP during training
