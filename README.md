# Churn Prediction Project

End-to-end ML pipeline that predicts telecom customer churn using XGBoost, MongoDB Atlas, and a live Streamlit demo.

## Live Demo

https://churnpredictionprojectgit-2pb9ahyzn6uabmfvgpzqgc.streamlit.app/

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
- MongoDB feature store integration
- Explainability with SHAP during training
