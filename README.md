# Tourism Package Prediction — MLOps Pipeline

Predicts whether a "Visit with Us" customer will buy the **Wellness Tourism Package** before they are contacted.

| Step | File |
|---|---|
| Data registration | `tourism_project/data_register.py` |
| Data preparation  | `tourism_project/prep.py` |
| Training + MLflow | `tourism_project/train.py` |
| Trained model (committed by CI) | `tourism_project/model/best_model.joblib` |
| Streamlit app | `tourism_project/deployment/app.py` |
| CI/CD pipeline | `.github/workflows/pipeline.yml` |

Every push to `main` runs: register → prep (artifact) → train (artifact in, model committed back to `main`).
Streamlit Community Cloud redeploys the app on each new model commit.

**Live app:** https://<your-app>.streamlit.app

Run locally:
```bash
pip install -r tourism_project/requirements.txt
python tourism_project/data_register.py && python tourism_project/prep.py && python tourism_project/train.py
streamlit run tourism_project/deployment/app.py
```
