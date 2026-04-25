# 🏗️ Real-Time Mechanical Property Prediction for Steel Rebars

Predict **Ultimate Tensile Strength (UTS)** and **Yield Strength (YS)** of steel rebars in real time from manufacturing parameters using machine learning.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0-006600?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white)

---

## 📋 Problem Statement

In steel rebar manufacturing, **mechanical properties** (UTS and Yield Strength) are critical quality parameters that determine whether a product meets **IS 1786 standards**. Currently, these properties are measured through destructive tensile testing — which is slow, expensive, and happens after production.

This project builds **ML models that predict mechanical properties in real time** using data available during the manufacturing process:
- **Chemical composition** (10 parameters)
- **Temperature profiles** (6 measurements)
- **Process parameters** (speed, rolling forces, etc.)

This enables **proactive quality control** — identifying potential failures *before* they happen.

---

## 🏛️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Raw CSV Data   │────▶│  Preprocessing   │────▶│  Split by Ø      │
│   5,156 records  │     │  3-stage impute  │     │  10/12/16 mm     │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                    ┌──────────────────────────────────────┤
                    ▼                  ▼                    ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │  Train Ø10   │  │  Train Ø12   │  │  Train Ø16   │
            │  RF vs XGB   │  │  RF vs XGB   │  │  RF vs XGB   │
            │  GridSearchCV│  │  GridSearchCV│  │  GridSearchCV│
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                  │
                   ▼                 ▼                  ▼
            ┌──────────────────────────────────────────────────┐
            │           Saved Model Artifacts (.joblib)        │
            └──────────────────────┬───────────────────────────┘
                                   │
                    ┌──────────────┤──────────────┐
                    ▼                              ▼
            ┌──────────────┐              ┌──────────────┐
            │  FastAPI      │              │  Streamlit   │
            │  REST API     │◀────────────▶│  Dashboard   │
            │  /predict     │              │  4-tab UI    │
            └──────────────┘              └──────────────┘
```

---

## 📊 Model Performance

| Diameter | Best Model | R² (Combined) | RMSE | MAPE |
|----------|-----------|---------------|------|------|
| 10mm | Random Forest | 0.3974 | 15.47 | 1.88% |
| 12mm | Random Forest | 0.3865 | 13.85 | 1.66% |
| 16mm | Random Forest | 0.4972 | 12.54 | 1.49% |

*Models compared: Random Forest vs XGBoost with GridSearchCV (5-fold CV)*

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train.py
# or train a specific diameter:
python train.py --diameter 12 --verbose
```

### 3. Launch the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### 4. Or use the REST API
```bash
# Start the API server
uvicorn app.api:app --port 8000

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "diameter": 12, "grade": "GR1",
    "CHEM1": 0.225, "CHEM2": 1.297, "CHEM3": 0.017,
    "CHEM4": 0.024, "CHEM5": 0.230, "CHEM6": 0.0026,
    "CHEM7": 0.0005, "CHEM8": 0.028, "CHEM9": 0.006,
    "CHEM10": 70.5, "TEMP1": 1000, "TEMP2": 965.0,
    "TEMP3": 70, "TEMP4": 158, "TEMP5": 581.0,
    "TEMP6": 549, "SPEED": 21, "PROCESS1": 2521,
    "PROCESS2": 2666, "PROCESS3": 1.32
  }'
```

API docs available at: http://localhost:8000/docs

---

## 📁 Project Structure

```
├── src/                        # Core Python package
│   ├── config.py               # All constants & hyperparameters
│   ├── data/
│   │   ├── loader.py           # CSV loading & validation
│   │   └── preprocessor.py     # 3-stage imputation pipeline
│   ├── features/
│   │   └── encoder.py          # OneHot encoding & feature assembly
│   └── models/
│       ├── trainer.py          # Multi-model training & comparison
│       ├── evaluator.py        # Metrics, plots, residual analysis
│       └── predictor.py        # Inference-time prediction API
├── app/
│   ├── api.py                  # FastAPI REST backend
│   ├── streamlit_app.py        # Streamlit dashboard frontend
│   └── assets/style.css        # Custom UI styling
├── tests/                      # pytest test suite (49 tests)
├── train.py                    # CLI training entry point
├── data/raw/                   # Raw source data
├── models/                     # Trained model artifacts (generated)
├── outputs/                    # Evaluation plots (generated)
├── notebooks/                  # Original exploration notebooks
└── requirements.txt
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

```
49 passed in 3.86s
```

---

## 🔧 Key Technical Decisions

- **Diameter-stratified models**: Different rebar diameters have fundamentally different rolling physics — separate models capture these differences.
- **3-stage domain-aware imputation**: Chemical features by Grade, temperatures by Diameter, quality targets by (Grade × Diameter) — preserving metallurgical relationships.
- **Multi-output regression**: Random Forest natively predicts both UTS and YS simultaneously, capturing their correlation structure.
- **Quality gate**: UTS/YS ratio ≥ 1.1 per IS 1786 standard for rebar ductility compliance.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Training | scikit-learn, XGBoost |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| API Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Testing | pytest |
| Serialization | joblib |

---

## 📄 License

This project is for educational and demonstration purposes.
