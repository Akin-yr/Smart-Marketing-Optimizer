# Smart Marketing Optimizer 🚀

A revenue forecasting and marketing campaign optimization system using Ensemble Learning (LightGBM + XGBoost + Prophet), connected directly to **Supabase PostgreSQL** and deployed via **Docker**.

---

## 📌 1. Introduction

This project builds a Decision Support Tool for businesses, combining advanced machine learning models through a **Weighted Ensemble** method automatically optimized via Nelder-Mead on the Validation set, designed to:

- Accurately forecast **Revenue** and **Cost of Goods Sold (COGS)** trends on a daily basis.
- Optimize budgets and evaluate marketing campaign effectiveness.
- Visualize real-time data through a **Streamlit** Dashboard connected to a **Cloud Database (Supabase PostgreSQL)**.

---

## 📂 2. Project Structure

```text
Smart_Marketing_Optimizer/
│
├── .streamlit/
│   └── secrets.toml            # [LOCAL ONLY] Secrets config (added to .gitignore)
│
├── notebooks/
│   ├── Forecast_EDA_insights.ipynb         # EDA & feature analysis
│   └── Smart_Marketing_Model_fixe.ipynb    # Model research & validation
│
├── src/
│   ├── app.py                  # Streamlit Dashboard
│   ├── bundle.py               # Model training & bundle saving
│   ├── etl.py                  # ETL pipeline
│   ├── forecast_pipeline.py    # Feature engineering & inference pipeline
│   └── model_bundle.pkl        # Trained models + metadata (added to .gitignore)
│
├── .env                        # [LOCAL ONLY] Environment variables (added to .gitignore)
├── .env.example                # Sample config file for deployment
├── .gitignore
├── credentials.json            # [LOCAL ONLY] GCP credentials (if used)
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Application image build
├── migrate_db.py               # Data migration script to Cloud
├── requirements.txt            # Library list & versions
└── README.md
```

---

## 🧠 3. Model Architecture

The system uses a **Weighted Ensemble** of 4 base models, with weights automatically optimized on the Validation set:

| Model | Target | Notes |
|---|---|---|
| LightGBM (MAE) | Revenue & COGS | 5 seeds, log-transform target |
| XGBoost (MAE) | Revenue & COGS | 5 seeds, log-transform target |
| Prophet + LightGBM Residual | Revenue & COGS | Hybrid: Prophet captures trend/seasonality, LGB captures residuals |
| LightGBM (Tweedie) | Revenue & COGS | Robust against right-skewed distributions |

Additionally, **LightGBM Q90** (quantile 0.9) is blended into the final forecast to increase conservatism.

### Key Feature Engineering

- **Calendar features:** day, month, quarter, dayofweek, weekofyear, is_weekend, is_holiday (VN)
- **Lunar New Year (Tết):** tet_distance, tet_proximity, tet_phase (pre21/pre7/week/post7/far_after)
- **Flash sale events:** 9.9, 10.10, 11.11, 12.12, Black Friday, event windows
- **Lag features:** Revenue/COGS lag 549–1460 days (safe lags to avoid data leakage)
- **Rolling features:** mean/std/max over 7–180 day windows
- **Exogenous:** web traffic, promotions, inventory, order metrics
- **Trend polynomial:** degree-2 fit on log(revenue/cogs)

---

## 📊 4. Data Description

The system connects directly to **Supabase PostgreSQL**, including the following tables:

| Table | Description |
|---|---|
| `sales` | Daily revenue (`revenue`) and cost of goods sold (`cogs`) |
| `sample_submission` | List of dates to forecast |
| `orders` | Daily orders |
| `order_items` | Product details within orders |
| `promotions` | Promotional campaigns (start_date, end_date, discount_value) |
| `web_traffic` | Sessions, page views, bounce rate, avg session duration per day |
| `inventory` | Stock levels, fill rate, stockout flag, days of supply |

---

## 🛡️ 5. Model Reliability

Ensemble weights are optimized using **Nelder-Mead** on the Validation set (from 2022-01-01 to end of training). Evaluation metrics:

**Coefficient of Determination R²:**

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**Mean Absolute Percentage Error (MAPE):**

$$\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

---

## ⚙️ 6. Setup & Run

### Requirements

- Python 3.10+
- Docker & Docker Compose (for container deployment)
- PostgreSQL (local or Supabase)

### Run Locally

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/Smart_Marketing_Optimizer.git
cd Smart_Marketing_Optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure database
cp .env.example .env
# Edit DATABASE_URL in .env

# 4. Train the model (only needed once, takes ~20–40 minutes)
python src/bundle.py

# 5. Run dashboard
streamlit run src/app.py
```

### Run with Docker

```bash
# Build & start
docker-compose up --build

# App will be available at http://localhost:8501
```

### `.env` Configuration

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

### Streamlit Secrets Configuration (`.streamlit/secrets.toml`)

```toml
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

> **DB connection priority:** UI input → `DATABASE_URL` env var → Streamlit secrets → localhost fallback

---

## 🐳 7. Docker

`Dockerfile` builds a Python image, installs all dependencies from `requirements.txt`, and exposes port 8501.

`docker-compose.yml` orchestrates:
- `app` service: Streamlit dashboard
- Mounts `model_bundle.pkl` into the container (pre-trained bundle — no retraining inside Docker)

---

## 🔐 8. Security

The following files are added to `.gitignore` and **must not be committed to GitHub**:

```
.env
.streamlit/secrets.toml
src/model_bundle.pkl
credentials.json
token.json
```

---

## 📦 9. Main Dependencies

| Library | Purpose |
|---|---|
| `lightgbm` | Gradient boosting model |
| `xgboost` | Gradient boosting model |
| `prophet` | Time series trend & seasonality |
| `streamlit` | Dashboard UI |
| `sqlalchemy` | PostgreSQL connection |
| `joblib` | Model bundle serialization |
| `lunardate` | Vietnamese Lunar New Year calculation |
| `holidays` | Vietnamese public holiday calendar |
| `scipy` | Ensemble weight optimization |
| `pandas` / `numpy` | Data processing |

[WEB DEMO](https://streamable.com/pufujp)
