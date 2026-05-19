# Smart Marketing Optimizer 🚀

Hệ thống dự báo doanh thu và tối ưu hóa chiến dịch Marketing sử dụng Ensemble Learning (LightGBM + XGBoost + Prophet), kết nối trực tiếp **Supabase PostgreSQL** và triển khai qua **Docker**.

---

## 📌 1. Giới thiệu (Introduction)

Dự án xây dựng một công cụ hỗ trợ ra quyết định (Decision Support Tool) dành cho doanh nghiệp, kết hợp các mô hình học máy tiên tiến qua phương pháp **Weighted Ensemble** được tối ưu bằng Nelder-Mead trên tập Validation nhằm:

- Dự báo chính xác xu hướng **Doanh thu (Revenue)** và **Giá vốn hàng bán (COGS)** theo ngày.
- Tối ưu hóa ngân sách và đánh giá hiệu quả chiến dịch Marketing.
- Trực quan hóa dữ liệu thời gian thực qua Dashboard **Streamlit** kết nối **Cloud Database (Supabase PostgreSQL)**.

---

## 📂 2. Cấu trúc thư mục (Project Structure)

```text
Smart_Marketing_Optimizer/
│
├── .streamlit/
│   └── secrets.toml            # [LOCAL ONLY] Cấu hình secrets (đã thêm vào .gitignore)
│
├── notebooks/
│   ├── Forecast_EDA_insights.ipynb         # EDA & phân tích đặc trưng
│   └── Smart_Marketing_Model_fixe.ipynb    # Nghiên cứu & kiểm chứng mô hình
│
├── src/
│   ├── app.py                  # Streamlit Dashboard
│   ├── bundle.py               # Huấn luyện & lưu model bundle
│   ├── etl.py                  # ETL pipeline
│   ├── forecast_pipeline.py    # Feature engineering & inference pipeline
│   └── model_bundle.pkl        # Trained models + metadata (đã thêm vào .gitignore)
│
├── .env                        # [LOCAL ONLY] Biến môi trường (đã thêm vào .gitignore)
├── .env.example                # File cấu hình mẫu cho deploy
├── .gitignore
├── credentials.json            # [LOCAL ONLY] GCP credentials nếu dùng
├── docker-compose.yml          # Điều phối container
├── Dockerfile                  # Build image ứng dụng
├── migrate_db.py               # Script migrate dữ liệu lên Cloud
├── requirements.txt            # Danh sách thư viện & phiên bản
└── README.md
```

---

## 🧠 3. Kiến trúc mô hình (Model Architecture)

Hệ thống sử dụng **Weighted Ensemble** gồm 4 base model, trọng số được tối ưu tự động trên tập Validation:

| Model | Mục tiêu | Ghi chú |
|---|---|---|
| LightGBM (MAE) | Revenue & COGS | 5 seeds, log-transform target |
| XGBoost (MAE) | Revenue & COGS | 5 seeds, log-transform target |
| Prophet + LightGBM Residual | Revenue & COGS | Hybrid: Prophet bắt trend/mùa vụ, LGB bắt phần dư |
| LightGBM (Tweedie) | Revenue & COGS | Robust với phân phối lệch phải |

Ngoài ra có thêm **LightGBM Q90** (quantile 0.9) được blend vào dự báo cuối để tăng độ bảo thủ.

### Feature Engineering nổi bật

- **Calendar features:** day, month, quarter, dayofweek, weekofyear, is_weekend, is_holiday (VN)
- **Tết Nguyên Đán:** tet_distance, tet_proximity, tet_phase (pre21/pre7/week/post7/far_after)
- **Flash sale events:** 9.9, 10.10, 11.11, 12.12, Black Friday, event windows
- **Lag features:** Revenue/COGS lag 549–1460 ngày (safe lags để tránh data leakage)
- **Rolling features:** mean/std/max trên cửa sổ 7–180 ngày
- **Exogenous:** web traffic, promotions, inventory, order metrics
- **Trend polynomial:** bậc 2 fit trên log(revenue/cogs)

---

## 📊 4. Mô tả dữ liệu (Data Description)

Hệ thống kết nối trực tiếp **Supabase PostgreSQL**, bao gồm các bảng:

| Bảng | Mô tả |
|---|---|
| `sales` | Doanh thu (`revenue`) và giá vốn (`cogs`) theo ngày |
| `sample_submission` | Danh sách ngày cần dự báo |
| `orders` | Đơn hàng theo ngày |
| `order_items` | Chi tiết sản phẩm trong đơn hàng |
| `promotions` | Các chương trình khuyến mãi (start_date, end_date, discount_value) |
| `web_traffic` | Sessions, page views, bounce rate, avg session duration theo ngày |
| `inventory` | Tồn kho, fill rate, stockout flag, days of supply |

---

## 🛡️ 5. Độ tin cậy mô hình (Model Reliability)

Trọng số ensemble được tối ưu bằng **Nelder-Mead** trên tập Validation (từ 2022-01-01 đến cuối train). Các chỉ số đánh giá:

**Hệ số xác định R²:**

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**Sai số phần trăm tuyệt đối trung bình (MAPE):**

$$\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

---

## ⚙️ 6. Hướng dẫn cài đặt & chạy (Setup & Run)

### Yêu cầu

- Python 3.10+
- Docker & Docker Compose (nếu chạy container)
- PostgreSQL (local hoặc Supabase)

### Chạy local

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/Smart_Marketing_Optimizer.git
cd Smart_Marketing_Optimizer

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Cấu hình database
cp .env.example .env
# Chỉnh DATABASE_URL trong .env

# 4. Huấn luyện mô hình (chỉ cần chạy 1 lần, mất ~20–40 phút)
python src/bundle.py

# 5. Chạy dashboard
streamlit run src/app.py
```

### Chạy bằng Docker

```bash
# Build & start
docker-compose up --build

# App sẽ chạy tại http://localhost:8501
```

### Cấu hình `.env`

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

### Cấu hình Streamlit Secrets (`.streamlit/secrets.toml`)

```toml
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
```

> **Ưu tiên kết nối DB:** UI input → `DATABASE_URL` env var → Streamlit secrets → localhost fallback

---

## 🐳 7. Docker

`Dockerfile` build image Python, cài toàn bộ dependencies từ `requirements.txt` và expose port 8501.

`docker-compose.yml` điều phối:
- Service `app`: Streamlit dashboard
- Mount `model_bundle.pkl` vào container (bundle được train sẵn, không train lại trong Docker)

---

## 🔐 8. Bảo mật (Security)

Các file sau đã được thêm vào `.gitignore`, **không được commit lên GitHub**:

```
.env
.streamlit/secrets.toml
src/model_bundle.pkl
credentials.json
token.json
```

---

## 📦 9. Dependencies chính

| Thư viện | Mục đích |
|---|---|
| `lightgbm` | Gradient boosting model |
| `xgboost` | Gradient boosting model |
| `prophet` | Time series trend & seasonality |
| `streamlit` | Dashboard UI |
| `sqlalchemy` | Kết nối PostgreSQL |
| `joblib` | Serialize model bundle |
| `lunardate` | Tính ngày Tết Nguyên Đán |
| `holidays` | Calendar ngày lễ Việt Nam |
| `scipy` | Tối ưu trọng số ensemble |
| `pandas` / `numpy` | Xử lý dữ liệu |

--
[WEB DEMO](assets/Smart_Marketing_Optimizer.mp4)
