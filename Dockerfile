FROM python:3.11-slim

WORKDIR /app

# Cài đặt công cụ hệ thống cần thiết cho thư viện C và Postgres
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy và cài đặt thư viện trước để tối ưu hóa cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

EXPOSE 8501

# Lệnh chạy Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]