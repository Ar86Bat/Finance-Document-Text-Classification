# ----------------------------
# 1) Base image with Python
# ----------------------------
FROM python:3.11-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# 2) Set work directory inside container
# ----------------------------
WORKDIR /app

# ----------------------------
# 3) Copy dependency list first (better caching)
# ----------------------------
COPY requirements.txt .

# ----------------------------
# 4) Install dependencies
# ----------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 5) Copy project files into container
# ----------------------------
COPY . .

# ----------------------------
# 6) Expose FastAPI port
# ----------------------------
EXPOSE 8000

# ----------------------------
# 7) Run FastAPI with uvicorn
# ----------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
