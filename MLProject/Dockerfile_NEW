FROM python:3.12.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY . .

# Expose port for serving
EXPOSE 5000

# Default command for serving model
CMD ["mlflow", "models", "serve", "-m", "model", "--host", "0.0.0.0", "--port", "5000"]
