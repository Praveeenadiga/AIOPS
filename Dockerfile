# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirement.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

# Copy project files
COPY . .

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["python", "src/models/train_first_model.py"]
