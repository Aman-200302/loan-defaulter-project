# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching — rebuilds are faster)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs
COPY api/ ./api/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]