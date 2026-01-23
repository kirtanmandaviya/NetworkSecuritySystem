FROM python:3.10-slim

WORKDIR /app

# Copy all files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p final_model prediction_output templates

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run FastAPI with uvicorn on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]