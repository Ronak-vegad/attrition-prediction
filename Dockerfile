# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# - PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# - STREAMLIT_SERVER_PORT: Sets the default port for Streamlit
# - STREAMLIT_SERVER_ADDRESS: Binds Streamlit to all network interfaces
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory in the container
WORKDIR /workspace

# Install minimal system dependencies required for building C extensions (e.g. for shap/xgboost if wheels aren't used)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and exported models into the container
COPY app/ ./app/
COPY models/ ./models/

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit dashboard
CMD ["streamlit", "run", "app/streamlit_app.py"]
