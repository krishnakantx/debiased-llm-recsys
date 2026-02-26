FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling the large CUDA bundle)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Default: run the pipeline with a config passed at runtime
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--help"]
