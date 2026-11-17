FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.title="NeuroGen-XAI"
LABEL org.opencontainers.image.description="Explainable AI and De Novo Molecular Design Pipeline for CNS Drugs"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system libraries required by RDKit rendering and Streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libsm6 \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment from environment.yml
COPY environment.yml /app/environment.yml
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Activate environment for subsequent commands and runtime
SHELL ["conda", "run", "-n", "neurogen-xai", "/bin/bash", "-c"]
ENV PATH="/opt/conda/envs/neurogen-xai/bin:$PATH"

# Copy application and pipeline source code
COPY src/ /app/src/
COPY app/ /app/app/
COPY data/ /app/data/
COPY README.md /app/README.md

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
