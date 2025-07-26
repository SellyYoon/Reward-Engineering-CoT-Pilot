# Dockerfile
# Base image with CUDA runtime and Ubuntu
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH="/opt/conda/bin:${PATH}"
ENV CONDA_ALWAYS_YES="true"
ENV CONDA_PKGS_DIRS="/tmp/conda_pkgs"

ARG HF_TOKEN="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"
ENV HF_HOME="/mnt/e/CoT_Result/.cache/huggingface"

# Install Miniconda
RUN apt-get clean && \
    mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb http://mirror.kakao.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirror.kakao.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirror.kakao.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirror.kakao.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update --fix-missing -o Acquire::http::No-Cache=True && \
    apt-get install -y wget bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Explicitly accept Conda Terms of Service for 'main' and 'r' channels (just in case)
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Configure Conda for non-interactive use and explicitly set channels (no defaults)
RUN /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda config --add channels pytorch && \
    /opt/conda/bin/conda config --add channels nvidia && \
    /opt/conda/bin/conda config --add channels conda-forge && \
    /opt/conda/bin/conda config --remove channels defaults --file ~/.condarc || true && \
    /opt/conda/bin/conda clean --all -f -y

# Create Conda environment with only core dependencies (Python, pip)
COPY environment.yml /tmp/environment.yml
RUN /opt/conda/bin/conda env create -f /tmp/environment.yml -n pilot --yes && \
    /opt/conda/bin/conda clean --all -f -y

# Install all other Python packages via pip
SHELL ["/opt/conda/bin/conda", "run", "-n", "pilot", "/bin/bash", "-c"]
RUN pip install --no-cache-dir --upgrade \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.3.1+cu121" \
    "torchvision==0.18.1+cu121" \
    "torchaudio==2.3.1+cu121" \
    "protobuf" \
    "sentencepiece" \
    "transformers==4.42.0" \
    "accelerate==0.31.0" \
    "bitsandbytes==0.43.1" \
    "openai" \
    "anthropic" \
    "datasets==2.20.0" \
    "pandas" \
    "huggingface_hub==0.24.1" \
    "python-dotenv" \
    "bert-score==0.3.13" \
    "scikit-learn"

# Hugging Face CLI login
RUN pip install huggingface_hub
RUN if [ -n "${HF_TOKEN}" ]; then huggingface-cli login --token ${HF_TOKEN}; else echo "HF_TOKEN not set, skipping huggingface-cli login."; fi

# Set the default shell back for subsequent commands
SHELL ["/bin/bash", "-c"]

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app