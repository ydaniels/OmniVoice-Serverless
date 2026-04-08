ARG CUDA_VERSION="12.8.1"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    OMNIVOICE_MODEL_ID=k2-fsa/OmniVoice \
    OMNIVOICE_AUDIO_TOKENIZER_ID=eustlb/higgs-audio-v2-tokenizer \
    OMNIVOICE_LOAD_ASR=0 \
    OMNIVOICE_DOWNLOAD_ASR_AT_BUILD=0 \
    OMNIVOICE_REPO_URL=https://github.com/k2-fsa/OmniVoice.git \
    OMNIVOICE_REPO_REF=main

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      git \
      ffmpeg \
      curl \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
ARG TORCH_VERSION="2.8.0+cu128"
RUN pip3 install --no-cache-dir --index-url ${PYTORCH_INDEX_URL} \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_VERSION}

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Install OmniVoice directly from GitHub (as requested).
RUN pip3 install --no-cache-dir "git+${OMNIVOICE_REPO_URL}@${OMNIVOICE_REPO_REF}"

COPY . /workspace
RUN chmod +x /workspace/start.sh

# Bake model artifacts into image layers.
RUN python3 /workspace/prefetch_model.py

ENTRYPOINT ["/workspace/start.sh"]
