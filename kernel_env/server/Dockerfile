# GPU image for kernel_env. Needs an NVIDIA GPU + nvidia-container-toolkit on the host.
#
#   docker build -t kernel_env:latest -f server/Dockerfile .
#   docker run --gpus all -p 8000:8000 -e KERNEL_ENV_GPUS=0,1 kernel_env:latest
#
# The -devel base ships nvcc (needed for language="cuda" / load_inline); torch + triton included.
ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Pinned KernelBench task set (reproducible train/eval split)
ARG KERNELBENCH_REF=423217d9fda91e0c2d67e4a43bf62f96f6d104f1
RUN git clone https://github.com/ScalingIntelligence/KernelBench /opt/KernelBench && \
    git -C /opt/KernelBench checkout ${KERNELBENCH_REF}

WORKDIR /app/env
COPY . /app/env
# torch/triton come from the base image; don't let pip replace them
RUN pip install --no-cache-dir "openenv-core[core]>=0.2.1" ninja

ENV KERNELBENCH_DIR=/opt/KernelBench/KernelBench \
    PYTHONPATH=/app/env

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
