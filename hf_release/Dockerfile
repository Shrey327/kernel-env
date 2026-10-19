# GPU runtime for the downloadable environment; the free static Space does not run this image.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
ARG KERNELBENCH_REF=423217d9fda91e0c2d67e4a43bf62f96f6d104f1
RUN git clone https://github.com/ScalingIntelligence/KernelBench /opt/KernelBench && \
    git -C /opt/KernelBench checkout ${KERNELBENCH_REF}
RUN pip install --no-cache-dir "openenv-core[core]==0.3.0" ninja
RUN useradd --create-home --uid 1000 kernel

WORKDIR /app
COPY --chown=kernel:kernel kernel_env /app/kernel_env
ENV HOME=/home/kernel PYTHONPATH=/app KERNELBENCH_DIR=/opt/KernelBench/KernelBench \
    KERNEL_ENV_REWARD=roofline KERNEL_ENV_MAX_SESSIONS=8 KERNEL_ENV_EVAL_TIMEOUT=60
USER kernel
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "kernel_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
