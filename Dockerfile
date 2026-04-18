# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY docker/patch_pyproject.py ./patch_pyproject.py
# Strip [tool.uv.*] sections so uv resolves torch from the CPU index we pass.
RUN pip install uv==0.5.11 \
 && python3 patch_pyproject.py \
 && uv pip install --system --no-cache \
      "torch>=2.5,<2.7" "torchvision>=0.17.0" \
      --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --system --no-cache . \
 && uv pip install --system --no-cache \
      fastapi "uvicorn[standard]>=0.29.0" "pydantic>=2.6.0" \
      python-multipart "prometheus-fastapi-instrumentator>=7.0.0" \
      "huggingface-hub>=0.22.0" "pillow>=10.2.0"

FROM python:3.13-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src /app/src
RUN mkdir -p /app/artifacts/main /app/artifacts/baseline /app/data/processed
COPY data/widget /app/data/widget
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "grnti_text_classifier.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
