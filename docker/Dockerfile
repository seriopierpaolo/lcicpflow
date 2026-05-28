FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/lcicpflow
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -e ".[dev,hdbscan]"

CMD ["bash"]
