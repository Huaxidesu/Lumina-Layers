FROM python:3.10-slim-bookworm AS builder
WORKDIR /app

# pycairo/rlpycairo 链路需要 cairo + pkg-config；build-essential 更稳（含 make 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install -U pip setuptools wheel \
    && pip install --user --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

FROM python:3.10-slim-bookworm
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 7860
CMD ["python", "main.py"]
