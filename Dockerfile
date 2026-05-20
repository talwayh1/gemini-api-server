FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/talwayh1/gemini-api-server"
LABEL org.opencontainers.image.description="Gemini Web -> OpenAI API. Multi-account pool. Built from HanaokaYuzu/Gemini-API source."

# git (pip install git+https) + curl (HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# 直接从 Gemini-API 源码安装（始终最新）
RUN pip install --no-cache-dir \
    "git+https://github.com/HanaokaYuzu/Gemini-API.git" \
    fastapi \
    "uvicorn[standard]" \
    pydantic

WORKDIR /app
COPY service/ /app/service/

ENV GEMINI_COOKIE_PATH=/app/cache
EXPOSE 8000

# 容器健康探针 — 每 30s 检查，连续 3 次失败标记 unhealthy
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "service.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
