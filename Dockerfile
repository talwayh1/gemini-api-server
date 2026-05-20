FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/talwayh1/gemini-api-server"
LABEL org.opencontainers.image.description="Gemini Web -> OpenAI API. Multi-account pool. Built from HanaokaYuzu/Gemini-API source."

# git 是 pip install git+https 的前置依赖
RUN apt-get update && apt-get install -y --no-install-recommends git \
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

CMD ["uvicorn", "service.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
