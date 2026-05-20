# Gemini API Server

基于 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 源码构建的 OpenAI 兼容 API 服务器。

**特性：**
- ✅ 直接从 Gemini-API 源码安装（始终最新）
- ✅ 多账号轮询 + 故障隔离
- ✅ Cookie 自动刷新 + 持久化
- ✅ OpenAI 兼容接口 (`/v1/chat/completions`)
- ✅ 流式输出 (SSE)
- ✅ GitHub Actions 每日自动构建镜像

## 快速部署

```bash
# 1. 准备 Cookie 配置
mkdir -p config cache
cat > config/accounts.json << 'EOF'
{
  "accounts": [
    {
      "name": "member-1",
      "secure_1psid": "你的 __Secure-1PSID",
      "secure_1psidts": "你的 __Secure-1PSIDTS",
      "proxy": null
    }
  ]
}
EOF

# 2. 拉取预构建镜像并启动
docker compose up -d

# 3. 测试
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-3-flash","messages":[{"role":"user","content":"你好"}]}'
```

## 升级

```bash
docker compose pull && docker compose up -d
```

## 镜像

预构建镜像推送至 `ghcr.io/talwayh1/gemini-api-server:latest`，GitHub Actions 每日自动构建。
