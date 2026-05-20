# Gemini API Server

基于 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 源码构建的 OpenAI 兼容 API 服务器。

**特性：**
- ✅ 多账号轮询 + 故障隔离（429 立即切换，401 标记死亡）
- ✅ Cookie 自动刷新 + 持久化（Docker Volume）
- ✅ **Cookie 保活**：每 2 小时自动 warmup，防止闲置失效
- ✅ **会话验证**：启动时检测 Cookie 有效性
- ✅ OpenAI 兼容：`/v1/chat/completions` + `/v1/models` + `/health`
- ✅ 流式 SSE / System Prompt / 多模态图片 / JSON Schema / Tool Calling
- ✅ CORS 全开放（ChatBox / OpenClaw / Claude Code 直接用）
- ✅ Docker HEALTHCHECK + 容器级监控（内存/请求计数/运行时间）
- ✅ GitHub Actions 每 30 分钟检测上游更新，自动构建镜像

## Cookie 获取（必读）

> ⚠️ **关键步骤，做错 Cookie 秒失效**

1. 打开 Chrome **无痕窗口**（Ctrl+Shift+N）
2. 登录 [gemini.google.com](https://gemini.google.com)
3. F12 → Application → Cookies → `gemini.google.com`
4. 复制 `__Secure-1PSID` 和 `__Secure-1PSIDTS`
5. **立刻关闭无痕窗口！** ← 最重要的步骤

> 谷歌会检测"多端同时在线"。如果你本地浏览器还开着 Gemini 页面，服务器端会被踢下线。无痕窗口关闭后，服务端就成了唯一的"合法客户端"，Cookie 可存活数周甚至数月。

## 快速部署

```bash
mkdir -p config cache

cat > config/accounts.json << 'EOF'
{
  "accounts": [
    {
      "name": "member-1",
      "secure_1psid": "你的 __Secure-1PSID",
      "secure_1psidts": "你的 __Secure-1PSIDTS（可选）",
      "proxy": null
    }
  ]
}
EOF

docker compose up -d
```

## API 端点

| 端点 | 功能 |
|------|------|
| `GET /health` | 容器监控 + 账号状态 |
| `GET /v1/models` | 可用模型列表 |
| `POST /v1/chat/completions` | 完整 OpenAI 兼容对话 |

## 接入你的工具

除 API Key 外，其他填写 `http://你的IP:8787/v1`

```yaml
# Hermes Agent (config.yaml)
custom_providers:
  gemini-web:
    base_url: http://100.80.1.3:8787/v1
    api_key: ""
    models: ["gemini-3-flash", "gemini-3-pro"]
```

**OpenClaw / ChatBox / Claude Code / OpenCode**：
```
API Base: http://100.80.1.3:8787/v1
API Key: 留空
Model: gemini-3-flash / gemini-3-pro
```

## 多账号配置

```json
{
  "accounts": [
    {"name": "account-1", "secure_1psid": "...", "secure_1psidts": "...", "proxy": null},
    {"name": "account-2", "secure_1psid": "...", "secure_1psidts": "...", "proxy": null},
    {"name": "account-3", "secure_1psid": "...", "secure_1psidts": "...", "proxy": null}
  ]
}
```

每个账号可单独配置代理：
```json
{"name": "account-3", "secure_1psid": "...", "proxy": "socks5://127.0.0.1:1080"}
```

## 故障处理机制

| 错误类型 | 行为 |
|---------|------|
| **429 限流** | 立即冷却该账号 5 分钟，自动切换下一个 |
| **401 认证失败** | 立即标记账号死亡（cookie 过期） |
| **常规错误** | 累积 3 次后冷却 5 分钟 |
| **所有账号不可用** | 返回 503，选最快恢复的账号 |

## 容器健康监控

```bash
# 容器状态
docker ps --filter name=gemini-api

# 详细监控
curl http://localhost:8787/health
# → uptime, memory (VmRSS + cgroup), request counts, per-account status

# 查看日志（含保活心跳）
docker logs -f gemini-api
```

## Cookie 保活

容器启动后自动运行后台保活任务：每 2 小时对所有健康账号发一条静默消息，**防止 Cookie 因长期闲置被 Google 回收**。

无需额外配置，自动运行。日志中 `[account-name] Keepalive OK` 即为成功。

## 网络要求

如果服务器 IP 被 Google 限制（常见于廉价 VPS）：
1. 安装 [Cloudflare WARP](https://developers.cloudflare.com/warp-client/) 解锁 Google 访问
2. 或使用代理：在 `accounts.json` 中为每个账号配置 `proxy` 字段

## 升级

```bash
docker compose pull && docker compose up -d
```

## 镜像

预构建镜像自动推送至 `ghcr.io/talwayh1/gemini-api-server:latest`。

GitHub Actions 每 30 分钟检测 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 上游更新，有新 commit 自动构建。
