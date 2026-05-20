"""
Gemini Web -> OpenAI Compatible API Server v3.0
================================================
基于 HanaokaYuzu/Gemini-API 源码构建。
完整 OpenAI 兼容：tools / response_format / 多模态 / system prompt / 多轮对话 / 图片生成

参考:
- https://github.com/Nativu5/Gemini-FastAPI (多账号 + LMDB 持久化)
- https://github.com/zhiyu1998/Gemi2Api-Server (多模态 + 流式优化)

升级: docker compose pull && docker compose up -d
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from gemini_webapi import GeminiClient


# ── Logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("gemini-api")

# ── Paths ──────────────────────────────────────────────────
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/app/config"))
ACCOUNTS_FILE = CONFIG_DIR / "accounts.json"
API_KEY = os.getenv("API_KEY", "")  # 可选 API Key 认证


# ════════════════════════════════════════════════════════════
# Container Monitoring
# ════════════════════════════════════════════════════════════

SERVER_START_TIME = time.time()
SERVER_REQUESTS_TOTAL = 0
SERVER_REQUESTS_FAILED = 0


def _read_proc_mem() -> dict:
    try:
        with open("/proc/self/status") as f:
            lines = f.read()
        mem = {}
        for key in ("VmRSS", "VmSize", "VmPeak"):
            for line in lines.split("\n"):
                if line.startswith(f"{key}:"):
                    mem[key.lower()] = line.split(":")[1].strip()
        try:
            with open("/sys/fs/cgroup/memory.current") as f:
                mem["cgroup_current"] = f"{int(f.read().strip()) // 1048576} MB"
        except Exception:
            pass
        try:
            with open("/sys/fs/cgroup/memory.max") as f:
                val = f.read().strip()
                mem["cgroup_limit"] = "unlimited" if val == "max" else f"{int(val) // 1048576} MB"
        except Exception:
            pass
        return mem
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════
# Account Pool — Multi-Account Round-Robin + Failover
# ════════════════════════════════════════════════════════════

@dataclass
class Account:
    name: str
    secure_1psid: str
    secure_1psidts: str = ""
    proxy: Optional[str] = None
    # runtime
    client: Optional[GeminiClient] = None
    fail_count: int = 0
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_failures: int = 0


class AccountPool:
    def __init__(self):
        self.accounts: list[Account] = []
        self._idx = 0
        self._lock = asyncio.Lock()

    def load(self, path: Path) -> int:
        if not path.exists():
            log.error("accounts.json not found at %s", path)
            return 0
        data = json.loads(path.read_text())
        raw = data.get("accounts", [])
        self.accounts = [
            Account(
                name=a["name"],
                secure_1psid=a["secure_1psid"],
                secure_1psidts=a.get("secure_1psidts", ""),
                proxy=a.get("proxy"),
            )
            for a in raw
        ]
        log.info("Loaded %d account(s): %s", len(self.accounts), [a.name for a in self.accounts])
        return len(self.accounts)

    async def init_all(self, timeout: int = 30):
        if not self.accounts:
            raise RuntimeError("No accounts configured")

        async def _init(acc: Account) -> bool:
            try:
                acc.client = GeminiClient(
                    secure_1psid=acc.secure_1psid,
                    secure_1psidts=acc.secure_1psidts,
                    proxy=acc.proxy,
                )
                await acc.client.init(timeout=timeout, auto_close=False, close_delay=600, auto_refresh=True)
                log.info("[%s] Ready", acc.name)
                return True
            except Exception as e:
                log.error("[%s] Init failed: %s", acc.name, e)
                acc.fail_count = 999
                return False

        results = await asyncio.gather(*[_init(a) for a in self.accounts])
        ready = sum(results)
        log.info("Pool: %d/%d accounts ready", ready, len(self.accounts))
        if ready == 0:
            raise RuntimeError("No accounts initialized")

    async def validate_all(self):
        """初始化后验证会话有效性（发一条静默消息检测 401/403）。"""
        TEST_PROMPT = "Reply with exactly OK."
        AUTH_FAIL_PATTERNS = ["sign in", "signed in", "log in", "logged in", "are you"]

        async def _validate(acc: Account) -> bool:
            if acc.fail_count >= 999 or not acc.client:
                return False
            try:
                resp = await acc.client.generate_content(TEST_PROMPT, temporary=True)
                text = (resp.text or "").strip().lower()
                if not text or any(p in text for p in AUTH_FAIL_PATTERNS):
                    log.warning("[%s] Session validation FAILED — auth degraded", acc.name)
                    acc.fail_count = 999
                    return False
                log.info("[%s] Session validated OK", acc.name)
                return True
            except Exception as e:
                log.warning("[%s] Session validation error: %s", acc.name, str(e)[:100])
                return False

        results = await asyncio.gather(*[_validate(a) for a in self.accounts])
        valid = sum(results)
        if valid == 0:
            log.critical("ALL sessions invalid — cookies may have expired!")
        else:
            log.info("Session validation: %d/%d passed", valid, len(self.accounts))

    async def keepalive(self, interval: int = 7200):
        """后台保活：每隔 interval 秒对所有健康账号发一次静默消息，防止闲置失效。"""
        KEEPALIVE_PROMPT = "Hello"
        while True:
            await asyncio.sleep(interval)
            for a in self.accounts:
                if a.fail_count >= 999 or not a.client:
                    continue
                try:
                    await a.client.generate_content(KEEPALIVE_PROMPT, temporary=True)
                    log.debug("[%s] Keepalive OK", a.name)
                except Exception as e:
                    log.warning("[%s] Keepalive failed: %s", a.name, str(e)[:80])
                    # 保活失败不触发冷却，只记录

    async def get_client(self) -> tuple[GeminiClient, str]:
        async with self._lock:
            now = time.time()
            n = len(self.accounts)
            for _ in range(n):
                acc = self.accounts[self._idx % n]
                self._idx = (self._idx + 1) % n
                if acc.fail_count >= 999:
                    continue
                if now < acc.cooldown_until:
                    continue
                return acc.client, acc.name
            best = min(self.accounts, key=lambda a: a.cooldown_until)
            wait = max(0, best.cooldown_until - now)
            raise RuntimeError(f"All {n} accounts down. Next: {best.name} in {wait:.0f}s")

    async def success(self, name: str):
        for a in self.accounts:
            if a.name == name:
                a.fail_count = 0
                a.total_requests += 1
                return

    async def failure(self, name: str, err: str):
        err_lower = err.lower()
        for a in self.accounts:
            if a.name == name:
                a.fail_count += 1
                a.total_failures += 1

                # 429 / rate limit → 立即标记冷却（不等 3 次）
                if "429" in err or "rate" in err_lower or "quota" in err_lower:
                    a.cooldown_until = time.time() + 300
                    log.warning("[%s] RATE LIMITED — cooldown 5min: %s", name, err[:120])
                    return

                # 401 / auth failure → 立即标记为 dead
                if "401" in err or "unauthorized" in err_lower or "authenticated" in err_lower:
                    a.fail_count = 999
                    log.error("[%s] AUTH FAILED — marked dead (cookie expired?): %s", name, err[:120])
                    return

                # 常规失败：累积 3 次后冷却
                if a.fail_count >= 3:
                    a.cooldown_until = time.time() + 300
                    log.warning("[%s] COOLDOWN 5min (x%d): %s", name, a.fail_count, err[:120])
                else:
                    log.warning("[%s] Fail %d/3: %s", name, a.fail_count, err[:120])
                return

    def list_models(self) -> list[dict]:
        for a in self.accounts:
            if a.client and a.fail_count < 999:
                try:
                    return [{"id": m.model_name, "display": m.display_name} for m in a.client.list_models()]
                except Exception:
                    continue
        return [{"id": "gemini-2.5-flash", "display": "Gemini 2.5 Flash (fallback)"}]

    def stats(self) -> list[dict]:
        now = time.time()
        return [
            {
                "name": a.name,
                "status": (
                    "cooldown" if now < a.cooldown_until else
                    "dead" if a.fail_count >= 999 else "active"
                ),
                "fails": a.fail_count,
                "cooldown_s": max(0, int(a.cooldown_until - now)),
                "requests": a.total_requests,
                "failures": a.total_failures,
            }
            for a in self.accounts
        ]


# ════════════════════════════════════════════════════════════
# FastAPI App
# ════════════════════════════════════════════════════════════

pool = AccountPool()


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = pool.load(ACCOUNTS_FILE)
    if n == 0:
        log.critical("No accounts! Edit config/accounts.json")
    else:
        await pool.init_all()
        await pool.validate_all()  # 验证会话有效性
    # 启动后台保活任务
    keepalive_task = asyncio.create_task(pool.keepalive(interval=7200))
    yield
    keepalive_task.cancel()
    for a in pool.accounts:
        if a.client:
            try:
                await a.client.close()
            except Exception:
                pass


app = FastAPI(title="Gemini Web API", version="3.0.0", lifespan=lifespan)

# CORS — 允许 ChatBox / OpenClaw / Claude Code 等客户端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════
# OpenAI Compatible Schemas
# ════════════════════════════════════════════════════════════

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None

class Message(BaseModel):
    role: str
    content: str | list[ContentPart]
    name: Optional[str] = None

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ResponseFormat(BaseModel):
    type: str = "text"  # "text" | "json_object" | "json_schema"
    json_schema: Optional[dict] = None

class ChatCompletionRequest(BaseModel):
    model: str = "gemini-2.5-flash"
    messages: list[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[str | dict] = None
    response_format: Optional[ResponseFormat] = None


# ════════════════════════════════════════════════════════════
# Message Construction — 多模态 + system prompt + tools
# ════════════════════════════════════════════════════════════

def _extract_text(content: str | list) -> str:
    """提取纯文本"""
    if isinstance(content, str):
        return content
    texts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        elif hasattr(item, "type") and item.type == "text":
            texts.append(item.text or "")
    return "\n".join(texts)


def _extract_images(content: str | list) -> list[str]:
    """提取 base64 图片并保存为临时文件，返回文件路径列表"""
    if isinstance(content, str):
        return []
    files = []
    for item in content:
        if isinstance(item, dict):
            item = ContentPart(**item)
        if hasattr(item, "image_url") and item.image_url:
            url = item.image_url.get("url", "") if isinstance(item.image_url, dict) else ""
            if url.startswith("data:image/"):
                try:
                    b64 = url.split(",", 1)[1]
                    data = base64.b64decode(b64)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                        f.write(data)
                        files.append(f.name)
                except Exception as e:
                    log.warning("Failed to decode image: %s", e)
    return files


def _build_tools_prompt(tools: Optional[list[Tool]]) -> str:
    """将 OpenAI tools 转为 prompt 指令"""
    if not tools:
        return ""
    lines = ["[Available Functions]", "You may call these functions by responding with:"]
    lines.append('{"name": "<function_name>", "arguments": <json_object>}')
    lines.append("")
    for t in tools:
        f = t.function
        lines.append(f"## {f.name}")
        if f.description:
            lines.append(f.description)
        if f.parameters:
            lines.append(f"Parameters: {json.dumps(f.parameters, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def _build_response_format_prompt(rf: Optional[ResponseFormat]) -> str:
    """将 response_format 转为 prompt 指令"""
    if not rf:
        return ""
    if rf.type == "json_object":
        return "\n[Output Format]\nYou MUST respond with a valid JSON object. No markdown, no explanation — just the JSON.\n"
    if rf.type == "json_schema" and rf.json_schema:
        schema_name = rf.json_schema.get("name", "output")
        schema = rf.json_schema.get("schema", rf.json_schema)
        return (
            f"\n[Output Format — {schema_name}]\n"
            "You MUST respond with a valid JSON object conforming to this schema:\n"
            f"```json\n{json.dumps(schema, ensure_ascii=False)}\n```\n"
            "No markdown wrapping, no explanation — just the JSON object.\n"
        )
    return ""


def build_prompt(messages: list[Message], tools: Optional[list[Tool]] = None,
                 response_format: Optional[ResponseFormat] = None) -> tuple[str, list[str]]:
    """
    构建 Gemini prompt：
    - system 消息作为 system instruction
    - user/assistant 交替拼接
    - 多模态图片提取为临时文件
    - tools 和 response_format 追加到 prompt
    """
    parts: list[str] = []
    image_files: list[str] = []

    for msg in messages:
        text = _extract_text(msg.content)
        images = _extract_images(msg.content)
        image_files.extend(images)

        if msg.role == "system":
            parts.append(f"[System Instruction]\n{text}")
        elif msg.role == "user":
            parts.append(text)
        elif msg.role == "assistant":
            parts.append(text)
        # tool / function roles: treat as assistant context
        elif msg.role in ("tool", "function"):
            parts.append(f"[Function Result]\n{text}")

    prompt = "\n\n".join(parts)

    # 追加 tools 提示
    tools_text = _build_tools_prompt(tools)
    if tools_text:
        prompt += f"\n\n{tools_text}"

    # 追加 response_format 提示
    rf_text = _build_response_format_prompt(response_format)
    if rf_text:
        prompt += f"\n\n{rf_text}"

    return prompt, image_files


# ════════════════════════════════════════════════════════════
# API Key Auth
# ════════════════════════════════════════════════════════════

async def verify_api_key(request: Request):
    if not API_KEY:
        return  # 未设置 API_KEY 则跳过认证
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = auth
    if token != API_KEY:
        raise HTTPException(401, "Invalid API key")


# ════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    uptime = int(time.time() - SERVER_START_TIME)
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "uptime_human": f"{uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s",
        "server": {
            "requests_total": SERVER_REQUESTS_TOTAL,
            "requests_failed": SERVER_REQUESTS_FAILED,
        },
        "memory": _read_proc_mem(),
        "accounts": pool.stats(),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "owned_by": "google"}
            for m in pool.list_models()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global SERVER_REQUESTS_TOTAL
    if not req.messages:
        raise HTTPException(400, "messages required")

    prompt, image_files = build_prompt(req.messages, req.tools, req.response_format)
    rid = f"chatcmpl-{uuid.uuid4()}"

    try:
        if req.stream:
            return StreamingResponse(
                _stream_response(prompt, image_files, req.model, rid),
                media_type="text/event-stream",
            )

        # ── Non-streaming with retry ─────────────────────
        retries = max(1, len(pool.accounts))
        last_err = None
        for attempt in range(retries):
            acc_name = "?"
            try:
                client, acc_name = await pool.get_client()
                log.info("[%s] %s...", acc_name, prompt[:80])

                gen_kwargs = {"model": req.model}
                if image_files:
                    gen_kwargs["files"] = image_files

                resp = await client.generate_content(prompt, **gen_kwargs)
                await pool.success(acc_name)
                SERVER_REQUESTS_TOTAL += 1

                text = resp.text if resp and resp.text else ""
                # 追加图片 markdown
                text += _extract_image_markdown(resp)

                return {
                    "id": rid,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            except Exception as e:
                last_err = str(e)
                try:
                    await pool.failure(acc_name, last_err)
                except Exception:
                    pass
        raise HTTPException(503, f"All accounts failed. Last: {last_err}")

    finally:
        # 清理临时图片文件
        for f in image_files:
            try:
                os.unlink(f)
            except Exception:
                pass


@app.middleware("http")
async def _count_failed_requests(request: Request, call_next):
    global SERVER_REQUESTS_FAILED
    response = await call_next(request)
    if response.status_code >= 500:
        SERVER_REQUESTS_FAILED += 1
    return response


# ════════════════════════════════════════════════════════════
# Streaming
# ════════════════════════════════════════════════════════════

async def _stream_response(prompt: str, image_files: list[str], model: str, rid: str):
    global SERVER_REQUESTS_TOTAL

    def _chunk(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    retries = max(1, len(pool.accounts))
    image_files_consumed: list[str] = list(image_files)  # copy for retry

    for _ in range(retries):
        acc_name = "?"
        try:
            client, acc_name = await pool.get_client()
            log.info("[%s] stream: %s...", acc_name, prompt[:80])

            yield _chunk({"role": "assistant"})

            gen_kwargs = {"model": model}
            if image_files_consumed:
                gen_kwargs["files"] = image_files_consumed

            buffer = ""
            yielded_images = 0

            async for chunk in client.generate_content_stream(prompt, **gen_kwargs):
                # 处理图片（流式中的内联图片）
                if hasattr(chunk, "images") and chunk.images and len(chunk.images) > yielded_images:
                    new_imgs = chunk.images[yielded_images:]
                    for img in new_imgs:
                        url = getattr(img, "url", None)
                        if url:
                            yield _chunk({"content": f"\n\n![Image]({url})\n\n"})
                    yielded_images = len(chunk.images)

                if chunk.text_delta:
                    buffer += chunk.text_delta
                    # 安全 yield：不在 markdown 链接中间切断
                    if buffer[-1].isspace() or len(buffer) > 200:
                        yield _chunk({"content": buffer})
                        buffer = ""

            if buffer:
                yield _chunk({"content": buffer})

            yield _chunk({}, "stop")
            yield "data: [DONE]\n\n"

            await pool.success(acc_name)
            SERVER_REQUESTS_TOTAL += 1
            return

        except Exception as e:
            try:
                await pool.failure(acc_name, str(e))
            except Exception:
                pass

    yield _chunk({"content": "\n\n[All accounts unavailable]"}, "stop")
    yield "data: [DONE]\n\n"


# ════════════════════════════════════════════════════════════
# Image Helpers
# ════════════════════════════════════════════════════════════

def _extract_image_markdown(response) -> str:
    """从 Gemini 响应中提取图片 URL 转为 markdown"""
    if not hasattr(response, "images") or not response.images:
        return ""
    parts = []
    for img in response.images:
        url = getattr(img, "url", None)
        alt = getattr(img, "alt", None) or getattr(img, "title", None) or "Image"
        if url:
            parts.append(f"\n\n![{alt}]({url})")
    return "\n".join(parts)
