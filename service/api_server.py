"""
Gemini Web -> OpenAI Compatible API Server
===========================================
基于 HanaokaYuzu/Gemini-API (PyPI: gemini-webapi)
多账号轮询 + 故障隔离 + Cookie 自动持久化
升级: docker compose build --no-cache && docker compose up -d
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gemini_webapi import GeminiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("gemini-pool")

CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/app/config"))
ACCOUNTS_FILE = CONFIG_DIR / "accounts.json"


# ============================================================
# Account Pool - Round-Robin + Failover + Cooldown
# ============================================================

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
                await acc.client.init(
                    timeout=timeout,
                    auto_close=False,
                    close_delay=600,
                    auto_refresh=True,
                )
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
        for a in self.accounts:
            if a.name == name:
                a.fail_count += 1
                a.total_failures += 1
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
                    return [
                        {"id": m.model_name, "display": m.display_name}
                        for m in a.client.list_models()
                    ]
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


# ============================================================
# FastAPI App
# ============================================================

pool = AccountPool()

@asynccontextmanager
async def lifespan(app: FastAPI):
    n = pool.load(ACCOUNTS_FILE)
    if n == 0:
        log.critical("No accounts! Edit config/accounts.json")
    else:
        await pool.init_all()
    yield
    for a in pool.accounts:
        if a.client:
            try:
                await a.client.close()
            except Exception:
                pass

app = FastAPI(title="Gemini Web API", version="2.0.0", lifespan=lifespan)


class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str = "gemini-2.5-flash"
    messages: list[Msg]
    stream: bool = False

class Choice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"

class ChatResp(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]


def _prompt(msgs: list[Msg]) -> str:
    parts = []
    for m in msgs:
        if m.role == "system":
            parts.append(f"[System]\n{m.content}")
        else:
            parts.append(m.content)
    return "\n\n".join(parts)


@app.get("/health")
async def health():
    return {"status": "ok", "accounts": pool.stats()}

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
async def chat(req: ChatReq):
    if not req.messages:
        raise HTTPException(400, "messages required")

    prompt = _prompt(req.messages)
    rid = f"chatcmpl-{int(time.time()*1000)}"

    if req.stream:
        return StreamingResponse(_stream(req.model, prompt, rid), media_type="text/event-stream")

    retries = max(1, len(pool.accounts))
    last_err = None
    for i in range(retries):
        acc_name = "?"
        try:
            client, acc_name = await pool.get_client()
            log.info("[%s] %s...", acc_name, prompt[:80])
            resp = await client.generate_content(prompt, model=req.model)
            await pool.success(acc_name)
            text = resp.text if resp and resp.text else ""
            return ChatResp(
                id=rid, created=int(time.time()), model=req.model,
                choices=[Choice(message={"role": "assistant", "content": text})],
            )
        except Exception as e:
            last_err = str(e)
            try:
                await pool.failure(acc_name, last_err)
            except Exception:
                pass
    raise HTTPException(503, f"All accounts failed. Last: {last_err}")


async def _stream(model: str, prompt: str, rid: str):
    import json as _json
    retries = max(1, len(pool.accounts))
    for _ in range(retries):
        acc_name = "?"
        try:
            client, acc_name = await pool.get_client()
            log.info("[%s] stream: %s...", acc_name, prompt[:80])
            yield f"data: {_json.dumps({'id':rid,'object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{'role':'assistant'}}]})}\n\n"
            async for chunk in client.generate_content_stream(prompt, model=model):
                if chunk.text_delta:
                    yield f"data: {_json.dumps({'id':rid,'object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{'content':chunk.text_delta}}]})}\n\n"
            yield f"data: {_json.dumps({'id':rid,'object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            await pool.success(acc_name)
            return
        except Exception as e:
            try:
                await pool.failure(acc_name, str(e))
            except Exception:
                pass
    yield f"data: {_json.dumps({'error':{'message':'All accounts failed'}})}\n\n"
    yield "data: [DONE]\n\n"
