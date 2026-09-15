from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any

import httpx
from aiohttp import web
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools, register

PLUGIN_NAME = "astrbot_plugin_vertex_bridge"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_GLOBAL_SITE = None
_GLOBAL_RUNNER = None

FALLBACK_CHAT_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-computer-use-preview-10-2025",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-flash-tts",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-pro-tts",
    "gemini-3-flash-preview",
    "gemini-3-pro-image",
    "gemini-3-pro-image-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-live-preview",
    "gemini-3.1-flash-tts-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-live-translate-preview",
    "gemini-3.5-transcribe",
    "gemini-3.5-transcribe-live",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-embedding-001",
    "gemini-embedding-2",
    "gemini-embedding-2-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-omni-1.1-flash",
    "gemini-omni-flash",
    "gemini-omni-flash-preview",
    "gemini-robotics-er-1.5-preview",
    "gemini-robotics-er-1.6-preview",
    "gemini-robotics-er-2-preview",
    "gemini-robotics-er-2-streaming-preview",
]

# gemini-3.x 系列迁移到全局端点，不再带 region 前缀
_GLOBAL_ENDPOINT_MODELS = {
    "gemini-3-flash-preview", "gemini-3-pro-image", "gemini-3-pro-image-preview", "gemini-3-pro-preview",
    "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-image", "gemini-3.1-flash-lite-preview", "gemini-3.1-flash-live-preview",
    "gemini-3.1-flash-tts-preview", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-live-translate-preview",
    "gemini-3.5-transcribe", "gemini-3.5-transcribe-live",
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "gemini-omni-1.1-flash", "gemini-omni-flash", "gemini-omni-flash-preview",
}
FALLBACK_EMBED_MODELS = ["text-embedding-004"]


def _plugin_data_dir() -> Path:
    try:
        return Path(StarTools.get_data_dir(PLUGIN_NAME)).resolve()
    except Exception:
        from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

        path = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()


def _chmod_secret(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _looks_like_sa(data: Any) -> bool:
    return isinstance(data, dict) and data.get("type") == "service_account" and bool(data.get("private_key"))


def _write_sa_file(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    _chmod_secret(path)
    return path


class TokenManager:
    def __init__(self, data_dir: Path, sa_file=None, sa_content: str = "", sa_path: str = ""):
        self.data_dir = data_dir.resolve()
        self.sa_dir = self.data_dir / "files" / "sa_file"
        self.canonical_path = self.data_dir / "vertex-sa.json"
        self.sa_file = sa_file or []
        self.sa_content = sa_content or ""
        self.sa_path = sa_path or ""
        self.credentials = None
        self.project_id = ""
        self.token = ""
        self.expiry = 0.0
        self.source = ""
        self.reload(sa_file, sa_content, sa_path)

    def _load_from_path(self, path: Path, source: str) -> bool:
        if not path.is_file():
            return False
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                str(path), scopes=SCOPES
            )
            self.project_id = self.credentials.project_id or ""
            self.source = source
            logger.info(f"凭证加载成功 (来源: {source}, 项目: {self.project_id})")
            return True
        except Exception as e:
            logger.error(f"从 {source} 加载凭证失败: {e}")
            return False

    def _load_from_info(self, info: dict, source: str) -> bool:
        if not _looks_like_sa(info):
            return False
        try:
            self.credentials = service_account.Credentials.from_service_account_info(
                info, scopes=SCOPES
            )
            self.project_id = self.credentials.project_id or ""
            self.source = source
            logger.info(f"凭证加载成功 (来源: {source}, 项目: {self.project_id})")
            return True
        except Exception as e:
            logger.error(f"从 {source} 加载凭证失败: {e}")
            return False

    def _resolve_candidate(self, raw: str) -> Path | None:
        text = (raw or "").strip()
        if not text:
            return None
        candidates: list[Path] = []
        p = Path(text)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend(
                [
                    self.data_dir / text,
                    self.sa_dir / text,
                    Path(text),
                ]
            )
        for c in candidates:
            try:
                resolved = c.resolve()
            except OSError:
                continue
            if resolved.is_file():
                return resolved
        return None

    def _persist_canonical(self, data: dict, source: str) -> None:
        try:
            _write_sa_file(self.canonical_path, data)
            logger.info(f"服务账号已写入 plugin_data: {self.canonical_path.name} ({source})")
        except Exception as e:
            logger.error(f"写入 plugin_data 服务账号失败: {e}")

    def _migrate_into_plugin_data(self) -> None:
        """把散落的 SA 收到 data/plugin_data，并清掉插件目录里的私钥。"""
        if self.canonical_path.is_file():
            return
        leftovers = [
            Path(__file__).resolve().parent / "vertex-sa.json",
            Path(__file__).resolve().parent / "config.json",
        ]
        for leftover in leftovers:
            if leftover.name != "vertex-sa.json" or not leftover.is_file():
                continue
            try:
                data = json.loads(leftover.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if _looks_like_sa(data):
                self._persist_canonical(data, f"migrate:{leftover.name}")
                try:
                    leftover.unlink()
                    logger.info(f"已从插件目录移除密钥文件: {leftover.name}")
                except OSError as e:
                    logger.warning(f"无法删除插件目录密钥 {leftover}: {e}")
                return

    def reload(self, sa_file=None, sa_content: str = "", sa_path: str = ""):
        if sa_file is not None:
            self.sa_file = sa_file
        if sa_content is not None:
            self.sa_content = sa_content
        if sa_path is not None:
            self.sa_path = sa_path

        self.credentials = None
        self.project_id = ""
        self.token = ""
        self.expiry = 0.0
        self.source = ""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sa_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_into_plugin_data()

        file_candidates: list[str] = []
        if isinstance(self.sa_file, list):
            file_candidates.extend(str(x) for x in self.sa_file if str(x).strip())
        elif isinstance(self.sa_file, str) and self.sa_file.strip():
            file_candidates.append(self.sa_file.strip())

        for raw in file_candidates:
            path = self._resolve_candidate(raw)
            if path and self._load_from_path(path, f"上传文件 {path.name}"):
                return

        if self.canonical_path.is_file() and self._load_from_path(self.canonical_path, "plugin_data"):
            return

        if self.sa_content and self.sa_content.strip():
            try:
                info = json.loads(self.sa_content.strip())
            except Exception as e:
                logger.error(f"sa_content 不是合法 JSON: {e}")
                info = None
            if isinstance(info, dict) and self._load_from_info(info, "配置文本"):
                self._persist_canonical(info, "sa_content")
                return

        path = self._resolve_candidate(self.sa_path)
        if path and self._load_from_path(path, f"路径 {path.name}"):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if _looks_like_sa(data):
                    self._persist_canonical(data, "sa_path")
            except Exception:
                pass
            return

        if self.sa_dir.is_dir():
            jsons = sorted(p for p in self.sa_dir.glob("*.json") if p.is_file())
            for path in jsons:
                if self._load_from_path(path, f"plugin_data {path.name}"):
                    return

    def get_token(self) -> str:
        if not self.credentials:
            self.reload()
            if not self.credentials:
                raise RuntimeError("未找到有效的 Vertex 服务账号凭证，请在插件配置中上传 JSON")
        now = time.time()
        if not self.token or now >= self.expiry - 60:
            self.credentials.refresh(Request())
            self.token = self.credentials.token
            self.expiry = self.credentials.expiry.timestamp() if self.credentials.expiry else now + 3600
        return self.token


def _build_vertex_url(region: str, project: str, model: str, action: str, extra_qs: str = "") -> str:
    """根据模型名称选择正确的端点前缀（全局 vs 区域）。"""
    base_model = model.split(":")[0]  # 去掉可能已附带的 action
    if base_model in _GLOBAL_ENDPOINT_MODELS:
        host = "aiplatform.googleapis.com"
    else:
        host = f"{region}-aiplatform.googleapis.com"
    url = (
        f"https://{host}/v1/projects/{project}"
        f"/locations/{region}/publishers/google/models/{model}:{action}"
    )
    if extra_qs:
        url += f"?{extra_qs}"
    return url


def _strip_models_prefix(name: str) -> str:
    text = (name or "").strip().lstrip("/")
    if text.startswith("models/"):
        text = text[len("models/") :]
    if "/models/" in text:
        text = text.rsplit("/models/", 1)[-1]
    if text.startswith("publishers/google/models/"):
        text = text[len("publishers/google/models/") :]
    return text


@register(
    PLUGIN_NAME,
    "ElaraKaya",
    "Vertex AI 本地桥接服务",
    "1.1.2",
    "https://github.com/ElaraKaya/astrbot_plugin_vertex_bridge",
)
class VertexBridgePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.port = int(self.config.get("port", 18317))
        raw_region = str(self.config.get("region", "us-central1")).strip() or "us-central1"
        self.region = raw_region.split()[0].split("(")[0].strip() or "us-central1"
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.sa_file = self.config.get("sa_file", [])
        self.sa_content = str(self.config.get("sa_content", "")).strip()
        self.sa_path = str(self.config.get("sa_path", "")).strip()
        self.data_dir = _plugin_data_dir()
        self.token_mgr = TokenManager(self.data_dir, self.sa_file, self.sa_content, self.sa_path)
        self._scrub_config_secrets()
        self._purge_plugin_dir_secrets()
        self.app = web.Application()
        self._setup_proxy_routes()

    def _get_http_client(self) -> httpx.AsyncClient:
        if not hasattr(self, "http_client") or self.http_client.is_closed:
            self.http_client = httpx.AsyncClient(timeout=120.0)
        return self.http_client

    def _scrub_config_secrets(self) -> None:
        """私钥只留 plugin_data；框架配置里尽量不长期存全文。"""
        if not hasattr(self.config, "save_config"):
            return
        changed = False
        if self.sa_content:
            self.config["sa_content"] = ""
            self.sa_content = ""
            changed = True
        current_path = str(self.config.get("sa_path", "")).strip()
        canonical = str(self.token_mgr.canonical_path)
        if current_path != canonical:
            self.config["sa_path"] = canonical
            self.sa_path = canonical
            changed = True
        if changed:
            try:
                self.config.save_config()
            except Exception as e:
                logger.warning(f"清理配置中的密钥副本失败: {e}")

    def _purge_plugin_dir_secrets(self) -> None:
        plugin_dir = Path(__file__).resolve().parent
        for name in ("vertex-sa.json", "config.json"):
            path = plugin_dir / name
            if not path.is_file():
                continue
            try:
                path.unlink()
                logger.info(f"已从插件目录删除 {name}")
            except OSError as e:
                logger.warning(f"无法删除插件目录文件 {path}: {e}")

    async def initialize(self) -> None:
        await self._start_server()

    def _setup_proxy_routes(self):
        self.app.router.add_get("/v1/models", self.handle_openai_models)
        self.app.router.add_post("/v1/chat/completions", self.handle_openai_chat)
        self.app.router.add_post("/v1/embeddings", self.handle_openai_embeddings)
        self.app.router.add_route("*", "/v1beta/{tail:.*}", self.handle_gemini_native)
        self.app.router.add_get("/health", self.handle_health)

    async def _start_server(self):
        global _GLOBAL_SITE, _GLOBAL_RUNNER
        try:
            if _GLOBAL_SITE:
                await _GLOBAL_SITE.stop()
                _GLOBAL_SITE = None
            if _GLOBAL_RUNNER:
                await _GLOBAL_RUNNER.cleanup()
                _GLOBAL_RUNNER = None

            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.port)
            await site.start()
            _GLOBAL_RUNNER = runner
            _GLOBAL_SITE = site
            logger.info(
                f"桥接监听已就绪: 0.0.0.0:{self.port} (区域: {self.region}, 项目: {self.token_mgr.project_id or '未绑定'})"
            )
        except Exception as e:
            logger.error(f"服务启动异常: {e}")

    async def terminate(self):
        global _GLOBAL_SITE, _GLOBAL_RUNNER
        try:
            if hasattr(self, "http_client") and not self.http_client.is_closed:
                await self.http_client.aclose()
            if _GLOBAL_SITE:
                await _GLOBAL_SITE.stop()
                _GLOBAL_SITE = None
            if _GLOBAL_RUNNER:
                await _GLOBAL_RUNNER.cleanup()
                _GLOBAL_RUNNER = None
            logger.info("桥接端口已释放")
        except Exception as e:
            logger.error(f"释放异常: {e}")

    @filter.command("vmodels", alias={"vertex_models"})
    async def cmd_list_models(self, event: AstrMessageEvent):
        """实时拉取 Vertex AI 全量可用模型列表"""
        try:
            token = self.token_mgr.get_token()
            project = self.token_mgr.project_id
        except Exception as e:
            yield event.plain_result(f"凭证获取失败: {e}")
            return

        url = (
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{self.region}/publishers/google/models"
        )
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                body_preview = resp.text[:300]
                msg = "API 返回 " + str(resp.status_code) + ":\n" + body_preview
                yield event.plain_result(msg)
                return
            data = resp.json()
            models = data.get("publisherModels", data.get("models", []))
            if not models:
                yield event.plain_result("没拉到模型，返回体为空。")
                return
            lines = []
            for m in models:
                name = m.get("name", "")
                # 只取最后一段作为短名
                short = name.rsplit("/", 1)[-1] if "/" in name else name
                lines.append(short)
            lines.sort()
            total = len(lines)
            text = "Vertex 全量模型 (" + str(total) + " 个):\n" + "\n".join(lines)
            yield event.plain_result(text)
        except Exception as e:
            yield event.plain_result(f"拉取失败: {e}")

    async def handle_health(self, req):
        return web.json_response(
            {
                "status": "ok",
                "project_id": self.token_mgr.project_id or "not_configured",
                "region": self.region,
                "port": self.port,
                "ready": bool(self.token_mgr.credentials),
                "source": self.token_mgr.source,
            }
        )

    def _model_catalog(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for name in FALLBACK_CHAT_MODELS:
            models.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                }
            )
        for name in FALLBACK_EMBED_MODELS:
            models.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["embedContent"],
                }
            )
        return models

    async def handle_gemini_native(self, req):
        tail = req.match_info.get("tail", "").lstrip("/")
        method = req.method

        if tail == "models" or tail.startswith("models?"):
            return web.json_response({"models": self._model_catalog()})

        body = await req.read()
        try:
            token = self.token_mgr.get_token()
            project = self.token_mgr.project_id
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        clean_tail = tail
        if clean_tail.startswith("models/"):
            clean_tail = clean_tail[len("models/") :]

        # gemini-3.8+ 用全局端点，其余用区域端点
        base_model_name = clean_tail.split(":")[0]
        if base_model_name in _GLOBAL_ENDPOINT_MODELS:
            host = "aiplatform.googleapis.com"
        else:
            host = f"{self.region}-aiplatform.googleapis.com"
        target_url = (
            f"https://{host}/v1/projects/{project}"
            f"/locations/{self.region}/publishers/google/models/{clean_tail}"
        )
        if req.query_string:
            target_url += f"?{req.query_string}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": req.headers.get("Content-Type", "application/json"),
        }

        client = httpx.AsyncClient(timeout=180.0)
        try:
            req_stream = client.build_request(method, target_url, headers=headers, content=body)
            r = await client.send(req_stream, stream=True)
            res = web.StreamResponse(
                status=r.status_code,
                headers={k: v for k, v in r.headers.items() if k.lower() in ["content-type", "cache-control"]},
            )
            await res.prepare(req)
            async for chunk in r.aiter_bytes():
                await res.write(chunk)
            await res.write_eof()
            await r.aclose()
            await client.aclose()
            return res
        except Exception as e:
            await client.aclose()
            return web.json_response({"error": f"Proxy request failed: {e}"}, status=500)

    async def handle_openai_embeddings(self, req):
        try:
            req_data = await req.json()
            model = _strip_models_prefix(req_data.get("model", "text-embedding-004"))
            inputs = req_data.get("input")
            if isinstance(inputs, str):
                inputs = [inputs]
            token = self.token_mgr.get_token()
            project = self.token_mgr.project_id
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

        target_url = _build_vertex_url(self.region, project, model, "predict")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        instances = [{"content": text} for text in inputs]
        payload = {"instances": instances}

        client = self._get_http_client()
        resp = await client.post(target_url, headers=headers, json=payload)
        if resp.status_code != 200:
            return web.Response(body=resp.content, status=resp.status_code, content_type="application/json")
        data = resp.json()
        predictions = data.get("predictions", [])
        openai_data = []
        for idx, pred in enumerate(predictions):
            emb = pred.get("embeddings", {}).get("values", [])
            openai_data.append({"object": "embedding", "index": idx, "embedding": emb})
        return web.json_response(
            {
                "object": "list",
                "data": openai_data,
                "model": model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
        )

    async def handle_openai_models(self, req):
        models = [_strip_models_prefix(m["name"]) for m in self._model_catalog()]
        data = [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": "google"}
            for m in models
        ]
        return web.json_response({"object": "list", "data": data})

    def _openai_content_to_parts(self, content: Any) -> list[dict]:
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            return [{"text": str(content)}]
        parts: list[dict] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"text": item})
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "text" or "text" in item:
                parts.append({"text": item.get("text", "")})
            elif itype in ("image_url", "image"):
                url = item.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url", "")
                url = str(url or "")
                if url.startswith("data:") and "," in url:
                    header, b64 = url.split(",", 1)
                    mime = "image/png"
                    if ";" in header:
                        mime = header[5:].split(";", 1)[0] or mime
                    parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                elif url:
                    parts.append({"text": f"[image_url] {url}"})
        return parts or [{"text": ""}]

    async def handle_openai_chat(self, req):
        try:
            req_data = await req.json()
            model = _strip_models_prefix(req_data.get("model", "gemini-1.5-flash"))
            messages = req_data.get("messages", [])
            stream = bool(req_data.get("stream"))
            token = self.token_mgr.get_token()
            project = self.token_mgr.project_id
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

        contents = []
        system_instruction = None
        for m in messages:
            role = m.get("role")
            parts = self._openai_content_to_parts(m.get("content", ""))
            if role == "system":
                system_instruction = {"parts": parts}
            elif role == "assistant":
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({"role": "user", "parts": parts})

        vertex_body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            vertex_body["systemInstruction"] = system_instruction
        if stream:
            vertex_body.setdefault("generationConfig", {})

        action = "streamGenerateContent" if stream else "generateContent"
        target_url = _build_vertex_url(self.region, project, model, action, "alt=sse" if stream else "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        if stream:
            client = httpx.AsyncClient(timeout=180.0)
            try:
                r = await client.post(target_url, headers=headers, json=vertex_body)
                if r.status_code != 200:
                    body = r.content
                    await client.aclose()
                    return web.Response(body=body, status=r.status_code, content_type="application/json")
                created = int(time.time())
                res = web.StreamResponse(
                    status=200,
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
                )
                await res.prepare(req)
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    payload = line[6:] if line.startswith("data: ") else line
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    text = ""
                    cands = obj.get("candidates") or []
                    if cands:
                        parts = cands[0].get("content", {}).get("parts") or []
                        if parts:
                            text = parts[0].get("text", "")
                    chunk = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": None}],
                    }
                    await res.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                done = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                await res.write(f"data: {json.dumps(done)}\n\n".encode())
                await res.write(b"data: [DONE]\n\n")
                await res.write_eof()
                await r.aclose()
                await client.aclose()
                return res
            except Exception as e:
                await client.aclose()
                return web.json_response({"error": f"Proxy request failed: {e}"}, status=500)

        client = self._get_http_client()
        resp = await client.post(target_url, headers=headers, json=vertex_body)
        if resp.status_code != 200:
            return web.Response(body=resp.content, status=resp.status_code, content_type="application/json")
        v_res = resp.json()
        candidates = v_res.get("candidates", [])
        reply_text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                reply_text = parts[0].get("text", "")
        return web.json_response(
            {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
