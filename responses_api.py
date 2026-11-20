"""OpenAI Responses API 兼容节点，底层转接 Vertex AI Interactions API。

只依赖 main.py 注入的 token_mgr / region / http client getter，不反向 import main。
注册入口：VertexResponsesEndpoint.register(router)。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import OrderedDict
from typing import Any

from aiohttp import web

from astrbot.api import logger

INTERACTIONS_PATH = "/v1beta1/projects/{project}/locations/global/interactions"

# 本地回放缓存上限，防内存无上限增长
_CACHE_LIMIT = 256

# Responses tools type -> Interactions tools type
_TOOL_TYPE_MAP = {
    "web_search": "google_search",
    "web_search_preview": "google_search",
    "web_search_2025_08_26": "google_search",
    "code_interpreter": "code_execution",
    "url_context": "url_context",
    "file_search": "file_search",
    "mcp": "mcp_server",
    "bash": "bash",
    "computer_use": "computer_use",
    "computer_use_preview": "computer_use",
}

# Interactions tools 里原样放行的 type
_PASSTHROUGH_TOOLS = {
    "function",
    "google_search",
    "code_execution",
    "url_context",
    "file_search",
    "mcp_server",
    "bash",
    "computer_use",
    "filesystem",
    "retrieval",
    "google_maps",
    "tool_search",
    "manage_task",
    "schedule",
}


def _text_of(content: Any) -> str:
    """把 content 可能是 str / list[dict] 的值拍平成纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is None:
                    text = part.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content)


def _split_data_url(value: str) -> tuple[str, str] | None:
    """data:<mime>;base64,<payload> -> (mime, payload)。非 data url 返回 None。"""
    if not value.startswith("data:") or "," not in value:
        return None
    header, payload = value.split(",", 1)
    mime = header[5:].split(";", 1)[0].strip() or "application/octet-stream"
    return mime, payload.strip()


def _media_part(part: dict) -> dict | None:
    """Responses 的 input_image / input_file -> Interactions content part。

    官方 input 的 content 是扁平 part：
    {"type": "image"|"document"|"audio"|"video", "data": base64, "mime_type": ...}
    或 {"type": ..., "uri": ..., "mime_type": ...}。
    """
    ptype = str(part.get("type") or "")
    if ptype in ("input_image", "image", "image_url"):
        kind = "image"
    elif ptype in ("input_file", "file", "document"):
        kind = "document"
    elif ptype in ("input_audio", "audio"):
        kind = "audio"
    elif ptype in ("input_video", "video"):
        kind = "video"
    else:
        return None

    source = part.get("image_url") or part.get("file_url") or part.get("url") or ""
    if isinstance(source, dict):
        source = source.get("url") or source.get("uri") or ""
    source = str(source or "")

    raw_b64 = part.get("file_data") or part.get("data") or ""
    mime = str(part.get("mime_type") or part.get("media_type") or "")

    if isinstance(raw_b64, str) and raw_b64.startswith("data:"):
        parsed = _split_data_url(raw_b64)
        if parsed:
            mime, raw_b64 = parsed
    if not raw_b64 and source:
        parsed = _split_data_url(source)
        if parsed:
            mime, raw_b64 = parsed
            source = ""

    if not mime:
        mime = {
            "image": "image/png",
            "document": "application/pdf",
            "audio": "audio/wav",
            "video": "video/mp4",
        }[kind]

    if raw_b64:
        return {"type": kind, "data": str(raw_b64), "mime_type": mime}
    if source.startswith(("http://", "https://", "gs://")):
        return {"type": kind, "uri": source, "mime_type": mime}
    return None


def _content_parts(content: Any) -> list[dict]:
    """Responses message.content -> Interactions user_input.content。"""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        text = str(content)
        return [{"type": "text", "text": text}] if text else []

    parts: list[dict] = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        media = _media_part(part)
        if media:
            parts.append(media)
            continue
        text = part.get("text")
        if text is None:
            text = part.get("content")
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
    return parts


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


class VertexResponsesEndpoint:
    """把 /v1/responses 的请求翻译成 Interactions API 调用。"""

    def __init__(
        self,
        token_mgr: Any,
        region: str,
        http_client_getter: Any,
        *,
        enabled: bool = False,
        models: list[str] | None = None,
        enable_flex: bool = True,
    ) -> None:
        self.token_mgr = token_mgr
        self.region = region
        self._get_http_client = http_client_getter
        self.enabled = bool(enabled)
        self.models = {m.strip() for m in (models or []) if m and m.strip()}
        self.enable_flex = bool(enable_flex)
        # response_id -> 原始 steps（含 thought signature），用于 reasoning 被客户端丢掉时兜底
        self._steps_cache: OrderedDict[str, list[dict]] = OrderedDict()
        # call_id -> response_id，用来反查某个 function_call 属于哪一轮
        self._call_owner: OrderedDict[str, str] = OrderedDict()

    # ---------------------------------------------------------------- 路由

    def register(self, router: web.UrlDispatcher) -> bool:
        """开关关着就不注册，让 /v1/responses 自然落到 404。"""
        if not self.enabled:
            logger.info("[responses] enable_responses=false，跳过 /v1/responses 路由注册")
            return False
        router.add_post("/v1/responses", self.handle_responses)
        router.add_post("/v1/responses:stream", self.handle_responses)
        logger.info("[responses] /v1/responses 已注册")
        return True

    # ---------------------------------------------------------------- 入口

    async def handle_responses(self, req: web.Request) -> web.StreamResponse | web.Response:
        try:
            body = await req.json()
        except Exception as e:
            return web.json_response({"error": {"message": f"invalid JSON body: {e}"}}, status=400)

        if not isinstance(body, dict):
            return web.json_response({"error": {"message": "request body must be an object"}}, status=400)

        try:
            parsed = self._parse_request(body)
        except ValueError as e:
            return web.json_response({"error": {"message": str(e)}}, status=400)

        try:
            token = self.token_mgr.get_token()
            project = self.token_mgr.project_id
        except Exception as e:
            return web.json_response(
                {"error": {"message": f"vertex credentials unavailable: {e}"}}, status=503
            )

        if not project:
            return web.json_response(
                {"error": {"message": "vertex project id is empty, check service account"}}, status=503
            )

        url = f"https://aiplatform.googleapis.com{INTERACTIONS_PATH.format(project=project)}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if self.enable_flex:
            headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
            headers["X-Vertex-AI-LLM-Shared-Request-Type"] = "flex"

        if parsed["stream"]:
            return await self._run_stream(req, url, headers, parsed)
        return await self._run_non_stream(url, headers, parsed)

    # ---------------------------------------------------------------- 缓存

    def _cache_put(self, response_id: str, steps: list[dict]) -> None:
        if not response_id or not steps:
            return
        self._steps_cache[response_id] = steps
        self._steps_cache.move_to_end(response_id)
        for step in steps:
            if isinstance(step, dict) and step.get("type") == "function_call" and step.get("id"):
                self._call_owner[str(step["id"])] = response_id
                self._call_owner.move_to_end(str(step["id"]))
        while len(self._steps_cache) > _CACHE_LIMIT:
            self._steps_cache.popitem(last=False)
        while len(self._call_owner) > _CACHE_LIMIT * 4:
            self._call_owner.popitem(last=False)

    def _recover_missing_thought(self, steps: list[dict]) -> list[dict]:
        """客户端没回传 reasoning 时，用本地缓存把 thought signature 补回去。

        gemini-4-argon 在有 function_call 的历史里强制要求 thought 原样回放，
        缺了直接 400 invalid argument。补在第一个 function_call 之前。
        """
        if any(isinstance(s, dict) and s.get("type") == "thought" for s in steps):
            return steps
        first_fc_idx = next(
            (i for i, s in enumerate(steps) if isinstance(s, dict) and s.get("type") == "function_call"),
            None,
        )
        if first_fc_idx is None:
            return steps
        for step in steps[first_fc_idx:]:
            if not isinstance(step, dict) or step.get("type") != "function_call":
                continue
            owner = self._call_owner.get(str(step.get("id") or ""))
            if not owner:
                continue
            cached = self._steps_cache.get(owner) or []
            for old in cached:
                if isinstance(old, dict) and old.get("type") == "thought" and old.get("signature"):
                    logger.debug("[responses] 从缓存补回 thought signature (call %s)", step.get("id"))
                    return steps[:first_fc_idx] + [old] + steps[first_fc_idx:]
        return steps

    # ------------------------------------------------------------ 请求解析

    def _parse_request(self, body: dict) -> dict:
        model = str(body.get("model") or "").strip()
        if not model:
            raise ValueError("model is required")
        model = self._normalize_model(model)
        if self.models and model not in self.models:
            raise ValueError(f"model '{model}' 不在 responses_models 白名单内")

        raw_input = body.get("input", body.get("messages"))
        if raw_input is None:
            prev_id = str(body.get("previous_response_id") or "").strip()
            cached = self._steps_cache.get(prev_id) if prev_id else None
            if cached:
                raw_input = list(cached)
            else:
                raise ValueError("input is required")

        call_names: dict[str, str] = {}
        prev_id = str(body.get("previous_response_id") or "").strip()
        if prev_id:
            cached = self._steps_cache.get(prev_id)
            if cached:
                if isinstance(raw_input, str):
                    extra = [self._user_step(raw_input)]
                elif isinstance(raw_input, list):
                    extra = [x for x in raw_input if isinstance(x, dict)]
                else:
                    extra = []
                raw_input = list(cached) + extra
        steps = self._convert_input(raw_input, call_names)

        if not steps:
            raise ValueError("input 里没有可用的 user_input")

        payload: dict[str, Any] = {
            "model": model,
            "input": steps,
            "store": False,
        }

        instructions = body.get("instructions") or body.get("system")
        if instructions:
            payload["system_instruction"] = _text_of(instructions)

        gen_config: dict[str, Any] = {}
        for src, dst in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("max_output_tokens", "max_output_tokens"),
            ("max_tokens", "max_output_tokens"),
        ):
            value = body.get(src)
            if value is not None:
                gen_config[dst] = value
        for key, value in (body.get("generation_config") or {}).items():
            gen_config[key] = value
        if gen_config:
            payload["generation_config"] = gen_config

        tools = self._convert_tools(body.get("tools"))
        if tools:
            payload["tools"] = tools

        tool_choice = body.get("tool_choice")
        if tool_choice and tool_choice != "auto":
            payload["tool_choice"] = tool_choice

        return {
            "model": model,
            "steps": steps,
            "payload": payload,
            "stream": bool(body.get("stream")),
            "call_names": call_names,
            "response_id": "resp_" + uuid.uuid4().hex,
            "created": int(time.time()),
        }

    @staticmethod
    def _normalize_model(name: str) -> str:
        text = (name or "").strip().lstrip("/")
        for prefix in ("models/", "publishers/google/models/"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
        if "/models/" in text:
            text = text.rsplit("/models/", 1)[-1]
        return text.split(":")[0]

    def _convert_input(self, raw_input: Any, call_names: dict[str, str]) -> list[dict]:
        """Responses input（str 或 item 数组）-> Interactions step_list。"""
        if isinstance(raw_input, str):
            return [self._user_step(raw_input)]

        if isinstance(raw_input, dict):
            raw_input = [raw_input]

        if not isinstance(raw_input, list):
            return []

        steps: list[dict] = []
        for item in raw_input:
            if isinstance(item, str):
                steps.append(self._user_step(item))
                continue
            if not isinstance(item, dict):
                continue

            itype = item.get("type")

            if itype == "message":
                role = item.get("role", "user")
                if role == "assistant":
                    text = _text_of(item.get("content"))
                    if text:
                        steps.append(self._model_step(text))
                else:
                    parts = _content_parts(item.get("content"))
                    if parts:
                        steps.append({"type": "user_input", "content": parts})
                continue

            if itype in (None, "user_input", "input_text"):
                parts = _content_parts(item.get("content", item.get("text", "")))
                if parts:
                    steps.append({"type": "user_input", "content": parts})
                continue

            if itype in ("input_image", "image", "image_url", "input_file", "input_audio", "input_video"):
                media = _media_part(item)
                if media:
                    steps.append({"type": "user_input", "content": [media]})
                continue

            if itype in ("model_output", "output_text"):
                text = _text_of(item.get("content", item.get("text", "")))
                if text:
                    steps.append(self._model_step(text))
                continue

            if itype == "function_call":
                name = str(item.get("name") or "")
                call_id = str(item.get("call_id") or item.get("id") or "")
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    # Interactions 侧 arguments 必须是 object，字符串要转回去
                    try:
                        parsed_args = json.loads(arguments)
                    except Exception:
                        parsed_args = {}
                    arguments = parsed_args if isinstance(parsed_args, dict) else {}
                if not isinstance(arguments, dict):
                    arguments = {}
                if call_id:
                    call_names[call_id] = name
                step: dict[str, Any] = {"type": "function_call", "name": name, "arguments": arguments}
                if call_id:
                    step["id"] = call_id
                steps.append(step)
                continue

            if itype in ("function_call_output", "function_result"):
                call_id = str(item.get("call_id") or item.get("id") or "")
                output = item.get("output", item.get("result", ""))
                name = str(item.get("name") or call_names.get(call_id, ""))
                # Interactions 的 function_result 强制要求 name，缺了直接 400
                if not name:
                    name = "function"
                try:
                    result = json.loads(output) if isinstance(output, str) else output
                except Exception:
                    result = output
                step = {"type": "function_result", "call_id": call_id, "name": name, "result": result}
                steps.append(step)
                continue

            if itype in ("reasoning", "thought"):
                signature = item.get("encrypted_content") or item.get("signature")
                if signature:
                    # 伪造 signature 会被拒，只回传当轮真实拿到的
                    steps.append({"type": "thought", "signature": signature})
                continue

            # item_reference / 其它未知类型：文本兜底，带图的按 user_input 保留
            parts = _content_parts(item.get("content"))
            if parts:
                steps.append({"type": "user_input", "content": parts})

        return self._recover_missing_thought(steps)

    @staticmethod
    def _user_step(text: str) -> dict:
        return {"type": "user_input", "content": [{"type": "text", "text": text}]}

    @staticmethod
    def _model_step(text: str) -> dict:
        return {"type": "model_output", "content": [{"type": "text", "text": text}]}

    def _convert_tools(self, tools: Any) -> list[dict]:
        if not tools or not isinstance(tools, list):
            return []
        converted: list[dict] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            ttype = str(tool.get("type") or "")
            if not ttype:
                continue
            if ttype == "function":
                converted.append(
                    {
                        "type": "function",
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
                continue
            mapped = _TOOL_TYPE_MAP.get(ttype, ttype)
            if mapped in _PASSTHROUGH_TOOLS:
                entry = {"type": mapped}
                for key in ("name", "description", "parameters", "config", "server", "url", "settings"):
                    if tool.get(key) is not None:
                        entry[key] = tool[key]
                converted.append(entry)
                continue
            logger.debug("[responses] 跳过不支持的 tools type: %s", ttype)
        return converted

    # ------------------------------------------------------------ 非流式

    async def _run_non_stream(self, url: str, headers: dict, parsed: dict) -> web.Response:
        payload = dict(parsed["payload"])
        payload.pop("stream", None)
        try:
            client = self._get_http_client()
            resp = await client.post(url, headers=headers, json=payload)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}" if str(e) else repr(e)
            logger.error("[responses] 上游请求失败: %s", err_msg)
            return web.json_response({"error": {"message": f"upstream request failed: {err_msg}"}}, status=502)

        text = resp.text
        if resp.status_code != 200:
            logger.warning("[responses] 上游 %s: %s", resp.status_code, text[:400])
            return web.Response(
                body=text.encode("utf-8", "replace"),
                status=resp.status_code,
                content_type="application/json",
            )

        try:
            data = resp.json()
        except Exception as e:
            return web.json_response({"error": {"message": f"bad upstream json: {e}"}}, status=502)

        response_obj = self._build_response(
            data, parsed["model"], response_id=parsed["response_id"], created=parsed["created"]
        )
        # 缓存原始 steps，供后续轮次补 thought signature / previous_response_id
        self._cache_put(parsed["response_id"], data.get("steps") or [])
        return web.json_response(response_obj)

    # -------------------------------------------------------------- 流式

    async def _run_stream(self, req: web.Request, url: str, headers: dict, parsed: dict) -> web.StreamResponse:
        stream_url = url + ("&" if "?" in url else "?") + "alt=sse"
        payload = dict(parsed["payload"])
        payload["stream"] = True

        res = web.StreamResponse(status=200)
        res.headers["Content-Type"] = "text/event-stream; charset=utf-8"
        res.headers["Cache-Control"] = "no-cache"
        res.headers["Connection"] = "keep-alive"
        res.headers["X-Accel-Buffering"] = "no"
        await res.prepare(req)

        response_id = parsed["response_id"]
        created = parsed["created"]
        await self._sse(
            res,
            "response.created",
            {
                "type": "response.created",
                "response": self._empty_response(response_id, created, parsed["model"]),
            },
        )

        # index -> 累积槽位，最后按 index 顺序还原成 steps 数组
        slots: dict[int, dict[str, Any]] = {}
        final_interaction: dict[str, Any] = {}
        status_seen = "in_progress"
        emitted_downstream = False

        for attempt in range(2):
            client = self._get_http_client()
            try:
                async with client.stream("POST", stream_url, headers=headers, json=payload) as upstream:
                    if upstream.status_code != 200:
                        body = (await upstream.aread()).decode("utf-8", "replace")
                        if attempt == 0 and not emitted_downstream and upstream.status_code in (429, 500, 502, 503, 504):
                            logger.warning("[responses] 流式上游 %s，准备重试一次: %s", upstream.status_code, body[:200])
                            slots.clear()
                            await asyncio.sleep(0.5)
                            continue
                        logger.warning("[responses] 流式上游 %s: %s", upstream.status_code, body[:400])
                        await self._sse(
                            res,
                            "error",
                            {"type": "error", "code": "upstream_error", "message": body[:1000]},
                        )
                        await res.write_eof()
                        return res

                    async for raw_line in upstream.aiter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            event = json.loads(data_str)
                        except Exception:
                            continue

                        etype = str(event.get("event_type") or "")
                        index = event.get("index")

                        if etype == "step.start":
                            if isinstance(index, int):
                                meta = event.get("step") or {}
                                slot = {
                                    "type": str(meta.get("type") or "model_output"),
                                    "id": str(meta.get("id") or ""),
                                    "item_id": "",
                                    "name": str(meta.get("name") or ""),
                                    "text": "",
                                    "args": "",
                                    "signature": "",
                                }
                                slots[index] = slot
                                if slot["type"] == "function_call":
                                    slot["item_id"] = "fc_" + uuid.uuid4().hex[:24]
                                    emitted_downstream = True
                                    await self._sse(
                                        res,
                                        "response.output_item.added",
                                        {
                                            "type": "response.output_item.added",
                                            "output_index": index,
                                            "item": {
                                                "type": "function_call",
                                                "id": slot["item_id"],
                                                "call_id": slot["id"],
                                                "name": slot["name"],
                                                "arguments": "",
                                                "status": "in_progress",
                                            },
                                        },
                                    )
                                elif slot["type"] != "thought":
                                    # 文本槽位先占个 output_item.added，done 时再补正文
                                    slot["item_id"] = f"msg_{index}"
                                    emitted_downstream = True
                                    await self._sse(
                                        res,
                                        "response.output_item.added",
                                        {
                                            "type": "response.output_item.added",
                                            "output_index": index,
                                            "item": {
                                                "type": "message",
                                                "id": slot["item_id"],
                                                "role": "assistant",
                                                "status": "in_progress",
                                                "content": [],
                                            },
                                        },
                                    )
                            continue

                        if etype == "step.delta":
                            if not isinstance(index, int):
                                continue
                            delta = event.get("delta") or {}
                            dtype = str(delta.get("type") or "")
                            slot = slots.setdefault(
                                index,
                                {"type": "model_output", "id": "", "item_id": f"msg_{index}", "name": "", "text": "", "args": "", "signature": ""},
                            )
                            if dtype == "thought_signature":
                                slot["type"] = "thought"
                                slot["signature"] = str(delta.get("signature") or "")
                                continue
                            if dtype == "arguments_delta":
                                chunk = str(delta.get("arguments") or "")
                                slot["type"] = "function_call"
                                slot["args"] += chunk
                                emitted_downstream = True
                                await self._sse(
                                    res,
                                    "response.function_call_arguments.delta",
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": slot["item_id"],
                                        "output_index": index,
                                        "delta": chunk,
                                    },
                                )
                                continue
                            if dtype == "text":
                                chunk = str(delta.get("text") or "")
                                if not chunk:
                                    continue
                                slot["text"] += chunk
                                emitted_downstream = True
                                await self._sse(
                                    res,
                                    "response.output_text.delta",
                                    {
                                        "type": "response.output_text.delta",
                                        "item_id": slot["item_id"],
                                        "output_index": index,
                                        "content_index": 0,
                                        "delta": chunk,
                                    },
                                )
                            continue

                        if etype == "step.stop":
                            if isinstance(index, int):
                                slot = slots.get(index)
                                if slot and slot["type"] == "function_call":
                                    emitted_downstream = True
                                    await self._sse(
                                        res,
                                        "response.function_call_arguments.done",
                                        {
                                            "type": "response.function_call_arguments.done",
                                            "item_id": slot["item_id"],
                                            "output_index": index,
                                            "arguments": slot["args"],
                                        },
                                    )
                            continue

                        if etype in ("interaction.completed", "interaction.failed", "interaction.cancelled"):
                            interaction = event.get("interaction") or {}
                            if interaction:
                                final_interaction = interaction
                            status_seen = str(interaction.get("status") or status_seen)
                            continue

                        if etype == "interaction.status_update":
                            status_seen = str(event.get("status") or status_seen)
                            continue
                break

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}" if str(e) else repr(e)
                if attempt == 0 and not emitted_downstream:
                    logger.warning("[responses] 流式连接中断(%s)，尚未下发分片，自动重试一次", err_msg)
                    slots.clear()
                    await asyncio.sleep(0.5)
                    continue
                logger.error("[responses] 流式转发异常: %s", err_msg)
                await self._sse(res, "error", {"type": "error", "code": "stream_failed", "message": err_msg})
                await res.write_eof()
                return res

        if not final_interaction:
            final_interaction = {"status": status_seen, "steps": [], "usage": {}}
        # 流式的 interaction.completed 不带 steps，必须靠累积槽位还原
        final_interaction.setdefault("steps", [])
        if not final_interaction["steps"]:
            final_interaction["steps"] = self._steps_from_slots(slots)
        if not final_interaction.get("usage"):
            final_interaction["usage"] = {}

        response_obj = self._build_response(
            final_interaction,
            parsed["model"],
            response_id=response_id,
            created=created,
        )
        self._cache_put(parsed["response_id"], final_interaction.get("steps") or [])
        # 流式的 item_id 用的是槽位里生成的随机 id，回填到最终对象保持一致
        slot_item_ids = {s.get("id"): s.get("item_id") for s in slots.values() if s.get("id") and s.get("item_id")}
        for item in response_obj.get("output", []):
            if item.get("type") == "function_call":
                item["id"] = slot_item_ids.get(item.get("call_id"), item.get("id"))
        # 按最终 output 补 output_item.done，保证流式语义闭合
        for out_index, item in enumerate(response_obj.get("output", [])):
            await self._sse(
                res,
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": out_index,
                    "item": item,
                },
            )
        await self._sse(res, "response.completed", {"type": "response.completed", "response": response_obj})
        await res.write(b"data: [DONE]\n\n")
        await res.write_eof()
        return res

    @staticmethod
    def _steps_from_slots(slots: dict[int, dict[str, Any]]) -> list[dict]:
        """把流式累积的槽位还原成 Interactions steps 数组。"""
        steps: list[dict] = []
        for index in sorted(slots):
            slot = slots[index]
            stype = slot.get("type")
            if stype == "thought":
                step = {"type": "thought"}
                if slot.get("signature"):
                    step["signature"] = slot["signature"]
                steps.append(step)
            elif stype == "function_call":
                step = {"type": "function_call", "name": slot.get("name") or ""}
                if slot.get("id"):
                    step["id"] = slot["id"]
                try:
                    args = json.loads(slot.get("args") or "{}")
                except Exception:
                    args = {}
                step["arguments"] = args if isinstance(args, dict) else {}
                steps.append(step)
            elif slot.get("text"):
                steps.append(
                    {"type": "model_output", "content": [{"type": "text", "text": slot["text"]}]}
                )
        return steps

    # -------------------------------------------------- Interaction -> Response

    def _build_response(
        self,
        interaction: dict,
        model: str,
        *,
        response_id: str | None = None,
        created: int | None = None,
    ) -> dict:
        steps = interaction.get("steps") or []
        output: list[dict] = []
        text_parts: list[str] = []

        for step in steps:
            if not isinstance(step, dict):
                continue
            stype = step.get("type")

            if stype == "model_output":
                text = _text_of(step.get("content"))
                if not text:
                    continue
                text_parts.append(text)
                output.append(
                    {
                        "type": "message",
                        "id": "msg_" + uuid.uuid4().hex[:24],
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                )
                continue

            if stype == "thought":
                signature = step.get("signature")
                output.append(
                    {
                        "type": "reasoning",
                        "id": "rs_" + uuid.uuid4().hex[:24],
                        "summary": [],
                        "encrypted_content": signature,
                    }
                )
                continue

            if stype == "function_call":
                arguments = step.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = _stringify(arguments)
                output.append(
                    {
                        "type": "function_call",
                        "id": "fc_" + uuid.uuid4().hex[:24],
                        "call_id": str(step.get("id") or ""),
                        "name": str(step.get("name") or ""),
                        "arguments": arguments,
                        "status": "completed",
                    }
                )
                continue

            if stype == "function_result":
                output.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(step.get("call_id") or ""),
                        "output": _stringify(step.get("result", "")),
                    }
                )
                continue

        usage_raw = interaction.get("usage") or {}
        input_tokens = int(usage_raw.get("total_input_tokens") or 0)
        output_tokens = int(usage_raw.get("total_output_tokens") or 0)
        thought_tokens = int(usage_raw.get("total_thought_tokens") or 0)
        cached_tokens = int(
            usage_raw.get("total_cached_tokens")
            or usage_raw.get("cached_tokens")
            or sum(
                int(m.get("tokens") or 0)
                for m in (usage_raw.get("cached_tokens_by_modality") or [])
                if isinstance(m, dict)
            )
            or 0
        )

        status = str(interaction.get("status") or "completed")
        if status in ("in_progress", "requires_action"):
            # requires_action = 还有 tool call 要客户端执行，对 Responses 语义就是 completed
            status = "completed"

        response: dict[str, Any] = {
            "id": response_id or ("resp_" + uuid.uuid4().hex),
            "object": "response",
            "created_at": created or int(time.time()),
            "status": status,
            "model": model,
            "output": output,
            "output_text": "".join(text_parts),
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": cached_tokens},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": thought_tokens},
                "total_tokens": input_tokens + output_tokens,
            },
            "error": None,
            "incomplete_details": None,
            "metadata": {},
        }
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        return response

    @staticmethod
    def _empty_response(response_id: str, created: int, model: str) -> dict:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "model": model,
            "output": [],
            "usage": None,
            "error": None,
            "incomplete_details": None,
            "metadata": {},
        }

    # ---------------------------------------------------------------- SSE

    @staticmethod
    async def _sse(res: web.StreamResponse, event: str, data: dict) -> None:
        frame = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        await res.write(frame.encode("utf-8"))
