# Vertex Bridge

AstrBot 本地 Vertex AI 桥。监听本机端口，把 OpenAI / Gemini 原生请求转到 Vertex。

灵感来自把 Gemini 提供商指到本地兼容端点的用法。实际请求走 Vertex REST（`httpx` + 服务账号）。

## 能力

- `GET /health`
- `GET /v1/models`（实时拉取，失败回退内置兜底列表）、`POST /v1/chat/completions`、`POST /v1/embeddings`
- `* /v1beta/{tail}` Gemini 原生转发（`generateContent` 等）
- `POST /v1/responses` OpenAI Responses 协议（默认关闭，见下）
- 服务账号只放 `data/plugin_data/astrbot_plugin_vertex_bridge/`，不进插件源码目录

## 配置

WebUI 上传服务账号 JSON（推荐），或一次性粘贴 JSON。保存后私钥写入：

`AstrBot/data/plugin_data/astrbot_plugin_vertex_bridge/vertex-sa.json`

框架配置里的 `sa_content` 会被清空。桥本身不加访问密钥。

默认端口 `18317`，区域 `us-central1`。

### Responses 端点（可选）

默认关闭。打开后新增 `POST /v1/responses`（含 `/v1/responses:stream`），走 Vertex 原生
Interactions 接口（`/v1beta1/projects/{id}/locations/global/interactions`），不是
`generateContent` 的补丁封装。

- `enable_responses`：开关。
- `responses_models`：白名单，逗号分隔。留空 = 不启用（防止拿 `/v1/models` 全表当白名单）。
- 命名带 `-preview` 会自动回退到不带后缀的模型名再重试一次。

支持：流式 / 非流式、多轮、`previous_response_id`（本地 steps 缓存回放）、function calling、
内置 `google_search` / `url_context` / `code_execution`、`input_image` / `input_file`
（data url 的 base64 或 http(s)/gs uri，转成 Interactions 的 image/document part）。
`max_output_tokens` 映射成 `incomplete`，`reasoning` 项带 `encrypted_content`
（即 thought signature）。

已知限制：`store=false`，历史必须随请求全量回传；Interactions 的 `function_call.arguments`
是 object 而不是字符串，适配层会做双向转换。

## 依赖

`httpx`、`google-auth`、`aiohttp`

## 更新日志

- `v1.4.1`：`/v1/models` 与 Gemini 原生 `models` 改为实时拉取 Vertex 的 google 模型列表（`v1beta1/publishers/google/models`，缓存 5 分钟，失败冷却 30 秒），并合并内置的 Claude 名单。拉取失败时自动切到内置兜底列表，并在最前面置顶一个占位模型 `!!!兜底列表-实时拉取失败` 作为提示，它不映射到任何真实模型，被调用会直接返回 400 与说明。同时修复 `/vmodels` 因使用错误的 v1 项目路径返回 404 的问题，现复用同一套实时拉取逻辑。
- `v1.4.0`：新增 `enable_flex` 配置项（默认开启），在调用 Vertex AI Gemini 模型时自动附带 Flex 降本推理请求头，降低调用资费。
- `v1.3.1`：修复 `/v1/responses` 端点 `usage.input_tokens_details.cached_tokens` 恒为 0 的问题；优化流式连接池保活上限、空异常信息格式化与未下发分片时的自动重试；重载插件时同步热重载 `responses_api` 子模块。
- `v1.3.0`：新增 `POST /v1/responses` OpenAI Responses 协议适配层（Interactions 原生），默认关闭，用 `enable_responses` + `responses_models` 开启。
- `v1.2.1`：新增 `gemini-4-argon` 模型目录与全局端点自动路由支持，配置新增 `global (全局端点)` 区域选项。
- `v1.2.0`：新增 Claude 5 / 4 系列模型支持与 Gemini 3.x 全局端点路由。
