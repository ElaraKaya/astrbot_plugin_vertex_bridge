# Vertex Bridge

AstrBot 本地 Vertex AI 桥。监听本机端口，把 OpenAI / Gemini 原生请求转到 Vertex。

灵感来自把 Gemini 提供商指到本地兼容端点的用法。实际请求走 Vertex REST（`httpx` + 服务账号）。

## 能力

- `GET /health`
- `GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/embeddings`
- `* /v1beta/{tail}` Gemini 原生转发（`generateContent` 等）
- 服务账号只放 `data/plugin_data/astrbot_plugin_vertex_bridge/`，不进插件源码目录

## 配置

WebUI 上传服务账号 JSON（推荐），或一次性粘贴 JSON。保存后私钥写入：

`AstrBot/data/plugin_data/astrbot_plugin_vertex_bridge/vertex-sa.json`

框架配置里的 `sa_content` 会被清空。桥本身不加访问密钥。

默认端口 `18317`，区域 `us-central1`。

## 依赖

`httpx`、`google-auth`、`aiohttp`
