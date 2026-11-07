# OpenClaw Gateway

轻量级 OpenAI 兼容 API 网关：检测请求里的多模态内容，把纯文本转发到 `openclaw-main`，把图片/视频/音频转发到 `openclaw-vision`。

## 功能

- 自动检测图片 / 视频 / 音频（`image_url` 结构消息、`data:` base64、常见媒体 URL）
- 模型路由：纯文本 → `openclaw-main`，多模态 → `openclaw-vision`，其他模型名透传
- 支持 SSE 流式响应，并在超过上限时截断
- `requests` 可用时复用连接池，未安装时自动回退到 `urllib`
- 多线程并发处理，支持 CORS 预检
- 请求体大小限制、日志控制字符过滤（防日志注入）

## 安装

```bash
pip install requests   # 可选，缺失时回退 urllib，性能较差
```

## 配置

全部通过环境变量配置，未设置时使用括号内默认值：

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `LITELLM_BASE` | 上游 LiteLLM 地址 | `http://localhost:4000` |
| `DEFAULT_MODEL` | 纯文本路由目标 | `openclaw-main` |
| `VISION_MODEL` | 多模态路由目标 | `openclaw-vision` |
| `MAX_REQUEST_SIZE` | 请求体上限（字节） | `524288000`（500MB） |
| `MAX_BODY_SIZE` | 响应体上限（字节） | `524288000`（500MB） |
| `LOG_DIR` | 日志目录 | `/var/log/openclaw`，不可写时退回 `./logs` |

示例：

```bash
export LITELLM_BASE=http://localhost:4000
export LOG_DIR=$PWD/logs
```

## 启动

```bash
python3 content_router.py 4001
```

端口默认 `4001`，可以用第一个命令行参数覆盖。启动后可访问：

- 网关：<http://localhost:4001/v1>
- 健康检查：<http://localhost:4001/health>

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | 聊天补全，自动路由 |
| `POST` | `/v1/completions` | 文本补全，自动路由 |
| `GET` | `/health` | 健康检查；上游不可达时返回 `503` + `status: degraded` |
| `GET` | `/v1/models` | 返回 `openclaw-auto` / `openclaw-main` / `openclaw-vision` |

用 `openclaw-auto`（或 `auto`）作为模型名即可启用自动路由：

```bash
curl http://localhost:4001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"openclaw-auto","messages":[{"role":"user","content":"hello"}]}'
```

## 测试

```bash
python3 -m pytest -q
```

覆盖多模态检测的三种输入形态，以及 `/health` 在上下游状态下的响应结构。
