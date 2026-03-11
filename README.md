# OpenClaw Gateway

智能API网关，自动路由多模态请求。

## 功能

- 自动检测图片/视频/音频内容
- 智能路由：纯文本 → openclaw-main，多模态 → openclaw-vision
- 支持流式传输
- HTTP连接复用
- 多线程并发处理

## 安装

```bash
pip install requests
```

## 配置

设置环境变量：
```bash
export LITELLM_BASE=http://localhost:4000
export MAX_REQUEST_SIZE=524288000  # 500MB
```

## 启动

```bash
python openclaw_gateway.py 4001
```

## API

- `POST /v1/chat/completions` - 聊天补全
- `GET /health` - 健康检查
- `GET /v1/models` - 模型列表
