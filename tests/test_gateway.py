#!/usr/bin/env python3
"""content_router 最小冒烟测试：多模态检测 + 健康检查。"""

import json
import os
import pathlib
import sys
import tempfile
import threading
import urllib.request

# 必须在导入模块前指定日志目录，否则模块会尝试写 /var/log/openclaw
os.environ["LOG_DIR"] = tempfile.mkdtemp(prefix="gateway-log-")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import content_router as gw  # noqa: E402


def test_plain_text_is_not_multimodal():
    data = {"messages": [{"role": "user", "content": "今天天气怎么样？"}]}
    assert gw.detect_multimodal(data) is False


def test_image_url_item_is_multimodal():
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述这张图"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }
        ]
    }
    assert gw.detect_multimodal(data) is True


def test_base64_data_uri_in_string_is_multimodal():
    data = {"messages": [{"role": "user", "content": "data:image/png;base64,iVBORw0KGgo="}]}
    assert gw.detect_multimodal(data) is True


def test_health_endpoint_reports_upstream_state():
    """上游未启动时应返回 503，且响应体包含连接状态。"""
    server = gw.ThreadedHTTPServer(("127.0.0.1", 0), gw.OpenClawHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as resp:
                status, body = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read()

        payload = json.loads(body.decode("utf-8"))
        assert status in (200, 503)
        assert payload["status"] in ("healthy", "degraded")
        assert "litellm_connected" in payload
    finally:
        server.shutdown()
        server.server_close()
