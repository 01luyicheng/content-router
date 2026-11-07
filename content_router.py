#!/usr/bin/env python3
"""
OpenClaw 智能网关 v2
- 自动检测多模态输入（图片/视频/音频）
- 路由到 openclaw-vision 或 openclaw-main
- 保持 OpenAI API 格式完全兼容
- 对外暴露统一接口 http://localhost:4001/v1
- 修复：UUID替代计数器、连接复用、错误分类、请求大小限制
"""

import json
import re
import uuid
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from socket import timeout as SocketTimeout
import http.client

# 尝试使用 requests 库进行连接复用
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("警告: 未安装 requests 库，使用 urllib (性能较差)")
    print("建议: pip install requests")

def _env_int(name, default):
    """读取整型环境变量；未设置时返回默认值，非法值直接报错退出。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} 必须是整数，当前为 {raw!r}")
    if value <= 0:
        raise SystemExit(f"{name} 必须为正整数，当前为 {value}")
    return value


# 配置（均可用环境变量覆盖）
LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://localhost:4000").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "openclaw-main")
VISION_MODEL = os.environ.get("VISION_MODEL", "openclaw-vision")

# 预编译正则表达式 - 用于多模态检测
# 限制 URL 长度防止 ReDoS 攻击
MULTIMODAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'data:image/[a-zA-Z0-9]+;base64,',
    r'data:video/[a-zA-Z0-9]+;base64,',
    r'data:audio/[a-zA-Z0-9]+;base64,',
    r'https?://[^\s"\']{1,2000}\.(?:jpg|jpeg|png|gif|webp|bmp|svg|ico)',
    r'https?://[^\s"\']{1,2000}\.(?:mp4|avi|mov|webm|mkv|flv|wmv)',
    r'https?://[^\s"\']{1,2000}\.(?:mp3|wav|ogg|m4a|aac|flac|wma|opus)',
]]

# 日志清理正则 - 过滤控制字符防止日志注入
LOG_SANITIZE_PATTERN = re.compile(r'[\x00-\x1F\x7F]')

# 安全限制
MAX_REQUEST_SIZE = _env_int("MAX_REQUEST_SIZE", 500 * 1024 * 1024)   # 500MB 最大请求大小
MAX_BODY_SIZE = _env_int("MAX_BODY_SIZE", 500 * 1024 * 1024)         # 500MB 最大响应大小

# 日志配置：优先使用 LOG_DIR，不可写时退回脚本同级的 logs/ 目录
def _resolve_log_dir():
    """返回一个可写的日志目录；全部候选都不可写时返回 None（仅保留控制台日志）。"""
    candidates = []
    if os.environ.get("LOG_DIR"):
        candidates.append(os.environ["LOG_DIR"])
    else:
        candidates.append("/var/log/openclaw")
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".write-probe")
            with open(probe, "w", encoding="utf-8"):
                pass
            os.remove(probe)
            return candidate
        except OSError:
            continue
    return None


LOG_DIR = _resolve_log_dir()
LOG_FILE = os.path.join(LOG_DIR, "gateway.log") if LOG_DIR else None

# 设置日志 - 使用轮转
logger = logging.getLogger('openclaw-gateway')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# 文件日志 - 轮转，最大100MB，保留5个备份
if LOG_FILE:
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
else:
    print("警告: 没有可写的日志目录，仅输出控制台日志（可用 LOG_DIR 指定目录）")

# 控制台日志
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# HTTP 会话池（连接复用）- 使用线程本地存储确保线程安全
_thread_local = threading.local()

def get_session():
    """获取线程本地的 requests Session"""
    if not HAS_REQUESTS:
        return None
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = requests.Session()
        # 配置重试策略
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=5,
            pool_maxsize=10
        )
        _thread_local.session.mount("http://", adapter)
        _thread_local.session.mount("https://", adapter)
    return _thread_local.session

# 为了保持向后兼容，保留 session 变量（实际使用 get_session()）
session = get_session() if HAS_REQUESTS else None


def detect_multimodal(data):
    """检测请求是否包含多模态内容"""
    try:
        messages = data.get('messages', [])
        
        for msg in messages:
            content = msg.get('content', '')
            
            # 处理数组格式（OpenAI 多模态标准格式）
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get('type', '')
                        if item_type in ['image_url', 'image', 'video', 'audio', 'file']:
                            return True
                        if item_type == 'text':
                            text = item.get('text', '')
                            for pattern in MULTIMODAL_PATTERNS:
                                if pattern.search(text):
                                    return True
            
            # 处理字符串格式 - 使用预编译的正则
            elif isinstance(content, str):
                for pattern in MULTIMODAL_PATTERNS:
                    if pattern.search(content):
                        return True
        
        return False
    except Exception:
        return False


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """支持并发的多线程服务器"""
    daemon_threads = True  # 确保线程随主进程退出
    allow_reuse_address = True


class OpenClawHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """简化日志到控制台"""
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")
    
    def _get_request_id(self):
        """生成唯一请求ID（使用 UUID，避免并发问题）"""
        return str(uuid.uuid4())[:16]
    
    def _sanitize_for_log(self, text):
        """清理日志内容，防止注入攻击"""
        if not isinstance(text, str):
            text = str(text)
        # 替换控制字符为空格，限制长度
        sanitized = LOG_SANITIZE_PATTERN.sub(' ', text)
        return sanitized[:200]  # 限制长度
    
    def _summarize_content(self, content):
        """摘要内容用于日志"""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get('type', 'unknown')
                    if item_type == 'text':
                        text = item.get('text', '')
                        parts.append(f"[text:{text[:50]}...]" if len(text) > 50 else f"[text:{text}]")
                    elif item_type == 'image_url':
                        url = item.get('image_url', {}).get('url', '')
                        if url.startswith('data:'):
                            parts.append(f"[image_url:base64({len(url)}ch)]")
                        else:
                            parts.append(f"[image_url:{url[:50]}...]")
                    else:
                        parts.append(f"[{item_type}]")
            return self._sanitize_for_log("|".join(parts))
        elif isinstance(content, str):
            if 'data:' in content and ';base64,' in content:
                return self._sanitize_for_log(f"[content_with_base64:{content[:80]}...({len(content)}ch)]")
            return self._sanitize_for_log(content[:100] + "..." if len(content) > 100 else content)
        return self._sanitize_for_log(str(content)[:100])
    
    def _log_request(self, req_id, data, route_decision):
        """记录请求日志"""
        try:
            messages = data.get('messages', [])
            msg_summary = []
            for i, msg in enumerate(messages[:3]):
                role = msg.get('role', 'unknown')
                content = self._summarize_content(msg.get('content', ''))
                msg_summary.append(f"{role}:{content}")
            
            if len(messages) > 3:
                msg_summary.append(f"...({len(messages)-3} more)")
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'request_id': req_id,
                'client_ip': self.client_address[0],
                'method': self.command,
                'path': self.path,
                'original_model': data.get('model', 'default'),
                'route_decision': route_decision,
                'messages_summary': ' | '.join(msg_summary),
                'total_messages': len(messages)
            }
            
            logger.info(f"[REQUEST] {json.dumps(log_entry, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"[REQUEST_LOG_ERROR] {req_id}: {e}")
    
    def _log_response(self, req_id, status, response_data, duration_ms, target_model, error_msg=None):
        """记录响应日志"""
        try:
            actual_model = 'unknown'
            usage = None
            
            if response_data and status < 400:
                try:
                    if isinstance(response_data, bytes):
                        resp_json = json.loads(response_data.decode('utf-8'))
                    else:
                        resp_json = response_data
                    
                    actual_model = resp_json.get('model', 'unknown')
                    usage = resp_json.get('usage', {})
                except:
                    pass
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'request_id': req_id,
                'status': status,
                'target_model': target_model,
                'actual_model': actual_model,
                'duration_ms': duration_ms,
                'error': error_msg,
                'usage': usage
            }
            
            if error_msg or status >= 400:
                logger.warning(f"[RESPONSE] {json.dumps(log_entry, ensure_ascii=False)}")
            else:
                logger.info(f"[RESPONSE] {json.dumps(log_entry, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"[RESPONSE_LOG_ERROR] {req_id}: {e}")
    
    def _send_response(self, data, status_code=200):
        """发送 JSON 响应"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        if isinstance(data, bytes):
            self.wfile.write(data)
        else:
            self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_error(self, status_code, error_type, message, details=None):
        """发送标准化的错误响应"""
        error_data = {
            'error': {
                'type': error_type,
                'message': message,
                'details': details or {}
            }
        }
        self._send_response(error_data, status_code)
    
    def _proxy_with_requests(self, method, path, body=None, headers=None, stream=False):
        """使用 requests 库代理请求（支持连接复用）"""
        if not HAS_REQUESTS:
            return None, None  # 回退到 urllib
        
        url = f"{LITELLM_BASE}{path}"
        
        try:
            # 获取线程本地的 session
            _session = get_session()
            if _session is None:
                return None, None
            
            if stream:
                # 流式传输
                response = _session.request(
                    method, url,
                    data=body.encode('utf-8') if body else None,
                    headers=headers,
                    stream=True,
                    timeout=(10, 300)  # (连接超时, 读取超时)
                )
                
                self.send_response(response.status_code)
                for header, value in response.headers.items():
                    if header.lower() not in ['transfer-encoding', 'content-length']:
                        self.send_header(header, value)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                chunk_count = 0
                try:
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                            chunk_count += 1
                except (BrokenPipeError, ConnectionResetError):
                    # 客户端断开连接
                    logger.info(f"Client disconnected during streaming")
                    return 'stream', -chunk_count  # 负数表示客户端断开
                
                return 'stream', chunk_count
            else:
                # 非流式传输
                response = _session.request(
                    method, url,
                    data=body.encode('utf-8') if body else None,
                    headers=headers,
                    timeout=(10, 300)
                )
                
                return response.status_code, response.content
                
        except requests.exceptions.Timeout as e:
            return 'timeout', str(e)
        except requests.exceptions.ConnectionError as e:
            return 'connection_error', str(e)
        except requests.exceptions.RequestException as e:
            return 'request_error', str(e)
    
    def _proxy_to_litellm(self, method, path, body=None, headers=None):
        """代理请求到 LiteLLM（非流式，urllib 回退）"""
        url = f"{LITELLM_BASE}{path}"
        
        req_headers = {}
        if headers:
            req_headers.update(headers)
        
        try:
            req = urllib.request.Request(
                url,
                data=body.encode('utf-8') if body else None,
                headers=req_headers,
                method=method
            )
            
            with urllib.request.urlopen(req, timeout=300) as response:
                return response.status, response.read()
                
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except urllib.error.URLError as e:
            if isinstance(e.reason, SocketTimeout):
                return 'timeout', json.dumps({'error': 'Connection timeout'}).encode()
            return 'connection_error', json.dumps({'error': str(e.reason)}).encode()
        except SocketTimeout:
            return 'timeout', json.dumps({'error': 'Request timeout'}).encode()
        except (http.client.HTTPException, OSError) as e:
            # 网络层错误
            logger.error(f"Network error: {e}")
            return 'connection_error', json.dumps({'error': 'Network communication error'}).encode()
        except Exception as e:
            return 'error', json.dumps({'error': str(e)}).encode()
    
    def _proxy_streaming_to_litellm(self, path, body=None, headers=None):
        """代理流式请求到 LiteLLM（SSE 流式传输，urllib 回退）"""
        url = f"{LITELLM_BASE}{path}"
        
        req_headers = {}
        if headers:
            req_headers.update(headers)
        
        try:
            req = urllib.request.Request(
                url,
                data=body.encode('utf-8') if body else None,
                headers=req_headers,
                method='POST'
            )
            
            # 使用上下文管理器确保响应对象正确关闭
            with urllib.request.urlopen(req, timeout=300) as response:
                self.send_response(response.status)
                for header, value in response.headers.items():
                    if header.lower() not in ['transfer-encoding', 'content-length']:
                        self.send_header(header, value)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                chunk_count = 0
                total_size = 0
                truncated = False
                client_disconnected = False
                
                try:
                    while True:
                        chunk = response.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        chunk_count += 1
                        total_size += len(chunk)
                        
                        # 防止响应过大
                        if total_size > MAX_BODY_SIZE:
                            logger.warning(f"响应超过最大限制 {MAX_BODY_SIZE}，截断")
                            truncated = True
                            # 发送 SSE 结束标记
                            try:
                                self.wfile.write(b'data: [TRUNCATED]\n\n')
                                self.wfile.flush()
                            except:
                                pass
                            break
                except (BrokenPipeError, ConnectionResetError):
                    # 客户端断开连接
                    logger.info(f"Client disconnected during streaming")
                    client_disconnected = True
                
                if client_disconnected:
                    return -chunk_count if chunk_count > 0 else -1  # 负数表示客户端断开
                if truncated:
                    return -chunk_count  # 负数表示被截断
                return chunk_count
            
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(e.read())
            return -1
        except (http.client.HTTPException, OSError) as e:
            # 网络层错误
            logger.error(f"Network error in streaming: {e}")
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Network communication error'}).encode())
            return -1
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
            return -1
    
    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_POST(self):
        """处理 POST 请求（支持流式传输）"""
        start_time = time.time()
        req_id = self._get_request_id()
        target_model = None
        error_msg = None
        
        # 只处理 chat completions
        if self.path not in ['/v1/chat/completions', '/v1/completions']:
            self._send_error(404, 'not_found', f'Path {self.path} not found')
            return
        
        try:
            # 检查 Content-Length
            content_length_str = self.headers.get('Content-Length', '0')
            try:
                content_length = int(content_length_str)
            except ValueError:
                self._send_error(400, 'bad_request', 'Invalid Content-Length header')
                return
            
            # 检查请求大小 - 必须在读取之前检查，防止内存耗尽攻击
            if content_length > MAX_REQUEST_SIZE:
                self._send_error(413, 'payload_too_large', 
                    f'Request body too large. Max: {MAX_REQUEST_SIZE / 1024 / 1024:.0f}MB, Got: {content_length / 1024 / 1024:.1f}MB')
                return
            
            if content_length == 0:
                self._send_error(400, 'bad_request', 'Empty request body')
                return
            
            # 读取请求体（设置超时保护）
            import socket
            original_timeout = self.connection.gettimeout()
            self.connection.settimeout(30)  # 读取 body 最多 30 秒
            try:
                body = self.rfile.read(content_length).decode('utf-8')
            finally:
                self.connection.settimeout(original_timeout)
            
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                duration_ms = int((time.time() - start_time) * 1000)
                self._log_response(req_id, 400, None, duration_ms, None, f'JSON parse error: {e}')
                self._send_error(400, 'json_parse_error', 'Invalid JSON in request body', {'details': str(e)})
                return
            
            # 检测是否为流式请求
            is_streaming = data.get('stream', False)
            
            # 获取原始模型
            original_model = data.get('model', DEFAULT_MODEL)
            route_decision = {
                'original_model': original_model,
                'streaming': is_streaming
            }
            
            # 只处理 openclaw-auto 或默认模型
            if original_model in ['openclaw-auto', 'auto', DEFAULT_MODEL]:
                # 检测是否为多模态
                if detect_multimodal(data):
                    data['model'] = VISION_MODEL
                    target_model = VISION_MODEL
                    route_decision['detected'] = 'multimodal'
                    route_decision['routed_to'] = VISION_MODEL
                    stream_info = "(流式)" if is_streaming else ""
                    print(f"  [{req_id}] 路由: 多模态 → {VISION_MODEL} {stream_info}")
                else:
                    data['model'] = DEFAULT_MODEL
                    target_model = DEFAULT_MODEL
                    route_decision['detected'] = 'text'
                    route_decision['routed_to'] = DEFAULT_MODEL
                    stream_info = "(流式)" if is_streaming else ""
                    print(f"  [{req_id}] 路由: 纯文本 → {DEFAULT_MODEL} {stream_info}")
                
                # 重新编码请求体
                body = json.dumps(data)
            else:
                target_model = original_model
                route_decision['detected'] = 'passthrough'
                route_decision['routed_to'] = original_model
            
            # 记录请求日志
            self._log_request(req_id, data, route_decision)
            
            # 转发到 LiteLLM
            auth_header = self.headers.get('Authorization', '')
            headers = {
                'Content-Type': 'application/json',
                'Authorization': auth_header
            }
            
            # 根据是否流式选择不同的代理方式
            if is_streaming:
                # 优先使用 requests 进行流式传输
                result, result_data = self._proxy_with_requests('POST', self.path, body, headers, stream=True)
                
                if result == 'stream':
                    duration_ms = int((time.time() - start_time) * 1000)
                    is_truncated = result_data < 0 if isinstance(result_data, int) else False
                    abs_chunks = abs(result_data) if isinstance(result_data, int) else result_data
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'request_id': req_id,
                        'type': 'streaming',
                        'target_model': target_model,
                        'duration_ms': duration_ms,
                        'chunks': abs_chunks,
                        'truncated': is_truncated
                    }
                    trunc_info = " [截断]" if is_truncated else ""
                    logger.info(f"[RESPONSE] {json.dumps(log_entry)}")
                    print(f"  [{req_id}] 流式响应完成{trunc_info}: {abs_chunks} chunks | {duration_ms}ms")
                elif result == 'timeout':
                    duration_ms = int((time.time() - start_time) * 1000)
                    error_msg = f'Upstream timeout: {result_data}'
                    self._log_response(req_id, 504, None, duration_ms, target_model, error_msg)
                    self._send_error(504, 'upstream_timeout', 'Upstream server timeout', {'details': result_data})
                elif result == 'connection_error':
                    duration_ms = int((time.time() - start_time) * 1000)
                    error_msg = f'Connection error: {result_data}'
                    self._log_response(req_id, 502, None, duration_ms, target_model, error_msg)
                    self._send_error(502, 'connection_error', 'Cannot connect to upstream server', {'details': result_data})
                else:
                    # 回退到 urllib
                    chunk_count = self._proxy_streaming_to_litellm(self.path, body, headers)
                    duration_ms = int((time.time() - start_time) * 1000)
                    
                    if chunk_count > 0:
                        log_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'request_id': req_id,
                            'type': 'streaming',
                            'target_model': target_model,
                            'duration_ms': duration_ms,
                            'chunks': chunk_count
                        }
                        logger.info(f"[RESPONSE] {json.dumps(log_entry)}")
                        print(f"  [{req_id}] 流式响应完成: {chunk_count} chunks | {duration_ms}ms")
            else:
                # 非流式传输 - 优先使用 requests
                status, response_data = self._proxy_with_requests('POST', self.path, body, headers)
                
                # 如果 requests 失败，回退到 urllib
                if status is None:
                    status, response_data = self._proxy_to_litellm('POST', self.path, body, headers)
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                # 处理不同类型的错误
                if status == 'timeout':
                    error_msg = f'Upstream timeout: {response_data}'
                    self._log_response(req_id, 504, None, duration_ms, target_model, error_msg)
                    self._send_error(504, 'upstream_timeout', 'Upstream server timeout', {'details': response_data})
                    return
                elif status == 'connection_error':
                    error_msg = f'Connection error: {response_data}'
                    self._log_response(req_id, 502, None, duration_ms, target_model, error_msg)
                    self._send_error(502, 'connection_error', 'Cannot connect to upstream server', {'details': response_data})
                    return
                elif status == 'request_error':
                    error_msg = f'Request error: {response_data}'
                    self._log_response(req_id, 500, None, duration_ms, target_model, error_msg)
                    self._send_error(500, 'request_error', 'Request failed', {'details': response_data})
                    return
                elif isinstance(status, int) and status >= 400:
                    error_msg = f'Upstream error: {status}'
                
                # 记录响应日志
                self._log_response(req_id, status, response_data, duration_ms, target_model, error_msg)
                
                # 解析响应，输出简要信息到控制台
                if isinstance(status, int) and status < 400:
                    try:
                        resp_json = json.loads(response_data.decode('utf-8'))
                        actual_model = resp_json.get('model', 'unknown')
                        if actual_model in [DEFAULT_MODEL, VISION_MODEL]:
                            usage = resp_json.get('usage', {})
                            tokens = usage.get('total_tokens', 0)
                            print(f"  [{req_id}] 响应: {actual_model} | {status} | {duration_ms}ms | {tokens}tokens")
                    except:
                        print(f"  [{req_id}] 响应: status={status} | {duration_ms}ms")
                else:
                    print(f"  [{req_id}] 错误: status={status} | {duration_ms}ms")
                
                if isinstance(status, int):
                    self._send_response(response_data, status)
                else:
                    self._send_error(500, 'internal_error', 'Invalid response from upstream')
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f'Internal error: {e}'
            self._log_response(req_id, 500, None, duration_ms, target_model, error_msg)
            logger.error(f"[{req_id}] Exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 生产环境不返回详细错误信息，防止信息泄露
            self._send_error(500, 'internal_error', 'Internal server error')
    
    def do_GET(self):
        """处理 GET 请求（健康检查等）"""
        if self.path == '/health':
            # 网关自身健康检查（使用 socket 检测 LiteLLM 端口，避免 401）
            litellm_healthy = False
            litellm_error = None
            sock = None
            upstream = urllib.parse.urlsplit(LITELLM_BASE)
            upstream_host = upstream.hostname or 'localhost'
            upstream_port = upstream.port or (443 if upstream.scheme == 'https' else 80)
            
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect((upstream_host, upstream_port))
                litellm_healthy = True
            except socket.timeout:
                litellm_error = f"Connection to LiteLLM {upstream_host}:{upstream_port} timed out"
            except ConnectionRefusedError:
                litellm_error = f"Connection to LiteLLM {upstream_host}:{upstream_port} refused"
            except Exception as e:
                litellm_error = str(e)
            finally:
                if sock:
                    try:
                        sock.close()
                    except:
                        pass
            
            health_data = {
                'status': 'healthy' if litellm_healthy else 'degraded',
                'litellm_connected': litellm_healthy,
                'litellm_base': LITELLM_BASE,
                'timestamp': datetime.now().isoformat(),
                'features': {
                    'http_connection_reuse': HAS_REQUESTS,
                    'max_request_size_mb': MAX_REQUEST_SIZE / 1024 / 1024,
                }
            }
            
            if litellm_error:
                health_data['litellm_error'] = litellm_error
            
            self._send_response(health_data, 200 if litellm_healthy else 503)
                
        elif self.path == '/v1/models':
            # 返回支持的模型列表
            models = {
                'object': 'list',
                'data': [
                    {'id': 'openclaw-auto', 'object': 'model'},
                    {'id': 'openclaw-main', 'object': 'model'},
                    {'id': 'openclaw-vision', 'object': 'model'},
                ]
            }
            self._send_response(models)
        else:
            # 其他请求转发到 LiteLLM
            headers = {'Authorization': self.headers.get('Authorization', '')}
            
            if HAS_REQUESTS:
                status, response_data = self._proxy_with_requests('GET', self.path, headers=headers)
                if status is None:
                    status, response_data = self._proxy_to_litellm('GET', self.path, headers=headers)
            else:
                status, response_data = self._proxy_to_litellm('GET', self.path, headers=headers)
            
            if isinstance(status, int):
                self._send_response(response_data, status)
            else:
                self._send_error(502, 'bad_gateway', 'Cannot connect to upstream server')


def run_server(port=4001):
    """启动服务"""
    server = ThreadedHTTPServer(('', port), OpenClawHandler)
    print(f"=" * 60)
    print(f"OpenClaw 智能网关 v2 已启动")
    print(f"地址: http://localhost:{port}/v1")
    print(f"健康检查: http://localhost:{port}/health")
    print(f"统一模型名: openclaw-auto")
    print(f"  - 纯文本 → openclaw-main")
    print(f"  - 多模态 → openclaw-vision")
    if HAS_REQUESTS:
        print(f"  - HTTP连接复用: 已启用")
    else:
        print(f"  - HTTP连接复用: 未启用 (pip install requests)")
    print(f"  - 最大请求大小: {MAX_REQUEST_SIZE / 1024 / 1024:.0f}MB")
    print(f"  - 多线程支持: 已启用")
    print(f"=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n网关已停止")
    finally:
        # 清理工作（线程本地存储的 session 会自动清理）
        print("网关资源已释放")


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4001
    run_server(port)
