# Qwen3-ASR REST API 服务

本文档介绍如何部署和调用 Qwen3-ASR 的 REST API 服务。

## 目录

- [服务端部署](#服务端部署)
- [客户端调用](#客户端调用)
- [API 接口说明](#api-接口说明)
- [使用示例](#使用示例)

---

## 服务端部署

### 环境要求

- Python 3.9+
- CUDA 环境（推荐）
- 已安装 `qwen-asr` 包

### 启动服务

```bash
# 激活 conda 环境
conda activate qwen3-asr

# 启动服务器（默认绑定 0.0.0.0:5000）
python asr_server.py

# 自定义参数启动
python asr_server.py \n    --host 0.0.0.0 \n    --port 8080 \n    --model Qwen/Qwen3-ASR-1.7B \n    --device cuda:0 \n    --dtype bfloat16 \n    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B
```

### 服务端参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 绑定的主机地址 |
| `--port` | `5000` | 绑定的端口 |
| `--model` | `Qwen/Qwen3-ASR-1.7B` | 模型名称或路径 |
| `--device` | `cuda:0` | 运行设备 |
| `--dtype` | `bfloat16` | 数据类型 (float32/float16/bfloat16) |
| `--forced-aligner` | `Qwen/Qwen3-ForcedAligner-0.6B` | 强制对齐模型路径 |
| `--max-inference-batch-size` | `32` | 最大推理批次大小 |
| `--max-new-tokens` | `256` | 最大生成 token 数 |
| `--debug` | `False` | 开启调试模式 |

---

## 客户端调用

### Python 客户端

使用 `asr_client.py` 脚本调用服务：

```bash
# 转录本地音频文件
python asr_client.py \n    --url http://SERVER_IP:5000 \n    --audio /path/to/audio.wav \n    --language Chinese

# 转录 URL 音频
python asr_client.py \n    --url http://SERVER_IP:5000 \n    --audio-url "https://example.com/audio.wav" \n    --language English

# 带时间戳的转录
python asr_client.py \n    --url http://SERVER_IP:5000 \n    --audio audio.wav \n    --language Chinese \n    --time-stamps

# 健康检查
python asr_client.py --url http://SERVER_IP:5000 --health-check

# 查看支持的语言
python asr_client.py --url http://SERVER_IP:5000 --list-languages

# 保存结果到文件
python asr_client.py \n    --url http://SERVER_IP:5000 \n    --audio audio.wav \n    --output result.json
```

### 客户端参数说明

| 参数 | 说明 |
|------|------|
| `--url` | 服务端地址 (默认: http://localhost:5000) |
| `--audio` | 本地音频文件路径 |
| `--audio-url` | 音频文件 URL |
| `--language` | 语言 (如 Chinese, English, Japanese) |
| `--context` | 上下文文本 |
| `--time-stamps` | 返回时间戳 |
| `--health-check` | 健康检查 |
| `--list-languages` | 列出支持的语言 |
| `--timeout` | 请求超时时间，秒 (默认: 300) |
| `--output` | 保存结果到 JSON 文件 |

### curl 调用

```bash
# 上传本地文件
curl -X POST http://SERVER_IP:5000/transcribe \n     -F "audio=@/path/to/audio.wav" \n     -F "language=Chinese"

# 使用音频 URL
curl -X POST http://SERVER_IP:5000/transcribe \n     -H "Content-Type: application/json" \n     -d '{"audio_url": "https://example.com/audio.wav", "language": "English"}'

# 使用 base64 编码
curl -X POST http://SERVER_IP:5000/transcribe \n     -H "Content-Type: application/json" \n     -d '{"audio_base64": "<base64_encoded_audio>", "language": "Chinese"}'

# 带时间戳
curl -X POST http://SERVER_IP:5000/transcribe \n     -F "audio=@audio.wav" \n     -F "language=Chinese" \n     -F "return_time_stamps=true"

# 健康检查
curl http://SERVER_IP:5000/health

# 获取支持的语言
curl http://SERVER_IP:5000/languages
```

---

## API 接口说明

### POST /transcribe

转录音频文件。

**请求方式：**

支持以下三种方式之一：

1. **文件上传 (multipart/form-data)**
   - `audio`: 音频文件
   - `language`: 语言 (可选)
   - `context`: 上下文文本 (可选)
   - `return_time_stamps`: 是否返回时间戳 (可选，默认 false)

2. **URL 方式 (JSON)**
   - `audio_url`: 音频文件 URL
   - `language`: 语言 (可选)
   - `context`: 上下文文本 (可选)
   - `return_time_stamps`: 是否返回时间戳 (可选)

3. **Base64 方式 (JSON)**
   - `audio_base64`: Base64 编码的音频数据
   - `language`: 语言 (可选)
   - `context`: 上下文文本 (可选)
   - `return_time_stamps`: 是否返回时间戳 (可选)

**响应格式：**

```json
{
  "language": "Chinese",
  "text": "转录的文本内容",
  "time_stamps": [
    {
      "text": "转录",
      "start_time": 0.48,
      "end_time": 0.88
    }
  ]
}
```

### GET /health

健康检查接口。

**响应：**

```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-04-15T05:30:00.000000"
}
```

### GET /languages

获取支持的语言列表。

**响应：**

```json
{
  "languages": ["Chinese", "English", "Japanese", "Korean", ...]
}
```

---

## 使用示例

### Python 代码集成

```python
from asr_client import ASRClient

# 初始化客户端
client = ASRClient(base_url="http://SERVER_IP:5000", timeout=300)

# 健康检查
health = client.health_check()
print(f"Server status: {health['status']}")

# 转录本地文件
result = client.transcribe_file(
    audio_path="/path/to/audio.wav",
    language="Chinese",
    return_time_stamps=True
)
print(f"Language: {result['language']}")
print(f"Text: {result['text']}")

# 转录 URL
result = client.transcribe_url(
    audio_url="https://example.com/audio.wav",
    language="English"
)

# 转录 base64 数据
with open("/path/to/audio.wav", "rb") as f:
    audio_bytes = f.read()
result = client.transcribe_base64(audio_bytes, language="Japanese")
```

### 批量处理示例

```python
import os
from asr_client import ASRClient

client = ASRClient(base_url="http://SERVER_IP:5000")

audio_dir = "/path/to/audio/files"
results = []

for filename in os.listdir(audio_dir):
    if filename.endswith((".wav", ".mp3", ".flac")):
        audio_path = os.path.join(audio_dir, filename)
        result = client.transcribe_file(audio_path)
        results.append({
            "filename": filename,
            "language": result["language"],
            "text": result["text"]
        })

# 保存结果
import json
with open("transcription_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

---

## 注意事项

1. **网络访问**：确保服务端端口（默认 5000）在防火墙中开放，客户端能够访问。

2. **代理设置**：如果需要通过代理访问外网下载模型，请在启动服务前设置环境变量：
   ```bash
   export https_proxy=http://agent.baidu.com:8891
   ```

3. **GPU 内存**：如果遇到 OOM 错误，可以尝试：
   - 减小 `--max-inference-batch-size`
   - 使用 `--dtype float16` 减少内存占用

4. **长音频处理**：对于长音频，建议增大 `--max-new-tokens` 参数。

5. **超时设置**：客户端默认超时 300 秒，可通过 `--timeout` 参数调整。