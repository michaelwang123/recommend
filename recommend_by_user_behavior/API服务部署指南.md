# 推荐系统 API 服务部署指南

## 📋 概述

在训练并保存了推荐模型后，你可以通过以下两种方式对外提供接口服务：

1. **Python API 服务**（推荐）- 直接使用 FastAPI 构建
2. **Go 客户端调用** - 通过 HTTP 调用 Python API

---

## 🐍 方案一：Python API 服务

### 1. 安装依赖

```bash
# 安装 API 服务依赖
pip install -r api_requirements.txt
```

### 2. 启动 API 服务

```bash
# 方式1：直接运行
python api_service.py

# 方式2：使用 uvicorn 命令
uvicorn api_service:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 验证服务状态

```bash
# 健康检查
curl http://localhost:8000/health

# 查看 API 文档
# 浏览器访问: http://localhost:8000/docs
```

### 4. API 接口说明

#### 4.1 健康检查
```http
GET /health
```

响应示例：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_users": 1000,
  "num_items": 5000
}
```

#### 4.2 获取推荐
```http
POST /recommend
Content-Type: application/json

{
  "user_id": "user_123",
  "top_n": 10,
  "exclude_rated": true
}
```

#### 4.3 Python 客户端调用示例

```python
import requests

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 获取推荐
data = {
    "user_id": "user_123",
    "top_n": 5,
    "exclude_rated": True
}
response = requests.post("http://localhost:8000/recommend", json=data)
recommendations = response.json()
print(recommendations)
```


 