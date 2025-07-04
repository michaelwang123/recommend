#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手设备推荐系统 - 部署示例

这个示例展示了如何将推荐系统部署到生产环境：
1. Flask Web API 服务
2. 模型服务化
3. 缓存优化
4. 监控和日志
5. Docker 容器化
6. 性能优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yaml
import json
import time
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import redis
from second_hand_device_recommender import SecondHandRecommendationSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationService:
    """推荐服务类 - 封装推荐逻辑"""
    
    def __init__(self, config_path='config.yaml'):
        """初始化推荐服务"""
        self.config = self._load_config(config_path)
        self.recommender = SecondHandRecommendationSystem()
        self.cache = None
        self.model_loaded = False
        
        # 初始化缓存
        if self.config.get('cache', {}).get('enable_cache', False):
            self._init_cache()
    
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件未找到: {config_path}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'recommendation': {
                'max_recommendations': 10,
                'similarity_threshold': 0.7
            },
            'cache': {
                'enable_cache': True,
                'cache_type': 'memory',
                'cache_ttl': 3600
            },
            'api': {
                'rate_limit': 60
            }
        }
    
    def _init_cache(self):
        """初始化缓存"""
        cache_config = self.config.get('cache', {})
        cache_type = cache_config.get('cache_type', 'memory')
        
        if cache_type == 'redis':
            try:
                redis_config = cache_config.get('redis', {})
                self.cache = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    password=redis_config.get('password'),
                    decode_responses=True
                )
                logger.info("Redis缓存初始化成功")
            except Exception as e:
                logger.error(f"Redis连接失败: {e}")
                self.cache = None
        else:
            # 使用内存缓存
            self.cache = {}
            logger.info("内存缓存初始化成功")
    
    def load_model(self, model_path=None):
        """加载推荐模型"""
        try:
            if model_path:
                self.recommender.load_model(model_path)
            else:
                # 加载默认模型或训练新模型
                self._train_default_model()
            
            self.model_loaded = True
            logger.info("推荐模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _train_default_model(self):
        """训练默认模型（用于演示）"""
        # 生成演示数据
        users = pd.DataFrame({
            'user_id': range(1, 1001),
            'age': np.random.randint(18, 65, 1000),
            'city': np.random.choice(['北京', '上海', '广州', '深圳'], 1000)
        })
        
        devices = pd.DataFrame({
            'device_id': range(1, 5001),
            'brand': np.random.choice(['苹果', '华为', '小米', '三星'], 5000),
            'category': np.random.choice(['手机', '笔记本', '平板'], 5000),
            'price': np.random.randint(1000, 10000, 5000)
        })
        
        interactions = pd.DataFrame({
            'user_id': np.random.randint(1, 1001, 20000),
            'device_id': np.random.randint(1, 5001, 20000),
            'interaction_type': np.random.choice(['view', 'like', 'purchase'], 20000),
            'rating': np.random.randint(1, 6, 20000),
            'timestamp': pd.date_range('2023-01-01', periods=20000, freq='H')
        })
        
        logger.info("开始训练默认模型...")
        self.recommender.train(users, devices, interactions)
        logger.info("默认模型训练完成")
    
    def get_cache_key(self, prefix, *args):
        """生成缓存键"""
        return f"{prefix}:{':'.join(map(str, args))}"
    
    def get_from_cache(self, key):
        """从缓存获取数据"""
        if not self.cache:
            return None
        
        try:
            if isinstance(self.cache, dict):
                # 内存缓存
                return self.cache.get(key)
            else:
                # Redis缓存
                data = self.cache.get(key)
                return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"缓存读取失败: {e}")
            return None
    
    def set_to_cache(self, key, data, ttl=None):
        """设置缓存数据"""
        if not self.cache:
            return
        
        try:
            if isinstance(self.cache, dict):
                # 内存缓存
                self.cache[key] = data
            else:
                # Redis缓存
                ttl = ttl or self.config.get('cache', {}).get('cache_ttl', 3600)
                self.cache.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.error(f"缓存写入失败: {e}")
    
    def recommend_for_user(self, user_id, k=10):
        """为用户推荐设备"""
        if not self.model_loaded:
            raise ValueError("模型未加载")
        
        # 检查缓存
        cache_key = self.get_cache_key('user_recs', user_id, k)
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            logger.info(f"从缓存获取用户 {user_id} 的推荐")
            return cached_result
        
        # 生成推荐
        start_time = time.time()
        recommendations = self.recommender.recommend_for_user(user_id, k=k)
        end_time = time.time()
        
        # 格式化结果
        result = {
            'user_id': user_id,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': (end_time - start_time) * 1000
        }
        
        # 设置缓存
        self.set_to_cache(cache_key, result)
        
        logger.info(f"为用户 {user_id} 生成推荐，耗时 {result['processing_time_ms']:.2f}ms")
        return result
    
    def recommend_similar_devices(self, device_id, k=10):
        """推荐相似设备"""
        if not self.model_loaded:
            raise ValueError("模型未加载")
        
        # 检查缓存
        cache_key = self.get_cache_key('similar_devices', device_id, k)
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            logger.info(f"从缓存获取设备 {device_id} 的相似推荐")
            return cached_result
        
        # 生成推荐
        start_time = time.time()
        similar_devices = self.recommender.recommend_similar_devices(device_id, k=k)
        end_time = time.time()
        
        # 格式化结果
        result = {
            'device_id': device_id,
            'similar_devices': similar_devices,
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': (end_time - start_time) * 1000
        }
        
        # 设置缓存
        self.set_to_cache(cache_key, result)
        
        logger.info(f"为设备 {device_id} 生成相似推荐，耗时 {result['processing_time_ms']:.2f}ms")
        return result
    
    def get_health_status(self):
        """获取服务健康状态"""
        status = {
            'service': 'second_hand_recommendation',
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': self.model_loaded,
            'cache_enabled': self.cache is not None
        }
        
        # 检查模型状态
        if not self.model_loaded:
            status['status'] = 'unhealthy'
            status['error'] = 'Model not loaded'
        
        return status

class FlaskApp:
    """Flask 应用类"""
    
    def __init__(self, recommendation_service):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.recommendation_service = recommendation_service
        self.request_count = 0
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify(self.recommendation_service.get_health_status())
        
        @self.app.route('/api/v1/recommend/user/<int:user_id>', methods=['GET'])
        def recommend_for_user(user_id):
            """为用户推荐设备"""
            try:
                k = request.args.get('k', 10, type=int)
                k = min(k, 50)  # 限制最大推荐数量
                
                result = self.recommendation_service.recommend_for_user(user_id, k=k)
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"用户推荐失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/recommend/similar/<int:device_id>', methods=['GET'])
        def recommend_similar_devices(device_id):
            """推荐相似设备"""
            try:
                k = request.args.get('k', 10, type=int)
                k = min(k, 50)  # 限制最大推荐数量
                
                result = self.recommendation_service.recommend_similar_devices(device_id, k=k)
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"相似设备推荐失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/stats', methods=['GET'])
        def get_stats():
            """获取服务统计信息"""
            stats = {
                'total_requests': self.request_count,
                'service_start_time': datetime.now().isoformat(),
                'memory_usage': self._get_memory_usage()
            }
            return jsonify(stats)
        
        @self.app.before_request
        def before_request():
            """请求前处理"""
            self.request_count += 1
        
        @self.app.after_request
        def after_request(response):
            """请求后处理"""
            # 添加响应头
            response.headers['X-Request-ID'] = f"req_{self.request_count}"
            return response
    
    def _get_memory_usage(self):
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """运行Flask应用"""
        self.app.run(host=host, port=port, debug=debug)

def create_docker_files():
    """创建Docker相关文件"""
    
    # Dockerfile
    dockerfile_content = """FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 5000

# 设置环境变量
ENV PYTHONPATH=/app
ENV FLASK_APP=deployment_example.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# 启动应用
CMD ["python", "deployment_example.py"]
"""
    
    # docker-compose.yml
    docker_compose_content = """version: '3.8'

services:
  recommendation-service:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - REDIS_HOST=redis
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - recommendation-service
    restart: unless-stopped

volumes:
  redis-data:
"""
    
    # nginx.conf
    nginx_conf_content = """events {
    worker_connections 1024;
}

http {
    upstream recommendation_service {
        server recommendation-service:5000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://recommendation_service;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://recommendation_service/health;
        }
    }
}
"""
    
    # 写入文件
    with open('second_hand_devices/Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('second_hand_devices/docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    with open('second_hand_devices/nginx.conf', 'w') as f:
        f.write(nginx_conf_content)
    
    print("Docker配置文件创建完成!")

def main():
    """主函数"""
    print("🚀 二手设备推荐系统 - 部署示例")
    print("=" * 50)
    
    # 创建Docker文件
    create_docker_files()
    
    # 初始化推荐服务
    print("\n📦 初始化推荐服务...")
    recommendation_service = RecommendationService()
    
    # 加载模型
    print("🤖 加载推荐模型...")
    recommendation_service.load_model()
    
    # 创建Flask应用
    print("🌐 创建Web服务...")
    flask_app = FlaskApp(recommendation_service)
    
    print("\n✅ 服务初始化完成!")
    print("\n🔗 可用的API端点:")
    print("  GET /health - 健康检查")
    print("  GET /api/v1/recommend/user/<user_id> - 用户推荐")
    print("  GET /api/v1/recommend/similar/<device_id> - 相似设备推荐")
    print("  GET /api/v1/stats - 服务统计")
    
    print("\n🐳 Docker部署命令:")
    print("  docker-compose up -d")
    
    print("\n🏃 启动开发服务器...")
    
    # 启动Flask应用
    try:
        flask_app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")

if __name__ == "__main__":
    main() 