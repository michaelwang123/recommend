# 二手设备推荐系统 - 快速开始指南

## 🚀 快速启动

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv recommend 
source recommend/bin/activate  # Linux/Mac


# 安装依赖
pip install -r requirements.txt
```

### 2. 基础使用

```bash
# 运行基础示例
python examples/basic_usage.py

# 查看推荐结果
python examples/advanced_features.py
```

### 3. 启动Web服务

```bash
# 启动推荐API服务
python examples/deployment_example.py

# 访问健康检查
curl http://localhost:5000/health

# 获取用户推荐
curl http://localhost:5000/api/v1/recommend/user/1?k=5

# 获取相似设备推荐
curl http://localhost:5000/api/v1/recommend/similar/1?k=5
```

### 4. Docker部署

```bash
# 构建Docker镜像
docker build -t second-hand-recommender .

# 运行容器
docker run -p 5000:5000 second-hand-recommender

# 或使用docker-compose
docker-compose up -d
```

## 🔧 配置说明

### 基础配置 (config.yaml)
```yaml
recommendation:
  max_recommendations: 10
  similarity_threshold: 0.7
  enable_location_filter: true
  max_distance_km: 50

cache:
  enable_cache: true
  cache_type: redis  # 或 memory
  cache_ttl: 3600

api:
  rate_limit: 60
  enable_cors: true
```

### 高级配置
```yaml
model:
  embedding_dim: 64
  learning_rate: 0.001
  batch_size: 256
  epochs: 30

redis:
  host: localhost
  port: 6379
  db: 0
  password: null

logging:
  level: INFO
  log_file: recommender.log
```

## 📊 API 接口

### 1. 用户推荐
```bash
GET /api/v1/recommend/user/{user_id}
参数: k=10 (推荐数量)
```

### 2. 相似设备推荐
```bash
GET /api/v1/recommend/similar/{device_id}
参数: k=10 (推荐数量)
```

### 3. 价格推荐
```bash
POST /api/v1/recommend/price
Body: {
  "brand": "苹果",
  "model": "iPhone 13",
  "condition": "良好",
  "age_months": 12
}
```

### 4. 地理位置推荐
```bash
POST /api/v1/recommend/nearby
Body: {
  "latitude": 39.9042,
  "longitude": 116.4074,
  "radius_km": 10
}
```

## 🧪 测试验证

### 运行单元测试
```bash
python -m pytest tests/
```

### 性能测试
```bash
# 推荐延迟测试
python tests/performance_test.py

# 并发测试
python tests/load_test.py
```

## 🔍 核心功能展示

### 1. 个性化推荐
```python
from second_hand_device_recommender import SecondHandRecommendationSystem

# 初始化推荐系统
recommender = SecondHandRecommendationSystem()

# 训练模型
recommender.train(users, devices, interactions)

# 获取推荐
recommendations = recommender.recommend_for_user(user_id=1, k=5)
print(f"为用户1推荐的设备: {recommendations}")
```

### 2. 价格推荐
```python
price_recommender = PriceRecommender()
price_recommender.train_price_model(device_data)

price_info = price_recommender.recommend_price({
    'brand': '苹果',
    'model': 'iPhone 13',
    'condition': '良好',
    'age_months': 12
})
print(f"推荐价格: {price_info}")
```

### 3. 地理位置推荐
```python
location_recommender = LocationRecommender(max_distance_km=20)
nearby_devices = location_recommender.recommend_nearby_devices(
    user_location=(39.9042, 116.4074),
    device_data=devices,
    top_k=10
)
```

## 📈 监控和运维

### 1. 日志查看
```bash
# 查看实时日志
tail -f logs/recommender.log

# 查看错误日志
grep ERROR logs/recommender.log
```

### 2. 性能监控
```bash
# 查看系统状态
curl http://localhost:5000/api/v1/stats

# 查看缓存状态
curl http://localhost:5000/api/v1/cache/stats
```

### 3. 数据更新
```bash
# 重新训练模型
curl -X POST http://localhost:5000/api/v1/model/retrain

# 清除缓存
curl -X POST http://localhost:5000/api/v1/cache/clear
```

## 🛠️ 常见问题

### Q1: 推荐结果为空
**A:** 检查数据质量和用户交互记录，确保有足够的训练数据。

### Q2: 服务启动失败
**A:** 检查端口是否被占用，确认依赖包是否正确安装。

### Q3: 推荐速度慢
**A:** 启用缓存，优化数据库查询，考虑使用GPU加速。

### Q4: 内存占用过高
**A:** 调整batch_size和embedding_dim，启用模型压缩。

## 📚 扩展开发

### 1. 自定义推荐算法
```python
class CustomRecommender(SecondHandRecommendationSystem):
    def custom_recommend(self, user_id, context):
        # 实现自定义推荐逻辑
        pass
```

### 2. 新增特征
```python
# 在DeviceFeatureExtractor中添加新特征
def extract_custom_features(self, device_data):
    # 提取自定义特征
    pass
```

### 3. 集成外部服务
```python
# 集成第三方API
def integrate_external_api(self, device_id):
    # 调用外部价格API
    pass
```

## 🎯 最佳实践

1. **数据质量**: 确保数据清洁和完整
2. **模型更新**: 定期重新训练模型
3. **性能优化**: 使用缓存和异步处理
4. **监控告警**: 设置关键指标监控
5. **A/B测试**: 持续优化推荐效果

## 📞 技术支持

如有问题，请查看详细文档或提交issue。 