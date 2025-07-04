# 二手设备交易推荐系统

这是一个专门针对二手设备交易平台的推荐系统解决方案，提供多种推荐场景和完整的技术实现。

## 📁 文件结构

```
second_hand_devices/
├── README.md                                    # 项目说明文档
├── second_hand_recommendation_scenarios.md     # 详细应用场景分析
├── second_hand_device_recommender.py          # 核心推荐系统实现
├── requirements.txt                            # 项目依赖
├── config.yaml                                # 配置文件示例
├── data/                                      # 数据文件夹
│   ├── sample_users.csv                       # 示例用户数据
│   ├── sample_devices.csv                     # 示例设备数据
│   └── sample_interactions.csv               # 示例交互数据
└── examples/                                  # 使用示例
    ├── basic_usage.py                         # 基础使用示例
    ├── advanced_features.py                   # 高级功能示例
    └── deployment_example.py                 # 部署示例
```

## 🎯 核心功能

### 1. 相似商品推荐
- 基于设备特征的相似性计算
- 支持多维度匹配（品牌、型号、价格等）
- 实时相似度计算

### 2. 个性化推荐
- 用户行为分析
- 个性化偏好建模
- 智能推荐算法

### 3. 价格推荐
- 智能定价建议
- 市场行情分析
- 动态价格调整

### 4. 地理位置推荐
- 附近设备推荐
- 交通便利性考虑
- 配送成本优化

### 5. 买家卖家匹配
- 信用评级匹配
- 交易偏好分析
- 潜在买家推荐

## 🚀 快速开始

### 1. 安装依赖

```bash
cd second_hand_devices
pip install -r requirements.txt
```

### 2. 运行基础示例

```bash
python second_hand_device_recommender.py
```

### 3. 查看应用场景

详细的应用场景分析请参考：[second_hand_recommendation_scenarios.md](second_hand_recommendation_scenarios.md)

## 📊 数据格式

### 用户数据格式
```python
users = pd.DataFrame({
    'user_id': [1, 2, 3, ...],
    'age': [25, 30, 28, ...],
    'city': ['北京', '上海', '广州', ...],
    'latitude': [39.9, 31.2, 23.1, ...],
    'longitude': [116.4, 121.5, 113.3, ...]
})
```

### 设备数据格式
```python
devices = pd.DataFrame({
    'device_id': [1, 2, 3, ...],
    'brand': ['苹果', '华为', '小米', ...],
    'model': ['iPhone 13', 'Mate 40', 'Mi 11', ...],
    'category': ['手机', '笔记本', '平板', ...],
    'condition': ['九成新', '八成新', '全新', ...],
    'price': [3000, 5000, 2000, ...],
    'age_months': [12, 6, 24, ...]
})
```

### 交互数据格式
```python
interactions = pd.DataFrame({
    'user_id': [1, 2, 1, ...],
    'device_id': [10, 15, 20, ...],
    'interaction_type': ['view', 'like', 'purchase', ...],
    'rating': [4, 5, 3, ...],
    'timestamp': ['2023-01-01', '2023-01-02', ...]
})
```

## 🔧 配置说明

### 模型参数配置
```yaml
model:
  embedding_dim: 64
  hidden_dims: [128, 64]
  learning_rate: 0.001
  batch_size: 256
  epochs: 50

recommendation:
  similarity_threshold: 0.7
  max_recommendations: 10
  enable_location_filter: true
  max_distance_km: 50

business:
  price_tolerance: 0.2
  condition_weight: 0.3
  brand_preference_weight: 0.4
```

## 📈 性能指标

### 推荐质量指标
- **准确率 (Precision)**: 推荐结果中相关物品的比例
- **召回率 (Recall)**: 相关物品中被推荐的比例
- **覆盖率 (Coverage)**: 推荐系统能够推荐的物品比例
- **多样性 (Diversity)**: 推荐结果的多样化程度

### 业务指标
- **点击率 (CTR)**: 推荐内容的点击率
- **转化率 (CVR)**: 推荐到购买的转化率
- **客单价 (AOV)**: 平均订单价值
- **用户留存率**: 用户回访和使用频率

## 🛠️ 扩展开发

### 添加新的推荐算法
```python
class CustomRecommender(nn.Module):
    def __init__(self, ...):
        # 自定义推荐算法实现
        pass
    
    def forward(self, ...):
        # 前向传播逻辑
        pass
```

### 集成外部数据源
```python
class ExternalDataIntegrator:
    def fetch_market_data(self):
        # 获取市场数据
        pass
    
    def update_price_trends(self):
        # 更新价格趋势
        pass
```

### 自定义评估指标
```python
class CustomMetrics:
    def calculate_business_impact(self):
        # 计算业务影响
        pass
    
    def user_satisfaction_score(self):
        # 用户满意度评分
        pass
```

## 🎨 用户界面集成

### Web API 接口示例
```python
from flask import Flask, jsonify, request

app = Flask(__name__)
recommender = SecondHandRecommendationSystem()

@app.route('/recommend/similar/<int:device_id>')
def recommend_similar(device_id):
    similar_devices = recommender.recommend_similar_devices(device_id)
    return jsonify({'similar_devices': similar_devices})

@app.route('/recommend/user/<int:user_id>')
def recommend_for_user(user_id):
    recommendations = recommender.recommend_for_user(user_id)
    return jsonify(recommendations)
```

### 前端集成示例
```javascript
// 获取相似商品推荐
async function getSimilarDevices(deviceId) {
    const response = await fetch(`/api/recommend/similar/${deviceId}`);
    const data = await response.json();
    return data.similar_devices;
}

// 获取个性化推荐
async function getPersonalizedRecommendations(userId) {
    const response = await fetch(`/api/recommend/user/${userId}`);
    const data = await response.json();
    return data;
}
```

## 📝 部署指南

### Docker 部署
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

### 云服务部署
- **AWS**: 使用 EC2 + RDS + S3
- **阿里云**: 使用 ECS + RDS + OSS
- **腾讯云**: 使用 CVM + CDB + COS

### 性能优化
- 使用 Redis 缓存热门推荐
- 使用 FAISS 加速相似性搜索
- 使用 GPU 加速模型训练和推理

## 📞 技术支持

### 问题反馈
- 提交 Issue 到项目仓库
- 发送邮件到技术支持邮箱
- 查看 FAQ 文档

### 贡献指南
1. Fork 项目仓库
2. 创建功能分支
3. 提交代码更改
4. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

## 🔗 相关链接

- [主项目 README](../README.md)
- [通用推荐系统实现](../similarity_recommendation_basic.py)
- [高级推荐算法](../advanced_similarity_recommender.py)
- [生产级推荐系统](../production_recommender.py)

---

💡 **提示**: 这是一个专门针对二手设备交易的推荐系统实现，可以根据具体业务需求进行定制和扩展。 