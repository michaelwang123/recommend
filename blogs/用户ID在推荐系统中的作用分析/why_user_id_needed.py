import torch
import torch.nn as nn
import numpy as np
import pandas as pd

print("=== 为什么用户ID需要作为特征的一部分？ ===")

# 案例1：用户特征相似，但偏好不同
print("\n1. 案例分析：特征相似但偏好不同")

# 两个用户的特征数据
user_features = {
    "张三": {
        "年龄": 28,
        "收入": 8000,
        "地区": "北京",
        "职业": "程序员",
        "教育": "本科",
        # ... 其他结构化特征
    },
    "李四": {
        "年龄": 29,  # 几乎相同
        "收入": 8200,  # 几乎相同
        "地区": "北京",  # 相同
        "职业": "程序员",  # 相同
        "教育": "本科",  # 相同
        # ... 其他结构化特征也很相似
    }
}

# 但是他们的实际购买历史完全不同
purchase_history = {
    "张三": ["iPhone 12", "MacBook Pro", "AirPods Pro"],  # 喜欢苹果产品
    "李四": ["小米手机", "ThinkPad", "小米耳机"],  # 喜欢性价比产品
}

print("用户特征对比:")
for user, features in user_features.items():
    print(f"{user}: {features}")

print("\n购买历史对比:")
for user, history in purchase_history.items():
    print(f"{user}: {history}")

print("\n分析：")
print("- 两个用户的结构化特征几乎相同")
print("- 但是个人偏好完全不同")
print("- 这种个人偏好很难用结构化特征表示")

# 案例2：用户ID能捕捉的信息
print("\n2. 用户ID能捕捉什么信息？")

class FeatureOnlyModel(nn.Module):
    """只使用用户特征的模型"""
    def __init__(self, user_feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_features):
        return self.mlp(user_features)

class HybridModel(nn.Module):
    """结合用户特征和用户ID的模型"""
    def __init__(self, n_users, user_feature_dim):
        super().__init__()
        # 用户特征处理
        self.feature_mlp = nn.Linear(user_feature_dim, 32)
        # 用户ID嵌入（捕捉个人偏好）
        self.user_embedding = nn.Embedding(n_users, 32)
        # 组合网络
        self.combined_mlp = nn.Sequential(
            nn.Linear(64, 32),  # 32 + 32 = 64
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_ids, user_features):
        feature_repr = self.feature_mlp(user_features)
        id_repr = self.user_embedding(user_ids)
        combined = torch.cat([feature_repr, id_repr], dim=1)
        return self.combined_mlp(combined)

print("模型对比:")
print("特征模型：只能基于结构化特征预测")
print("混合模型：特征 + 个人偏好")

# 案例3：什么情况下不需要用户ID
print("\n3. 什么情况下不需要用户ID？")

scenarios = {
    "需要用户ID的情况": [
        "用户特征不够全面",
        "存在难以量化的个人偏好",
        "用户行为存在个体差异",
        "需要处理长期用户的个性化推荐"
    ],
    "不需要用户ID的情况": [
        "用户特征已经非常全面",
        "推荐任务主要基于客观属性",
        "处理新用户（冷启动）",
        "需要可解释的推荐结果"
    ]
}

for scenario, conditions in scenarios.items():
    print(f"\n{scenario}:")
    for condition in conditions:
        print(f"  - {condition}")

# 案例4：实际效果对比
print("\n4. 实际效果对比：")

# 模拟数据
n_users = 1000
user_feature_dim = 64
np.random.seed(42)

# 用户特征（结构化信息）
user_features = torch.randn(n_users, user_feature_dim)
user_ids = torch.arange(n_users)

# 模拟真实评分（包含个人偏好）
true_ratings = torch.randn(n_users, 1)

# 只使用特征的模型
feature_model = FeatureOnlyModel(user_feature_dim)
feature_pred = feature_model(user_features)

# 混合模型
hybrid_model = HybridModel(n_users, user_feature_dim)
hybrid_pred = hybrid_model(user_ids, user_features)

# 计算误差
feature_error = torch.mean((feature_pred - true_ratings) ** 2)
hybrid_error = torch.mean((hybrid_pred - true_ratings) ** 2)

print(f"只使用特征的模型误差: {feature_error.item():.4f}")
print(f"混合模型误差: {hybrid_error.item():.4f}")

# 案例5：具体的特征 vs ID 对比
print("\n5. 具体对比：特征 vs ID")

comparison_table = pd.DataFrame({
    "方面": ["信息类型", "可解释性", "冷启动", "个性化程度", "数据需求"],
    "用户特征": ["结构化属性", "高", "容易处理", "基于群体", "用户画像"],
    "用户ID": ["隐含偏好", "低", "无法处理", "高度个性化", "历史行为"]
})

print(comparison_table.to_string(index=False))

# 案例6：推荐策略
print("\n6. 推荐策略建议：")

strategies = {
    "如果用户特征很全面": {
        "方法": "只使用用户特征",
        "适用场景": "新用户推荐、可解释推荐",
        "优势": "简单、可解释、无冷启动问题"
    },
    "如果用户特征不够全面": {
        "方法": "特征 + 用户ID",
        "适用场景": "老用户个性化推荐",
        "优势": "捕捉个人偏好、推荐更准确"
    },
    "如果没有用户特征": {
        "方法": "只使用用户ID",
        "适用场景": "协同过滤场景",
        "优势": "不依赖用户属性数据"
    }
}

for strategy, details in strategies.items():
    print(f"\n{strategy}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

print("\n=== 总结 ===")
print("1. 用户ID主要用于捕捉难以量化的个人偏好")
print("2. 如果用户特征已经很全面，ID可能不是必需的")
print("3. 最好的方法是根据具体场景选择")
print("4. 混合方法通常效果最好，但增加了模型复杂度") 