import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=== 用户转换为64维向量详细演示 ===")

# 步骤1：原始用户数据
print("\n1. 原始用户数据:")
users = ["张三", "李四", "王五", "赵六", "钱七"]
print(f"用户列表: {users}")

# 步骤2：用户ID编码
print("\n2. 用户ID编码:")
user_encoder = LabelEncoder()
user_encoded = user_encoder.fit_transform(users)
print(f"编码后的用户ID: {user_encoded}")
print(f"编码映射关系:")
for i, user in enumerate(users):
    print(f"  '{user}' -> {user_encoded[i]}")

# 步骤3：创建嵌入层
print("\n3. 创建嵌入层:")
n_users = len(users)  # 5个用户
embedding_dim = 64    # 64维向量
user_embedding = nn.Embedding(n_users, embedding_dim)
print(f"嵌入层参数: nn.Embedding({n_users}, {embedding_dim})")
print(f"嵌入层权重形状: {user_embedding.weight.shape}")
print(f"总参数数量: {n_users * embedding_dim}")

# 步骤4：转换过程演示
print("\n4. 转换过程演示:")
print(f"用户'张三'的转换过程:")
print(f"  原始用户名: '张三'")
print(f"  编码后ID: {user_encoded[0]}")

# 转换为PyTorch张量
user_id_tensor = torch.LongTensor([user_encoded[0]])
print(f"  PyTorch张量: {user_id_tensor}")

# 通过嵌入层转换
user_vector = user_embedding(user_id_tensor)
print(f"  64维向量形状: {user_vector.shape}")
print(f"  64维向量前10个值: {user_vector[0][:10].detach().numpy()}")

print("\n=== 转换过程总结 ===")
print("1. 用户名 → 数字ID（编码）")
print("2. 数字ID → PyTorch张量")
print("3. 张量 → 64维向量（嵌入层）")
print("4. 向量用于计算推荐评分")
