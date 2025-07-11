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

# 步骤5：批量转换
print("\n5. 批量转换演示:")
all_user_ids = torch.LongTensor(user_encoded)
all_user_vectors = user_embedding(all_user_ids)
print(f"批量输入形状: {all_user_ids.shape}")
print(f"批量输出形状: {all_user_vectors.shape}")

# 展示每个用户的向量
for i, user in enumerate(users):
    vector = all_user_vectors[i]
    print(f"用户'{user}': 向量长度={len(vector)}, 前5个值={vector[:5].detach().numpy()}")

# 步骤6：嵌入层的学习过程
print("\n6. 嵌入层的学习过程:")
print("初始状态：向量是随机初始化的")
print("训练过程：通过反向传播不断调整向量值")
print("训练目标：让相似用户的向量更接近")

# 步骤7：实际使用示例
print("\n7. 实际使用示例:")
print("在推荐系统中的使用:")
print("  user_emb = self.user_embedding(user_ids)")
print("  device_emb = self.device_embedding(device_ids)")
print("  score = torch.sum(user_emb * device_emb, dim=1)")

# 步骤8：向量相似度计算
print("\n8. 向量相似度计算:")
user1_vector = all_user_vectors[0]  # 张三
user2_vector = all_user_vectors[1]  # 李四
similarity = torch.cosine_similarity(user1_vector, user2_vector, dim=0)
print(f"张三和李四的相似度: {similarity.item():.4f}")

print("\n=== 转换过程总结 ===")
print("1. 用户名 → 数字ID（编码）")
print("2. 数字ID → PyTorch张量")
print("3. 张量 → 64维向量（嵌入层）")
print("4. 向量用于计算推荐评分") 