#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简洁版：PyTorch用户ID嵌入向量生成示例
展示核心代码和基本用法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def basic_user_embedding_demo():
    """基础用户嵌入向量生成演示"""
    print("🚀 基础用户嵌入向量生成演示")
    print("=" * 50)
    
    # 1. 创建用户嵌入层
    num_users = 1000      # 用户总数
    embedding_dim = 64    # 嵌入向量维度
    
    user_embedding = nn.Embedding(num_users, embedding_dim)
    
    print(f"✅ 创建嵌入层: {num_users} 用户 × {embedding_dim} 维度")
    print(f"   参数数量: {user_embedding.weight.numel():,}")
    print(f"   嵌入矩阵形状: {user_embedding.weight.shape}")
    
    # 2. 单个用户ID嵌入
    user_id = torch.tensor([123])
    user_vector = user_embedding(user_id)
    
    print(f"\n🔍 单个用户嵌入：")
    print(f"   用户ID: {user_id.item()}")
    print(f"   嵌入向量形状: {user_vector.shape}")
    print(f"   向量前5维: {user_vector[0][:5].detach().numpy()}")
    
    # 3. 批量用户ID嵌入
    user_ids = torch.tensor([10, 25, 50, 100])
    user_vectors = user_embedding(user_ids)
    
    print(f"\n📊 批量用户嵌入：")
    print(f"   用户IDs: {user_ids.tolist()}")
    print(f"   嵌入向量形状: {user_vectors.shape}")
    
    for i, uid in enumerate(user_ids):
        print(f"   用户{uid.item():3d}: {user_vectors[i][:3].detach().numpy()}")
    
    # 4. 计算用户相似度
    print(f"\n🔗 用户相似度计算：")
    user1_vec = user_vectors[0]  # 用户10
    user2_vec = user_vectors[1]  # 用户25
    
    similarity = torch.cosine_similarity(user1_vec, user2_vec, dim=0)
    print(f"   用户10 vs 用户25 相似度: {similarity.item():.4f}")
    
    return user_embedding

def training_embedding_demo():
    """训练过程中的嵌入向量更新演示"""
    print(f"\n🎓 训练过程演示")
    print("=" * 50)
    
    # 创建简单的推荐模型
    class SimpleModel(nn.Module):
        def __init__(self, num_users, num_items, embed_dim):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embed_dim)
            self.item_embedding = nn.Embedding(num_items, embed_dim)
            
        def forward(self, user_ids, item_ids):
            user_vecs = self.user_embedding(user_ids)
            item_vecs = self.item_embedding(item_ids)
            scores = torch.sum(user_vecs * item_vecs, dim=1)
            return scores
    
    # 初始化模型
    model = SimpleModel(num_users=100, num_items=50, embed_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print(f"✅ 创建推荐模型")
    print(f"   用户数: 100, 物品数: 50, 嵌入维度: 32")
    
    # 模拟训练数据
    batch_size = 16
    user_ids = torch.randint(0, 100, (batch_size,))
    item_ids = torch.randint(0, 50, (batch_size,))
    ratings = torch.rand(batch_size) * 5  # 评分 0-5
    
    print(f"\n📝 训练数据示例:")
    print(f"   批大小: {batch_size}")
    print(f"   用户ID: {user_ids[:5].tolist()}")
    print(f"   物品ID: {item_ids[:5].tolist()}")
    print(f"   评分: {ratings[:5].tolist()}")
    
    # 训练前的嵌入向量
    initial_embedding = model.user_embedding.weight.data[0].clone()
    print(f"\n🔄 训练前用户0嵌入: {initial_embedding[:5].numpy()}")
    
    # 训练步骤
    for epoch in range(3):
        # 前向传播
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # 训练后的嵌入向量
    final_embedding = model.user_embedding.weight.data[0]
    print(f"\n✅ 训练后用户0嵌入: {final_embedding[:5].numpy()}")
    
    # 计算变化
    change = torch.norm(final_embedding - initial_embedding).item()
    print(f"   嵌入向量变化量: {change:.6f}")
    
    return model

def save_load_demo(user_embedding):
    """保存和加载嵌入向量演示"""
    print(f"\n💾 保存和加载演示")
    print("=" * 50)
    
    # 保存模型
    torch.save(user_embedding.state_dict(), 'user_embedding.pth')
    print(f"✅ 嵌入向量已保存")
    
    # 加载模型
    new_embedding = nn.Embedding(1000, 64)
    new_embedding.load_state_dict(torch.load('user_embedding.pth'))
    print(f"✅ 嵌入向量已加载")
    
    # 验证
    test_id = torch.tensor([42])
    original = user_embedding(test_id)
    loaded = new_embedding(test_id)
    
    is_same = torch.allclose(original, loaded)
    print(f"✅ 验证结果: {'成功' if is_same else '失败'}")
    
    return new_embedding

def practical_example():
    """实际使用示例"""
    print(f"\n🎯 实际使用示例")
    print("=" * 50)
    
    # 创建用户嵌入
    user_embedding = nn.Embedding(1000, 64)
    
    # 模拟实际用户ID
    active_users = [15, 42, 88, 156, 299]
    user_ids = torch.tensor(active_users)
    
    # 获取嵌入向量
    user_vectors = user_embedding(user_ids)
    
    print(f"活跃用户: {active_users}")
    print(f"嵌入向量形状: {user_vectors.shape}")
    
    # 找出最相似的用户
    target_user = 0  # 目标用户索引
    target_vec = user_vectors[target_user]
    
    similarities = torch.cosine_similarity(
        target_vec.unsqueeze(0), 
        user_vectors
    )
    
    print(f"\n用户{active_users[target_user]}与其他用户的相似度:")
    for i, uid in enumerate(active_users):
        if i != target_user:
            print(f"   用户{uid}: {similarities[i].item():.4f}")
    
    # 找出最相似的用户
    similarities[target_user] = -1  # 排除自己
    most_similar_idx = torch.argmax(similarities).item()
    
    print(f"\n🔍 最相似的用户: {active_users[most_similar_idx]}")
    print(f"   相似度: {similarities[most_similar_idx].item():.4f}")
    
    return user_vectors

def main():
    """主函数"""
    print("🎉 PyTorch用户ID嵌入向量生成 - 简洁版")
    print("=" * 80)
    
    # 1. 基础演示
    user_embedding = basic_user_embedding_demo()
    
    # 2. 训练演示
    model = training_embedding_demo()
    
    # 3. 保存加载演示
    save_load_demo(user_embedding)
    
    # 4. 实际应用示例
    practical_example()
    
    print(f"\n" + "=" * 80)
    print("🎯 核心代码模板:")
    print("""
# 1. 创建嵌入层
user_embedding = nn.Embedding(num_users, embedding_dim)

# 2. 获取嵌入向量
user_ids = torch.tensor([10, 20, 30])
user_vectors = user_embedding(user_ids)

# 3. 计算相似度
similarity = torch.cosine_similarity(vec1, vec2, dim=0)

# 4. 在模型中使用
class RecommendModel(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
    
    def forward(self, user_ids):
        return self.user_embedding(user_ids)

# 5. 训练
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss.backward()
optimizer.step()
""")
    print("=" * 80)

if __name__ == "__main__":
    main() 