#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch用户ID嵌入向量生成示例
简单实用的代码示范，展示如何使用nn.Embedding生成用户嵌入向量
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UserEmbeddingDemo:
    def __init__(self, num_users=1000, embedding_dim=64):
        """
        初始化用户嵌入演示
        
        Args:
            num_users: 用户总数
            embedding_dim: 嵌入向量维度
        """
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        
        # 创建用户嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        print(f"✅ 创建用户嵌入层：")
        print(f"   用户数量: {num_users}")
        print(f"   嵌入维度: {embedding_dim}")
        print(f"   参数数量: {num_users * embedding_dim:,}")
        print(f"   嵌入矩阵形状: {self.user_embedding.weight.shape}")
        
    def basic_embedding_example(self):
        """基础嵌入向量生成示例"""
        print(f"\n🔍 基础嵌入向量生成示例")
        print("=" * 60)
        
        # 单个用户ID
        user_id = torch.tensor([123])
        user_vector = self.user_embedding(user_id)
        
        print(f"单个用户示例：")
        print(f"  用户ID: {user_id.item()}")
        print(f"  输入形状: {user_id.shape}")
        print(f"  输出形状: {user_vector.shape}")
        print(f"  嵌入向量前5维: {user_vector[0][:5].detach().numpy()}")
        
        # 批量用户ID
        user_ids = torch.tensor([10, 25, 50, 100, 200])
        user_vectors = self.user_embedding(user_ids)
        
        print(f"\n批量用户示例：")
        print(f"  用户IDs: {user_ids.tolist()}")
        print(f"  输入形状: {user_ids.shape}")
        print(f"  输出形状: {user_vectors.shape}")
        
        # 显示每个用户的嵌入向量
        for i, uid in enumerate(user_ids):
            vector = user_vectors[i]
            print(f"  用户{uid.item():3d}: {vector[:5].detach().numpy()} ...")
            
        return user_vectors
    
    def embedding_similarity_example(self):
        """嵌入向量相似度计算示例"""
        print(f"\n📊 嵌入向量相似度计算")
        print("=" * 60)
        
        # 选择几个用户
        user_ids = torch.tensor([10, 11, 50, 100])
        user_vectors = self.user_embedding(user_ids)
        
        print(f"计算用户间的余弦相似度：")
        print(f"用户IDs: {user_ids.tolist()}")
        
        # 计算余弦相似度
        def cosine_similarity(v1, v2):
            return torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        
        print(f"\n相似度矩阵：")
        print("用户ID  ", end="")
        for uid in user_ids:
            print(f"{uid.item():8d}", end="")
        print()
        
        for i, uid1 in enumerate(user_ids):
            print(f"用户{uid1.item():3d}  ", end="")
            for j, uid2 in enumerate(user_ids):
                sim = cosine_similarity(user_vectors[i], user_vectors[j])
                print(f"{sim:8.3f}", end="")
            print()
        
        return user_vectors
    
    def training_example(self):
        """训练过程示例"""
        print(f"\n🚀 训练过程示例")
        print("=" * 60)
        
        # 创建简单的推荐模型
        class SimpleRecommendModel(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                
            def forward(self, user_ids, item_ids):
                user_vectors = self.user_embedding(user_ids)
                item_vectors = self.item_embedding(item_ids)
                
                # 计算用户和物品的相似度得分
                scores = torch.sum(user_vectors * item_vectors, dim=1)
                return scores
        
        # 模型参数
        num_items = 500
        model = SimpleRecommendModel(self.num_users, num_items, self.embedding_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        print(f"模型结构：")
        print(f"  用户数: {self.num_users}, 物品数: {num_items}")
        print(f"  嵌入维度: {self.embedding_dim}")
        print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 生成模拟训练数据
        batch_size = 32
        user_ids = torch.randint(0, self.num_users, (batch_size,))
        item_ids = torch.randint(0, num_items, (batch_size,))
        ratings = torch.rand(batch_size) * 5  # 模拟评分 0-5
        
        print(f"\n训练数据示例：")
        print(f"  批大小: {batch_size}")
        print(f"  用户ID样本: {user_ids[:5].tolist()}")
        print(f"  物品ID样本: {item_ids[:5].tolist()}")
        print(f"  评分样本: {ratings[:5].tolist()}")
        
        # 训练几个步骤
        print(f"\n开始训练...")
        initial_user_embedding = model.user_embedding.weight.data[0].clone()
        
        for epoch in range(5):
            # 前向传播
            predicted_scores = model(user_ids, item_ids)
            loss = criterion(predicted_scores, ratings)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        # 检查参数更新
        final_user_embedding = model.user_embedding.weight.data[0]
        change = torch.norm(final_user_embedding - initial_user_embedding).item()
        
        print(f"\n训练结果：")
        print(f"  用户0嵌入向量变化量: {change:.6f}")
        print(f"  训练前: {initial_user_embedding[:5].numpy()}")
        print(f"  训练后: {final_user_embedding[:5].numpy()}")
        
        return model
    
    def practical_application_example(self):
        """实际应用示例"""
        print(f"\n🎯 实际应用示例")
        print("=" * 60)
        
        # 模拟用户行为数据
        np.random.seed(42)
        
        # 生成用户交互数据
        num_interactions = 10000
        user_ids = np.random.randint(0, self.num_users, num_interactions)
        item_ids = np.random.randint(0, 100, num_interactions)  # 100个物品
        ratings = np.random.normal(3.5, 1.0, num_interactions)  # 正态分布评分
        ratings = np.clip(ratings, 1, 5)  # 限制在1-5范围
        
        # 转换为DataFrame
        interactions_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings
        })
        
        print(f"用户交互数据统计：")
        print(f"  交互总数: {len(interactions_df):,}")
        print(f"  用户数: {interactions_df['user_id'].nunique()}")
        print(f"  物品数: {interactions_df['item_id'].nunique()}")
        print(f"  平均评分: {interactions_df['rating'].mean():.2f}")
        
        # 展示数据样本
        print(f"\n数据样本：")
        print(interactions_df.head(10))
        
        # 获取特定用户的嵌入向量
        target_user_id = 42
        user_vector = self.user_embedding(torch.tensor([target_user_id]))
        
        print(f"\n用户{target_user_id}的嵌入向量：")
        print(f"  向量维度: {user_vector.shape}")
        print(f"  向量预览: {user_vector[0][:10].detach().numpy()}")
        
        # 计算该用户与其他用户的相似度
        sample_users = torch.tensor([10, 20, 30, 40, 50])
        sample_vectors = self.user_embedding(sample_users)
        
        similarities = torch.cosine_similarity(
            user_vector.expand_as(sample_vectors), 
            sample_vectors
        )
        
        print(f"\n用户{target_user_id}与其他用户的相似度：")
        for i, uid in enumerate(sample_users):
            print(f"  用户{uid.item():2d}: {similarities[i].item():.4f}")
        
        return interactions_df, user_vector
    
    def save_and_load_example(self):
        """保存和加载嵌入向量示例"""
        print(f"\n💾 保存和加载嵌入向量")
        print("=" * 60)
        
        # 保存嵌入向量
        torch.save(self.user_embedding.state_dict(), 'user_embedding.pth')
        print(f"✅ 嵌入向量已保存到 'user_embedding.pth'")
        
        # 创建新的嵌入层并加载参数
        new_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        new_embedding.load_state_dict(torch.load('user_embedding.pth'))
        
        print(f"✅ 嵌入向量已加载到新的嵌入层")
        
        # 验证加载是否正确
        test_user_id = torch.tensor([100])
        original_vector = self.user_embedding(test_user_id)
        loaded_vector = new_embedding(test_user_id)
        
        is_same = torch.allclose(original_vector, loaded_vector)
        print(f"✅ 验证加载结果: {'成功' if is_same else '失败'}")
        
        # 导出为numpy数组
        embedding_matrix = self.user_embedding.weight.data.numpy()
        np.save('user_embedding_matrix.npy', embedding_matrix)
        
        print(f"✅ 嵌入矩阵已导出为numpy数组")
        print(f"   文件大小: {embedding_matrix.nbytes / (1024*1024):.2f} MB")
        
        return embedding_matrix
    
    def visualization_example(self):
        """可视化示例"""
        print(f"\n📊 嵌入向量可视化")
        print("=" * 60)
        
        # 选择一些用户进行可视化
        user_ids = torch.tensor([0, 1, 2, 3, 4, 50, 100, 200, 500, 999])
        user_vectors = self.user_embedding(user_ids)
        
        # 计算相似度矩阵
        similarity_matrix = torch.cosine_similarity(
            user_vectors.unsqueeze(1), 
            user_vectors.unsqueeze(0), 
            dim=2
        )
        
        # 绘制相似度热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix.detach().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar(label='余弦相似度')
        plt.title('用户嵌入向量相似度矩阵')
        plt.xlabel('用户索引')
        plt.ylabel('用户索引')
        
        # 设置坐标轴标签
        user_labels = [f'User{uid.item()}' for uid in user_ids]
        plt.xticks(range(len(user_ids)), user_labels, rotation=45)
        plt.yticks(range(len(user_ids)), user_labels)
        
        plt.tight_layout()
        plt.savefig('user_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 相似度热力图已保存为 'user_similarity_heatmap.png'")
        
        # 显示嵌入向量的统计信息
        print(f"\n嵌入向量统计信息：")
        all_embeddings = self.user_embedding.weight.data
        print(f"  向量范数均值: {torch.norm(all_embeddings, dim=1).mean():.4f}")
        print(f"  向量范数标准差: {torch.norm(all_embeddings, dim=1).std():.4f}")
        print(f"  向量元素均值: {all_embeddings.mean():.6f}")
        print(f"  向量元素标准差: {all_embeddings.std():.6f}")
        
        return similarity_matrix

def main():
    """主函数"""
    print("🚀 PyTorch用户ID嵌入向量生成示例")
    print("=" * 80)
    
    # 创建演示实例
    demo = UserEmbeddingDemo(num_users=1000, embedding_dim=64)
    
    # 1. 基础嵌入向量生成
    demo.basic_embedding_example()
    
    # 2. 相似度计算
    demo.embedding_similarity_example()
    
    # 3. 训练过程演示
    demo.training_example()
    
    # 4. 实际应用示例
    demo.practical_application_example()
    
    # 5. 保存和加载
    demo.save_and_load_example()
    
    # 6. 可视化
    demo.visualization_example()
    
    print(f"\n" + "=" * 80)
    print("🎯 关键代码总结:")
    print("""
    # 1. 创建嵌入层
    user_embedding = nn.Embedding(num_users, embedding_dim)
    
    # 2. 获取嵌入向量
    user_ids = torch.tensor([10, 20, 30])
    user_vectors = user_embedding(user_ids)
    
    # 3. 在训练中使用
    optimizer = optim.Adam(user_embedding.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()
    
    # 4. 保存和加载
    torch.save(user_embedding.state_dict(), 'embedding.pth')
    user_embedding.load_state_dict(torch.load('embedding.pth'))
    """)
    print("=" * 80)

if __name__ == "__main__":
    main() 