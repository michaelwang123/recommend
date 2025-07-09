#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有意义的用户ID嵌入向量训练
通过用户行为数据学习用户偏好的嵌入向量
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MeaningfulUserEmbedding:
    def __init__(self):
        self.num_users = 1000
        self.num_items = 500
        self.embedding_dim = 64
        
    def generate_user_behavior_data(self):
        """生成模拟的用户行为数据"""
        print("📊 生成用户行为数据...")
        
        # 设置随机种子以获得可重复的结果
        np.random.seed(42)
        
        # 生成用户-物品交互数据
        num_interactions = 50000
        
        # 创建一些有意义的用户群体
        # 用户0-299: 喜欢科技产品 (物品0-199)
        # 用户300-599: 喜欢时尚产品 (物品200-399)  
        # 用户600-999: 喜欢运动产品 (物品300-499)
        
        interactions = []
        
        # 科技爱好者
        for _ in range(20000):
            user_id = np.random.randint(0, 300)
            item_id = np.random.randint(0, 200)  # 偏好科技产品
            rating = np.random.normal(4.0, 0.8)  # 高评分
            rating = np.clip(rating, 1, 5)
            interactions.append([user_id, item_id, rating])
        
        # 时尚爱好者
        for _ in range(20000):
            user_id = np.random.randint(300, 600)
            item_id = np.random.randint(200, 400)  # 偏好时尚产品
            rating = np.random.normal(4.2, 0.7)  # 高评分
            rating = np.clip(rating, 1, 5)
            interactions.append([user_id, item_id, rating])
        
        # 运动爱好者
        for _ in range(10000):
            user_id = np.random.randint(600, 1000)
            item_id = np.random.randint(300, 500)  # 偏好运动产品
            rating = np.random.normal(3.8, 0.9)  # 较高评分
            rating = np.clip(rating, 1, 5)
            interactions.append([user_id, item_id, rating])
        
        # 创建DataFrame
        df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])
        
        print(f"✅ 生成了 {len(df)} 条用户行为记录")
        print(f"   用户数: {df['user_id'].nunique()}")
        print(f"   物品数: {df['item_id'].nunique()}")
        print(f"   平均评分: {df['rating'].mean():.2f}")
        
        # 显示不同用户群体的偏好
        print(f"\n用户群体分析:")
        print(f"  科技爱好者 (用户0-299): 主要购买物品0-199")
        print(f"  时尚爱好者 (用户300-599): 主要购买物品200-399")
        print(f"  运动爱好者 (用户600-999): 主要购买物品300-499")
        
        return df
    
    def create_recommendation_model(self):
        """创建推荐模型"""
        class MatrixFactorization(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                # 用户嵌入层 - 这里的嵌入向量会学习用户偏好
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                # 物品嵌入层 - 学习物品特征
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                
                # 偏置项
                self.user_bias = nn.Embedding(num_users, 1)
                self.item_bias = nn.Embedding(num_items, 1)
                self.global_bias = nn.Parameter(torch.zeros(1))
                
                # 初始化参数
                self._init_weights()
            
            def _init_weights(self):
                """初始化权重"""
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)
                nn.init.normal_(self.user_bias.weight, std=0.1)
                nn.init.normal_(self.item_bias.weight, std=0.1)
            
            def forward(self, user_ids, item_ids):
                # 获取用户和物品的嵌入向量
                user_vectors = self.user_embedding(user_ids)
                item_vectors = self.item_embedding(item_ids)
                
                # 计算用户和物品的相似度
                interaction = torch.sum(user_vectors * item_vectors, dim=1)
                
                # 添加偏置项
                user_bias = self.user_bias(user_ids).squeeze()
                item_bias = self.item_bias(item_ids).squeeze()
                
                # 最终预测评分
                prediction = interaction + user_bias + item_bias + self.global_bias
                
                return prediction
        
        return MatrixFactorization(self.num_users, self.num_items, self.embedding_dim)
    
    def train_meaningful_embeddings(self, df):
        """训练有意义的用户嵌入向量"""
        print(f"\n🚀 开始训练有意义的用户嵌入向量...")
        
        # 准备训练数据
        user_ids = torch.tensor(df['user_id'].values, dtype=torch.long)
        item_ids = torch.tensor(df['item_id'].values, dtype=torch.long)
        ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        
        # 分割训练和验证集
        train_indices, val_indices = train_test_split(
            range(len(df)), test_size=0.2, random_state=42
        )
        
        # 创建模型
        model = self.create_recommendation_model()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        print(f"模型参数:")
        print(f"  用户嵌入维度: {self.embedding_dim}")
        print(f"  物品嵌入维度: {self.embedding_dim}")
        print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练过程
        model.train()
        for epoch in range(50):
            # 训练
            train_user_ids = user_ids[train_indices]
            train_item_ids = item_ids[train_indices]
            train_ratings = ratings[train_indices]
            
            optimizer.zero_grad()
            predictions = model(train_user_ids, train_item_ids)
            loss = criterion(predictions, train_ratings)
            loss.backward()
            optimizer.step()
            
            # 验证
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_user_ids = user_ids[val_indices]
                    val_item_ids = item_ids[val_indices]
                    val_ratings = ratings[val_indices]
                    
                    val_predictions = model(val_user_ids, val_item_ids)
                    val_loss = criterion(val_predictions, val_ratings)
                    
                print(f"  Epoch {epoch:2d}: 训练损失={loss.item():.4f}, 验证损失={val_loss.item():.4f}")
                model.train()
        
        return model
    
    def analyze_learned_embeddings(self, model, df):
        """分析学习到的用户嵌入向量"""
        print(f"\n🔍 分析学习到的用户嵌入向量...")
        
        # 获取所有用户的嵌入向量
        all_user_ids = torch.arange(self.num_users)
        user_embeddings = model.user_embedding(all_user_ids)
        
        # 分析不同用户群体的相似性
        print(f"\n用户群体内部相似性分析:")
        
        # 科技爱好者 (用户0-299)
        tech_users = torch.arange(0, 300)
        tech_embeddings = model.user_embedding(tech_users)
        tech_similarities = torch.cosine_similarity(
            tech_embeddings.unsqueeze(1), 
            tech_embeddings.unsqueeze(0), 
            dim=2
        )
        # 排除对角线（自己和自己的相似度）
        tech_avg_sim = (tech_similarities.sum() - tech_similarities.trace()) / (300 * 299)
        print(f"  科技爱好者群体内平均相似度: {tech_avg_sim.item():.4f}")
        
        # 时尚爱好者 (用户300-599)
        fashion_users = torch.arange(300, 600)
        fashion_embeddings = model.user_embedding(fashion_users)
        fashion_similarities = torch.cosine_similarity(
            fashion_embeddings.unsqueeze(1), 
            fashion_embeddings.unsqueeze(0), 
            dim=2
        )
        fashion_avg_sim = (fashion_similarities.sum() - fashion_similarities.trace()) / (300 * 299)
        print(f"  时尚爱好者群体内平均相似度: {fashion_avg_sim.item():.4f}")
        
        # 运动爱好者 (用户600-999)
        sport_users = torch.arange(600, 1000)
        sport_embeddings = model.user_embedding(sport_users)
        sport_similarities = torch.cosine_similarity(
            sport_embeddings.unsqueeze(1), 
            sport_embeddings.unsqueeze(0), 
            dim=2
        )
        sport_avg_sim = (sport_similarities.sum() - sport_similarities.trace()) / (400 * 399)
        print(f"  运动爱好者群体内平均相似度: {sport_avg_sim.item():.4f}")
        
        # 分析跨群体相似性
        print(f"\n跨群体相似性分析:")
        
        # 科技 vs 时尚
        tech_fashion_sim = torch.cosine_similarity(
            tech_embeddings.mean(dim=0), 
            fashion_embeddings.mean(dim=0), 
            dim=0
        )
        print(f"  科技爱好者 vs 时尚爱好者: {tech_fashion_sim.item():.4f}")
        
        # 科技 vs 运动
        tech_sport_sim = torch.cosine_similarity(
            tech_embeddings.mean(dim=0), 
            sport_embeddings.mean(dim=0), 
            dim=0
        )
        print(f"  科技爱好者 vs 运动爱好者: {tech_sport_sim.item():.4f}")
        
        # 时尚 vs 运动
        fashion_sport_sim = torch.cosine_similarity(
            fashion_embeddings.mean(dim=0), 
            sport_embeddings.mean(dim=0), 
            dim=0
        )
        print(f"  时尚爱好者 vs 运动爱好者: {fashion_sport_sim.item():.4f}")
        
        return user_embeddings
    
    def demonstrate_recommendations(self, model, df):
        """演示推荐效果"""
        print(f"\n🎯 演示推荐效果...")
        
        # 选择不同群体的代表用户
        test_users = [50, 350, 650]  # 科技、时尚、运动爱好者各一个
        user_names = ["科技爱好者", "时尚爱好者", "运动爱好者"]
        
        for user_id, user_name in zip(test_users, user_names):
            print(f"\n👤 用户{user_id} ({user_name}):")
            
            # 获取该用户的历史行为
            #user_history = df[df['user_id'] == user_id]
            user_history = df.query('user_id == @user_id')
            if len(user_history) > 0:
                print(f"   历史行为: 对物品 {user_history['item_id'].tolist()[:5]} 的评分")
            
            # 为该用户推荐物品
            user_tensor = torch.tensor([user_id])
            all_items = torch.arange(self.num_items)
            
            # 计算对所有物品的预测评分
            with torch.no_grad():
                user_repeated = user_tensor.repeat(self.num_items)
                predictions = model(user_repeated, all_items)
            
            # 获取评分最高的物品
            top_items = torch.topk(predictions, k=5).indices.tolist()
            top_scores = torch.topk(predictions, k=5).values.tolist()
            
            print(f"   推荐物品: {top_items}")
            print(f"   预测评分: {[f'{score:.2f}' for score in top_scores]}")
        
        return True

def main():
    """主函数"""
    print("🎓 有意义的用户ID嵌入向量训练示例")
    print("=" * 80)
    
    # 创建实例
    demo = MeaningfulUserEmbedding()
    
    # 1. 生成用户行为数据
    df = demo.generate_user_behavior_data()
    
    # 2. 训练有意义的嵌入向量
    model = demo.train_meaningful_embeddings(df)
    
    # 3. 分析学习到的嵌入向量
    user_embeddings = demo.analyze_learned_embeddings(model, df)
    
    # 4. 演示推荐效果
    demo.demonstrate_recommendations(model, df)
    
    print(f"\n" + "=" * 80)
    print("🎯 关键洞察:")
    print("• 用户ID嵌入向量只有通过用户行为数据训练才有意义")
    print("• 相似偏好的用户会有相似的嵌入向量")
    print("• 嵌入向量捕捉了用户的潜在偏好特征")
    print("• 训练后的嵌入向量可以用于个性化推荐")
    print("• 仅仅基于ID的随机嵌入向量没有实际意义")
    print("=" * 80)

if __name__ == "__main__":
    main() 