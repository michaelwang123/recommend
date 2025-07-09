#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于MySQL数据的推荐系统
从MySQL中获取非连续用户ID和物品ID数据，训练推荐模型
"""

import mysql.connector
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json
from datetime import datetime
import os

class MySQLRecommendationSystem:
    def __init__(self, db_config=None):
        """初始化推荐系统"""
        # 默认数据库配置
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 3306,
            'user': 'test',
            'password': 'test',
            'database': 'testdb'
        }
        
        # 模型参数
        self.embedding_dim = 64
        self.model = None
        
        # ID映射字典
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}
        
        # 数据统计
        self.num_users = 0
        self.num_items = 0
        
        print("🚀 MySQL推荐系统初始化完成")
    
    def connect_to_database(self):
        """连接到MySQL数据库"""
        try:
            print("🔌 连接到MySQL数据库...")
            connection = mysql.connector.connect(**self.db_config)
            print("✅ 数据库连接成功")
            return connection
        except mysql.connector.Error as err:
            print(f"❌ 数据库连接失败: {err}")
            return None
    
    def load_data_from_mysql(self):
        """从MySQL加载用户行为数据"""
        print("📊 从MySQL加载用户行为数据...")
        
        connection = self.connect_to_database()
        if not connection:
            return None
        
        try:
            # 执行查询
            query = """
            SELECT user_id, item_id, rating, user_type, item_category, created_at
            FROM user_behavior
            ORDER BY created_at
            """
            
            df = pd.read_sql(query, connection)
            
            print(f"✅ 成功加载数据: {len(df)} 条记录")
            print(f"   唯一用户数: {df['user_id'].nunique()}")
            print(f"   唯一物品数: {df['item_id'].nunique()}")
            print(f"   评分范围: {df['rating'].min():.2f} - {df['rating'].max():.2f}")
            print(f"   平均评分: {df['rating'].mean():.2f}")
            
            # 显示数据类型分布
            print(f"\n用户类型分布:")
            user_type_stats = df.groupby('user_type').agg({
                'user_id': 'count',
                'rating': 'mean'
            }).round(2)
            user_type_stats.columns = ['记录数', '平均评分']
            print(user_type_stats)
            
            return df
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None
        
        finally:
            connection.close()
            print("🔌 数据库连接已关闭")
    
    def create_id_mappings(self, df):
        """创建ID映射（非连续ID → 连续索引）"""
        print("\n🔄 创建ID映射...")
        
        # 获取唯一ID并排序
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())
        
        # 创建映射字典
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        # 创建反向映射
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_idx.items()}
        
        # 更新数量
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        print(f"✅ ID映射创建完成:")
        print(f"   用户映射: {self.num_users} 个用户")
        print(f"   物品映射: {self.num_items} 个物品")
        
        # 显示映射示例
        print(f"\n📝 映射示例:")
        sample_users = list(self.user_id_to_idx.items())[:5]
        sample_items = list(self.item_id_to_idx.items())[:5]
        
        for user_id, idx in sample_users:
            print(f"   用户 {user_id} → 索引 {idx}")
        
        for item_id, idx in sample_items:
            print(f"   物品 {item_id} → 索引 {idx}")
        
        return True
    
    def prepare_training_data(self, df):
        """准备训练数据（转换为连续索引）"""
        print("\n📋 准备训练数据...")
        
        # 转换ID为索引
        df['user_idx'] = df['user_id'].map(self.user_id_to_idx)
        df['item_idx'] = df['item_id'].map(self.item_id_to_idx)
        
        # 检查是否有映射失败的数据
        missing_users = df[df['user_idx'].isna()]
        missing_items = df[df['item_idx'].isna()]
        
        if len(missing_users) > 0:
            print(f"⚠️  发现 {len(missing_users)} 条用户ID映射失败的记录")
        
        if len(missing_items) > 0:
            print(f"⚠️  发现 {len(missing_items)} 条物品ID映射失败的记录")
        
        # 删除映射失败的记录
        df_clean = df.dropna(subset=['user_idx', 'item_idx'])
        
        print(f"✅ 训练数据准备完成:")
        print(f"   有效记录数: {len(df_clean)}")
        print(f"   用户索引范围: 0 - {df_clean['user_idx'].max()}")
        print(f"   物品索引范围: 0 - {df_clean['item_idx'].max()}")
        
        return df_clean
    
    def create_recommendation_model(self):
        """创建推荐模型（矩阵分解）"""
        class MatrixFactorization(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                # 用户嵌入层
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                # 物品嵌入层
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                
                # 偏置项
                self.user_bias = nn.Embedding(num_users, 1)
                self.item_bias = nn.Embedding(num_items, 1)
                self.global_bias = nn.Parameter(torch.zeros(1))
                
                # 初始化权重
                self._init_weights()
            
            def _init_weights(self):
                """初始化权重"""
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)
                nn.init.normal_(self.user_bias.weight, std=0.1)
                nn.init.normal_(self.item_bias.weight, std=0.1)
            
            def forward(self, user_ids, item_ids):
                # 获取嵌入向量
                user_vectors = self.user_embedding(user_ids)
                item_vectors = self.item_embedding(item_ids)
                
                # 计算交互得分
                interaction = torch.sum(user_vectors * item_vectors, dim=1)
                
                # 添加偏置项
                user_bias = self.user_bias(user_ids).squeeze()
                item_bias = self.item_bias(item_ids).squeeze()
                
                # 最终预测
                prediction = interaction + user_bias + item_bias + self.global_bias
                
                return prediction
        
        return MatrixFactorization(self.num_users, self.num_items, self.embedding_dim)
    
    def train_model(self, df_train, epochs=100, learning_rate=0.01, test_size=0.2):
        """训练推荐模型"""
        print(f"\n🚀 开始训练推荐模型...")
        
        # 准备张量数据
        user_ids = torch.tensor(df_train['user_idx'].values, dtype=torch.long)
        item_ids = torch.tensor(df_train['item_idx'].values, dtype=torch.long)
        ratings = torch.tensor(df_train['rating'].values, dtype=torch.float32)
        
        # 分割训练和验证集
        train_indices, val_indices = train_test_split(
            range(len(df_train)), test_size=test_size, random_state=42
        )
        
        # 创建模型
        self.model = self.create_recommendation_model()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        print(f"模型参数:")
        print(f"  用户数量: {self.num_users}")
        print(f"  物品数量: {self.num_items}")
        print(f"  嵌入维度: {self.embedding_dim}")
        print(f"  总参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  训练集大小: {len(train_indices)}")
        print(f"  验证集大小: {len(val_indices)}")
        
        # 训练循环
        train_losses = []
        val_losses = []
        
        self.model.train()
        for epoch in range(epochs):
            # 训练阶段
            train_user_ids = user_ids[train_indices]
            train_item_ids = item_ids[train_indices]
            train_ratings = ratings[train_indices]
            
            optimizer.zero_grad()
            predictions = self.model(train_user_ids, train_item_ids)
            train_loss = criterion(predictions, train_ratings)
            train_loss.backward()
            optimizer.step()
            
            train_losses.append(train_loss.item())
            
            # 验证阶段
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_user_ids = user_ids[val_indices]
                    val_item_ids = item_ids[val_indices]
                    val_ratings = ratings[val_indices]
                    
                    val_predictions = self.model(val_user_ids, val_item_ids)
                    val_loss = criterion(val_predictions, val_ratings)
                    val_losses.append(val_loss.item())
                    
                    print(f"  Epoch {epoch:3d}: 训练损失={train_loss.item():.4f}, 验证损失={val_loss.item():.4f}")
                
                self.model.train()
        
        print(f"✅ 模型训练完成!")
        
        # 最终验证
        self.model.eval()
        with torch.no_grad():
            val_user_ids = user_ids[val_indices]
            val_item_ids = item_ids[val_indices]
            val_ratings = ratings[val_indices]
            
            final_predictions = self.model(val_user_ids, val_item_ids)
            final_loss = criterion(final_predictions, val_ratings)
            
            print(f"  最终验证损失: {final_loss.item():.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_loss': final_loss.item()
        }
    
    def get_user_recommendations(self, user_id, top_n=10, exclude_rated=True, df=None):
        """为指定用户生成推荐"""
        if self.model is None:
            print("❌ 模型尚未训练，请先调用 train_model()")
            return []
        
        # 检查用户ID是否存在
        if user_id not in self.user_id_to_idx:
            print(f"❌ 用户 {user_id} 不存在于训练数据中")
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        
        print(f"\n🎯 为用户 {user_id} (索引:{user_idx}) 生成推荐...")
        
        # 准备数据
        user_tensor = torch.tensor([user_idx])
        all_items = torch.arange(self.num_items)
        
        # 批量预测所有物品的评分
        self.model.eval()
        with torch.no_grad():
            user_repeated = user_tensor.repeat(self.num_items)
            predictions = self.model(user_repeated, all_items)
        
        # 排除已评分的物品
        available_items = list(range(self.num_items))
        
        if exclude_rated and df is not None:
            user_history = df[df['user_id'] == user_id]['item_id'].unique()
            rated_item_indices = [self.item_id_to_idx[item_id] for item_id in user_history 
                                if item_id in self.item_id_to_idx]
            available_items = [idx for idx in available_items if idx not in rated_item_indices]
            print(f"   排除已评分物品: {len(rated_item_indices)} 个")
        
        # 获取可推荐物品的预测评分
        available_predictions = predictions[available_items]
        available_item_indices = torch.tensor(available_items)
        
        # 获取TopN推荐
        top_scores, top_indices = torch.topk(available_predictions, k=min(top_n, len(available_items)))
        top_item_indices = available_item_indices[top_indices]
        
        # 转换回原始物品ID
        recommendations = []
        for i, (item_idx, score) in enumerate(zip(top_item_indices, top_scores)):
            item_id = self.idx_to_item_id[item_idx.item()]
            recommendations.append({
                'rank': i + 1,
                'item_id': item_id,
                'item_idx': item_idx.item(),
                'predicted_rating': score.item()
            })
        
        print(f"✅ 生成了 {len(recommendations)} 个推荐")
        
        return recommendations
    
    def analyze_user_groups(self, df):
        """分析不同用户群体的特征"""
        print(f"\n🔍 分析用户群体特征...")
        
        if self.model is None:
            print("❌ 模型尚未训练")
            return
        
        # 获取所有用户的嵌入向量
        all_user_indices = torch.arange(self.num_users)
        user_embeddings = self.model.user_embedding(all_user_indices)
        
        # 按用户类型分组分析
        user_types = df[['user_id', 'user_type']].drop_duplicates()
        
        print(f"\n用户群体相似性分析:")
        
        for user_type in user_types['user_type'].unique():
            type_users = user_types[user_types['user_type'] == user_type]['user_id'].tolist()
            type_indices = [self.user_id_to_idx[user_id] for user_id in type_users 
                           if user_id in self.user_id_to_idx]
            
            if len(type_indices) > 1:
                type_embeddings = user_embeddings[type_indices]
                
                # 计算群体内相似度
                similarities = torch.cosine_similarity(
                    type_embeddings.unsqueeze(1),
                    type_embeddings.unsqueeze(0),
                    dim=2
                )
                
                # 排除对角线
                mask = torch.eye(len(type_indices), dtype=torch.bool)
                similarities_no_diag = similarities[~mask]
                avg_similarity = similarities_no_diag.mean().item()
                
                print(f"  {user_type}: {len(type_indices)} 个用户, 平均相似度: {avg_similarity:.4f}")
    
    def demonstrate_recommendations(self, df):
        """演示推荐效果"""
        print(f"\n🎯 演示推荐效果...")
        
        # 为每种用户类型选择一个代表用户
        user_types = df[['user_id', 'user_type']].drop_duplicates()
        
        for user_type in user_types['user_type'].unique():
            type_users = user_types[user_types['user_type'] == user_type]['user_id'].tolist()
            if type_users:
                # 选择第一个用户作为代表
                representative_user = type_users[0]
                
                print(f"\n👤 代表用户: {representative_user} ({user_type})")
                
                # 显示用户历史
                user_history = df[df['user_id'] == representative_user]
                print(f"   历史行为: {len(user_history)} 条记录")
                print(f"   平均评分: {user_history['rating'].mean():.2f}")
                print(f"   交互物品: {user_history['item_id'].tolist()[:5]}")
                
                # 生成推荐
                recommendations = self.get_user_recommendations(
                    representative_user, top_n=5, exclude_rated=True, df=df
                )
                
                print(f"   推荐结果:")
                for rec in recommendations:
                    print(f"     {rec['rank']}. {rec['item_id']} - 预测评分: {rec['predicted_rating']:.2f}")
    
    def save_model_and_mappings(self, save_dir="./saved_model"):
        """保存模型和ID映射"""
        if self.model is None:
            print("❌ 没有模型可保存")
            return False
        
        print(f"💾 保存模型到 {save_dir}...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        torch.save(self.model.state_dict(), f"{save_dir}/model.pth")
        
        # 保存映射字典
        mappings = {
            'user_id_to_idx': self.user_id_to_idx,
            'item_id_to_idx': self.item_id_to_idx,
            'idx_to_user_id': self.idx_to_user_id,
            'idx_to_item_id': self.idx_to_item_id,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim
        }
        
        with open(f"{save_dir}/mappings.json", 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config = {
            'model_class': 'MatrixFactorization',
            'embedding_dim': self.embedding_dim,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'save_time': datetime.now().isoformat()
        }
        
        with open(f"{save_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 模型保存成功:")
        print(f"   模型文件: {save_dir}/model.pth")
        print(f"   映射文件: {save_dir}/mappings.json")
        print(f"   配置文件: {save_dir}/config.json")
        
        return True
    
    def load_model_and_mappings(self, save_dir="./saved_model"):
        """加载模型和ID映射"""
        print(f"📂 从 {save_dir} 加载模型...")
        
        try:
            # 加载映射
            with open(f"{save_dir}/mappings.json", 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            self.user_id_to_idx = mappings['user_id_to_idx']
            self.item_id_to_idx = mappings['item_id_to_idx']
            self.idx_to_user_id = {int(k): v for k, v in mappings['idx_to_user_id'].items()}
            self.idx_to_item_id = {int(k): v for k, v in mappings['idx_to_item_id'].items()}
            self.num_users = mappings['num_users']
            self.num_items = mappings['num_items']
            self.embedding_dim = mappings['embedding_dim']
            
            # 创建模型
            self.model = self.create_recommendation_model()
            
            # 加载模型权重
            self.model.load_state_dict(torch.load(f"{save_dir}/model.pth"))
            self.model.eval()
            
            print(f"✅ 模型加载成功:")
            print(f"   用户数量: {self.num_users}")
            print(f"   物品数量: {self.num_items}")
            print(f"   嵌入维度: {self.embedding_dim}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def run_complete_pipeline(self):
        """运行完整的推荐系统流程"""
        print("🚀 运行完整的推荐系统流程")
        print("=" * 80)
        
        try:
            # 1. 从MySQL加载数据
            df = self.load_data_from_mysql()
            if df is None:
                return False
            
            # 2. 创建ID映射
            if not self.create_id_mappings(df):
                return False
            
            # 3. 准备训练数据
            df_train = self.prepare_training_data(df)
            if df_train is None or len(df_train) == 0:
                return False
            
            # 4. 训练模型
            training_result = self.train_model(df_train)
            
            # 5. 分析用户群体
            self.analyze_user_groups(df)
            
            # 6. 演示推荐效果
            self.demonstrate_recommendations(df)
            
            # 7. 保存模型
            self.save_model_and_mappings()
            
            print("\n" + "=" * 80)
            print("🎯 推荐系统训练完成!")
            print("• 成功从MySQL加载非连续ID数据")
            print("• 创建了完整的ID映射机制")
            print("• 训练了矩阵分解推荐模型")
            print("• 分析了用户群体特征")
            print("• 演示了个性化推荐效果")
            print("• 保存了模型和配置文件")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"❌ 流程执行失败: {e}")
            return False

def main():
    """主函数"""
    # 创建推荐系统实例
    recommender = MySQLRecommendationSystem()
    
    # 运行完整流程
    success = recommender.run_complete_pipeline()
    
    if success:
        print("\n✅ 推荐系统构建成功!")
        print("\n🎯 后续可以:")
        print("1. 使用 get_user_recommendations() 为用户生成推荐")
        print("2. 使用 save_model_and_mappings() 保存模型")
        print("3. 使用 load_model_and_mappings() 加载模型")
        print("4. 集成到Web API服务中")
    else:
        print("\n❌ 推荐系统构建失败，请检查错误信息")

if __name__ == "__main__":
    main()
