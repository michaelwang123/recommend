import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import faiss
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionRecommenderDataset(Dataset):
    """生产环境数据集"""
    
    def __init__(self, user_ids, item_ids, ratings, user_features=None, item_features=None):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
        self.user_features = torch.FloatTensor(user_features) if user_features is not None else None
        self.item_features = torch.FloatTensor(item_features) if item_features is not None else None
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        sample = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'rating': self.ratings[idx]
        }
        
        if self.user_features is not None:
            sample['user_features'] = self.user_features[idx]
        if self.item_features is not None:
            sample['item_features'] = self.item_features[idx]
            
        return sample

class ProductionRecommender(nn.Module):
    """生产级推荐系统模型"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, 
                 use_features=False, user_feature_dim=0, item_feature_dim=0):
        super(ProductionRecommender, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_features = use_features
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 偏置项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 特征处理
        if use_features:
            self.user_feature_linear = nn.Linear(user_feature_dim, embedding_dim)
            self.item_feature_linear = nn.Linear(item_feature_dim, embedding_dim)
        
        # 深度网络
        self.deep_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
        # 初始化深度网络权重
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, item_ids, user_features=None, item_features=None):
        """前向传播"""
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 如果使用特征，融合特征信息
        if self.use_features and user_features is not None:
            user_feat_emb = self.user_feature_linear(user_features)
            user_emb = user_emb + user_feat_emb
        
        if self.use_features and item_features is not None:
            item_feat_emb = self.item_feature_linear(item_features)
            item_emb = item_emb + item_feat_emb
        
        # 计算偏置项
        user_bias = self.user_bias(user_ids).squeeze(-1)
        item_bias = self.item_bias(item_ids).squeeze(-1)
        
        # 矩阵分解部分
        mf_output = torch.sum(user_emb * item_emb, dim=1)
        
        # 深度学习部分
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        deep_output = self.deep_layers(deep_input).squeeze(-1)
        
        # 最终预测
        prediction = mf_output + deep_output + user_bias + item_bias + self.global_bias
        
        return prediction
    
    def get_item_embeddings(self):
        """获取物品嵌入"""
        return self.item_embedding.weight.detach()
    
    def get_user_embeddings(self):
        """获取用户嵌入"""
        return self.user_embedding.weight.detach()

class SimilarityEngine:
    """相似性计算引擎"""
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.index = None
        self.item_embeddings = None
        self.item_mapping = None
    
    def build_index(self, item_embeddings, item_mapping):
        """构建FAISS索引"""
        logger.info("构建FAISS索引...")
        
        self.item_embeddings = item_embeddings.numpy()
        self.item_mapping = item_mapping
        
        # 创建FAISS索引
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积索引
        
        # 标准化嵌入
        faiss.normalize_L2(self.item_embeddings)
        self.index.add(self.item_embeddings)
        
        logger.info(f"索引构建完成，包含 {self.index.ntotal} 个物品")
    
    def find_similar_items(self, item_id, top_k=10):
        """查找相似物品"""
        if self.index is None:
            raise ValueError("索引未构建")
        
        if item_id not in self.item_mapping:
            return [], []
        
        # 获取物品嵌入
        item_idx = self.item_mapping[item_id]
        query_embedding = self.item_embeddings[item_idx:item_idx+1]
        
        # 搜索相似物品
        similarities, indices = self.index.search(query_embedding, top_k + 1)
        
        # 排除自己
        similar_items = []
        similar_scores = []
        
        reverse_mapping = {v: k for k, v in self.item_mapping.items()}
        
        for i, (idx, score) in enumerate(zip(indices[0], similarities[0])):
            if idx != item_idx:  # 排除自己
                similar_items.append(reverse_mapping[idx])
                similar_scores.append(float(score))
        
        return similar_items[:top_k], similar_scores[:top_k]

class RecommenderSystem:
    """完整的推荐系统"""
    
    def __init__(self, model_path="recommender_model.pth"):
        self.model = None
        self.similarity_engine = SimilarityEngine()
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False
        
        # 元数据
        self.item_metadata = {}
        self.user_metadata = {}
    
    def prepare_data(self, df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        # 编码用户和物品ID
        df['user_encoded'] = self.user_encoder.fit_transform(df[user_col])
        df['item_encoded'] = self.item_encoder.fit_transform(df[item_col])
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        logger.info(f"用户数量: {n_users}, 物品数量: {n_items}")
        
        return df, n_users, n_items
    
    def train(self, df, epochs=50, batch_size=256, learning_rate=0.001,
              validation_split=0.2, use_features=False):
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 准备数据
        df, n_users, n_items = self.prepare_data(df)
        
        # 划分训练集和验证集
        train_df, val_df = train_test_split(df, test_size=validation_split, random_state=42)
        
        # 创建数据集
        train_dataset = ProductionRecommenderDataset(
            train_df['user_encoded'].values,
            train_df['item_encoded'].values,
            train_df['rating'].values
        )
        
        val_dataset = ProductionRecommenderDataset(
            val_df['user_encoded'].values,
            val_df['item_encoded'].values,
            val_df['rating'].values
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        self.model = ProductionRecommender(n_users, n_items, use_features=use_features)
        
        # 训练设置
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        # 训练循环
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                predictions = self.model(batch['user_id'], batch['item_id'])
                loss = criterion(predictions, batch['rating'])
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    predictions = self.model(batch['user_id'], batch['item_id'])
                    loss = criterion(predictions, batch['rating'])
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 构建相似性索引
        self.build_similarity_index()
        
        self.is_trained = True
        logger.info("模型训练完成!")
        
        return train_losses, val_losses
    
    def build_similarity_index(self):
        """构建相似性索引"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 获取物品嵌入
        item_embeddings = self.model.get_item_embeddings()
        
        # 创建物品映射
        item_mapping = {item_id: idx for idx, item_id in enumerate(self.item_encoder.classes_)}
        
        # 构建索引
        self.similarity_engine.build_index(item_embeddings, item_mapping)
    
    def predict_rating(self, user_id, item_id):
        """预测用户对物品的评分"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 编码用户和物品ID
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
            item_encoded = self.item_encoder.transform([item_id])[0]
        except ValueError:
            return 0.0  # 未知用户或物品
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_encoded])
            item_tensor = torch.LongTensor([item_encoded])
            prediction = self.model(user_tensor, item_tensor)
        
        return prediction.item()
    
    def recommend_similar_items(self, item_id, top_k=10):
        """推荐相似物品"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 编码物品ID
        try:
            item_encoded = self.item_encoder.transform([item_id])[0]
        except ValueError:
            return [], []
        
        # 获取相似物品
        similar_items, scores = self.similarity_engine.find_similar_items(item_encoded, top_k)
        
        # 解码物品ID
        decoded_items = []
        for item_idx in similar_items:
            decoded_items.append(self.item_encoder.inverse_transform([item_idx])[0])
        
        return decoded_items, scores
    
    def recommend_for_user(self, user_id, top_k=10, exclude_seen=True):
        """为用户推荐物品"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 编码用户ID
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except ValueError:
            return [], []
        
        # 获取所有物品
        all_items = list(range(len(self.item_encoder.classes_)))
        
        # 预测评分
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_encoded] * len(all_items))
            item_tensor = torch.LongTensor(all_items)
            
            # 批量预测
            batch_size = 1000
            all_predictions = []
            
            for i in range(0, len(all_items), batch_size):
                batch_user = user_tensor[i:i+batch_size]
                batch_item = item_tensor[i:i+batch_size]
                batch_pred = self.model(batch_user, batch_item)
                all_predictions.extend(batch_pred.tolist())
            
            predictions = all_predictions
        
        # 排序并返回top_k
        item_scores = list(zip(all_items, predictions))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 解码物品ID
        recommended_items = []
        recommended_scores = []
        
        for item_idx, score in item_scores[:top_k]:
            item_id = self.item_encoder.inverse_transform([item_idx])[0]
            recommended_items.append(item_id)
            recommended_scores.append(score)
        
        return recommended_items, recommended_scores
    
    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = self.model_path
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'model_config': {
                'n_users': self.model.n_users,
                'n_items': self.model.n_items,
                'embedding_dim': self.model.embedding_dim,
                'use_features': self.model.use_features
            }
        }
        
        torch.save(save_dict, path)
        logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path=None):
        """加载模型"""
        if path is None:
            path = self.model_path
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # 重建模型
        config = checkpoint['model_config']
        self.model = ProductionRecommender(
            config['n_users'], config['n_items'], 
            config['embedding_dim'], config['use_features']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.user_encoder = checkpoint['user_encoder']
        self.item_encoder = checkpoint['item_encoder']
        
        # 重建相似性索引
        self.build_similarity_index()
        
        self.is_trained = True
        logger.info(f"模型已从 {path} 加载")

def generate_sample_ecommerce_data(n_users=1000, n_items=500, n_interactions=10000):
    """生成示例电商数据"""
    np.random.seed(42)
    
    # 生成用户-物品交互数据
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    
    # 生成更真实的评分分布
    ratings = np.random.choice([1, 2, 3, 4, 5], n_interactions, 
                              p=[0.1, 0.1, 0.2, 0.3, 0.3])
    
    # 添加一些噪声和用户偏好
    for i in range(len(ratings)):
        user_bias = np.random.normal(0, 0.5)
        item_bias = np.random.normal(0, 0.3)
        ratings[i] = max(1, min(5, ratings[i] + user_bias + item_bias))
    
    # 创建DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': pd.date_range('2023-01-01', periods=n_interactions, freq='1H')
    })
    
    return df

def main():
    """主函数"""
    logger.info("=== 生产级推荐系统示例 ===")
    
    # 生成示例数据
    df = generate_sample_ecommerce_data(n_users=1000, n_items=500, n_interactions=10000)
    logger.info(f"生成数据: {len(df)} 条交互记录")
    
    # 创建推荐系统
    recommender = RecommenderSystem()
    
    # 训练模型
    train_losses, val_losses = recommender.train(
        df, epochs=30, batch_size=512, learning_rate=0.001
    )
    
    # 保存模型
    recommender.save_model()
    
    # 测试推荐
    test_user = df['user_id'].iloc[0]
    test_item = df['item_id'].iloc[0]
    
    logger.info(f"\n=== 推荐测试 ===")
    
    # 预测评分
    predicted_rating = recommender.predict_rating(test_user, test_item)
    logger.info(f"用户 {test_user} 对物品 {test_item} 的预测评分: {predicted_rating:.2f}")
    
    # 推荐相似物品
    similar_items, scores = recommender.recommend_similar_items(test_item, top_k=5)
    logger.info(f"与物品 {test_item} 相似的物品:")
    for item, score in zip(similar_items, scores):
        logger.info(f"  物品 {item}: {score:.4f}")
    
    # 为用户推荐物品
    recommended_items, rec_scores = recommender.recommend_for_user(test_user, top_k=5)
    logger.info(f"为用户 {test_user} 推荐的物品:")
    for item, score in zip(recommended_items, rec_scores):
        logger.info(f"  物品 {item}: {score:.4f}")
    
    # 可视化训练过程
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练过程')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(df['rating'], bins=5, alpha=0.7, edgecolor='black')
    plt.title('评分分布')
    plt.xlabel('评分')
    plt.ylabel('频次')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('production_recommender_results.png')
    plt.show()
    
    logger.info("\n生产级推荐系统演示完成!")

if __name__ == "__main__":
    main() 