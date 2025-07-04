import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ContentBasedRecommender:
    """基于内容的推荐系统"""
    
    def __init__(self, items_features):
        """
        初始化推荐器
        Args:
            items_features: 物品特征矩阵 (n_items, n_features)
        """
        self.items_features = torch.FloatTensor(items_features)
        self.similarity_matrix = self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self):
        """计算物品相似性矩阵"""
        # 使用余弦相似度
        normalized_features = F.normalize(self.items_features, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        return similarity_matrix
    
    def recommend(self, item_id, top_k=5):
        """
        为给定物品推荐相似物品
        Args:
            item_id: 物品ID
            top_k: 推荐数量
        Returns:
            推荐的物品ID列表和相似度分数
        """
        similarities = self.similarity_matrix[item_id]
        # 排除自己
        similarities[item_id] = -1
        
        # 获取top_k个最相似的物品
        top_k_indices = torch.topk(similarities, top_k).indices
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()

class CollaborativeFilteringRecommender(nn.Module):
    """基于协同过滤的推荐系统"""
    
    def __init__(self, n_users, n_items, n_factors=50):
        super(CollaborativeFilteringRecommender, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # 偏置项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
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
        """前向传播"""
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 获取偏置
        user_bias = self.user_bias(user_ids).squeeze()
        item_bias = self.item_bias(item_ids).squeeze()
        
        # 计算评分
        dot_product = torch.sum(user_emb * item_emb, dim=1)
        prediction = dot_product + user_bias + item_bias + self.global_bias
        
        return prediction
    
    def compute_item_similarity(self):
        """计算物品相似性矩阵"""
        item_embeddings = self.item_embedding.weight
        normalized_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        return similarity_matrix
    
    def recommend_similar_items(self, item_id, top_k=5):
        """推荐相似物品"""
        similarity_matrix = self.compute_item_similarity()
        similarities = similarity_matrix[item_id]
        similarities[item_id] = -1  # 排除自己
        
        top_k_indices = torch.topk(similarities, top_k).indices
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()

# 示例数据生成和使用
def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    
    # 生成用户-物品评分数据
    n_users, n_items = 100, 50
    n_interactions = 1000
    
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.randint(1, 6, n_interactions)
    
    # 生成物品特征数据
    n_features = 10
    item_features = np.random.randn(n_items, n_features)
    
    return user_ids, item_ids, ratings, item_features

def train_collaborative_filtering(user_ids, item_ids, ratings, n_users, n_items, epochs=100):
    """训练协同过滤模型"""
    model = CollaborativeFilteringRecommender(n_users, n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 转换数据
    user_ids_tensor = torch.LongTensor(user_ids)
    item_ids_tensor = torch.LongTensor(item_ids)
    ratings_tensor = torch.FloatTensor(ratings)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(user_ids_tensor, item_ids_tensor)
        loss = criterion(predictions, ratings_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model, losses

def main():
    """主函数示例"""
    print("=== PyTorch相似性推荐系统示例 ===")
    
    # 生成示例数据
    user_ids, item_ids, ratings, item_features = generate_sample_data()
    n_users, n_items = 100, 50
    
    print(f"数据规模: {len(user_ids)} 交互, {n_users} 用户, {n_items} 物品")
    
    # 1. 基于内容的推荐
    print("\n1. 基于内容的推荐:")
    content_recommender = ContentBasedRecommender(item_features)
    similar_items, scores = content_recommender.recommend(item_id=0, top_k=5)
    print(f"与物品0相似的物品: {similar_items}")
    print(f"相似度分数: {[f'{score:.3f}' for score in scores]}")
    
    # 2. 协同过滤推荐
    print("\n2. 协同过滤推荐:")
    cf_model, losses = train_collaborative_filtering(
        user_ids, item_ids, ratings, n_users, n_items, epochs=100
    )
    
    # 推荐相似物品
    similar_items_cf, scores_cf = cf_model.recommend_similar_items(item_id=0, top_k=5)
    print(f"协同过滤推荐的相似物品: {similar_items_cf}")
    print(f"相似度分数: {[f'{score:.3f}' for score in scores_cf]}")
    
    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('协同过滤训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    print("\n训练完成!")

if __name__ == "__main__":
    main() 