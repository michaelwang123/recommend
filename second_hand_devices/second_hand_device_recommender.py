import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceFeatureExtractor:
    """二手设备特征提取器"""
    
    def __init__(self):
        self.brand_encoder = LabelEncoder()
        self.model_encoder = LabelEncoder()
        self.condition_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def extract_features(self, device_data):
        """提取设备特征"""
        features = {}
        
        # 基本特征
        features['brand'] = self.brand_encoder.fit_transform(device_data['brand'])
        features['model'] = self.model_encoder.fit_transform(device_data['model'])
        features['category'] = self.category_encoder.fit_transform(device_data['category'])
        features['condition'] = self.condition_encoder.fit_transform(device_data['condition'])
        
        # 数值特征
        numerical_features = ['price', 'age_months', 'storage_gb', 'ram_gb', 'screen_size']
        for feature in numerical_features:
            if feature in device_data.columns:
                features[feature] = device_data[feature].fillna(0).values
        
        # 特征标准化
        feature_matrix = np.column_stack([features[key] for key in features.keys()])
        features['normalized'] = self.scaler.fit_transform(feature_matrix)
        
        return features

class SecondHandDeviceRecommender(nn.Module):
    """二手设备推荐模型"""
    
    def __init__(self, n_users, n_devices, n_brands, embedding_dim=64):
        super(SecondHandDeviceRecommender, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.device_embedding = nn.Embedding(n_devices, embedding_dim)
        self.brand_embedding = nn.Embedding(n_brands, embedding_dim)
        
        # 深度网络
        self.deep_layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for embedding in [self.user_embedding, self.device_embedding, self.brand_embedding]:
            nn.init.normal_(embedding.weight, std=0.1)
    
    def forward(self, user_ids, device_ids, brand_ids):
        user_emb = self.user_embedding(user_ids)
        device_emb = self.device_embedding(device_ids)
        brand_emb = self.brand_embedding(brand_ids)
        
        combined = torch.cat([user_emb, device_emb, brand_emb], dim=1)
        output = self.deep_layers(combined).squeeze()
        
        return output

class PriceRecommender:
    """价格推荐器"""
    
    def __init__(self):
        self.price_model = None
        self.feature_extractor = DeviceFeatureExtractor()
    
    def train_price_model(self, device_data):
        """训练价格预测模型"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 特征提取
        features = self.feature_extractor.extract_features(device_data)
        X = features['normalized']
        y = device_data['price'].values
        
        # 训练模型
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.price_model.fit(X, y)
        
        logger.info("价格预测模型训练完成")
    
    def recommend_price(self, device_info):
        """推荐价格"""
        if self.price_model is None:
            raise ValueError("价格模型未训练")
        
        # 特征提取
        features = self.feature_extractor.extract_features(pd.DataFrame([device_info]))
        X = features['normalized']
        
        # 预测价格
        predicted_price = self.price_model.predict(X)[0]
        
        # 考虑市场波动，给出价格区间
        price_range = {
            'min_price': predicted_price * 0.9,
            'recommended_price': predicted_price,
            'max_price': predicted_price * 1.1
        }
        
        return price_range

class LocationRecommender:
    """地理位置推荐器"""
    
    def __init__(self, max_distance_km=50):
        self.max_distance_km = max_distance_km
    
    def calculate_distance(self, loc1, loc2):
        """计算两点间距离"""
        return geodesic(loc1, loc2).kilometers
    
    def recommend_nearby_devices(self, user_location, device_data, top_k=10):
        """推荐附近的设备"""
        nearby_devices = []
        
        for idx, device in device_data.iterrows():
            device_location = (device['latitude'], device['longitude'])
            distance = self.calculate_distance(user_location, device_location)
            
            if distance <= self.max_distance_km:
                nearby_devices.append({
                    'device_id': device['device_id'],
                    'distance': distance,
                    'device_info': device
                })
        
        # 按距离排序
        nearby_devices.sort(key=lambda x: x['distance'])
        
        return nearby_devices[:top_k]

class BuyerSellerMatcher:
    """买家卖家匹配器"""
    
    def __init__(self):
        self.user_clusters = None
        self.kmeans_model = None
    
    def build_user_profiles(self, user_data, interaction_data):
        """构建用户画像"""
        # 计算用户特征
        user_features = []
        
        for user_id in user_data['user_id'].unique():
            user_interactions = interaction_data[interaction_data['user_id'] == user_id]
            
            if len(user_interactions) == 0:
                continue
            
            # 用户偏好特征
            avg_price = user_interactions['price'].mean()
            price_std = user_interactions['price'].std()
            preferred_brands = user_interactions['brand'].mode().values
            interaction_count = len(user_interactions)
            
            user_features.append({
                'user_id': user_id,
                'avg_price': avg_price,
                'price_std': price_std if not pd.isna(price_std) else 0,
                'interaction_count': interaction_count,
                'preferred_brand': preferred_brands[0] if len(preferred_brands) > 0 else 'unknown'
            })
        
        return pd.DataFrame(user_features)
    
    def cluster_users(self, user_profiles, n_clusters=5):
        """用户聚类"""
        # 选择数值特征进行聚类
        features_for_clustering = user_profiles[['avg_price', 'price_std', 'interaction_count']]
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_clustering)
        
        # K-means聚类
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans_model.fit_predict(features_scaled)
        
        user_profiles['cluster'] = clusters
        self.user_clusters = user_profiles
        
        return user_profiles
    
    def recommend_potential_buyers(self, seller_id, device_info, top_k=10):
        """为卖家推荐潜在买家"""
        if self.user_clusters is None:
            raise ValueError("用户聚类未完成")
        
        seller_info = self.user_clusters[self.user_clusters['user_id'] == seller_id]
        if len(seller_info) == 0:
            return []
        
        seller_cluster = seller_info.iloc[0]['cluster']
        
        # 找到同一聚类的用户
        potential_buyers = self.user_clusters[
            (self.user_clusters['cluster'] == seller_cluster) & 
            (self.user_clusters['user_id'] != seller_id)
        ]
        
        # 根据价格匹配度排序
        device_price = device_info.get('price', 0)
        potential_buyers['price_match_score'] = potential_buyers['avg_price'].apply(
            lambda x: 1 / (1 + abs(x - device_price) / device_price) if device_price > 0 else 0
        )
        
        # 排序并返回top_k
        potential_buyers = potential_buyers.sort_values('price_match_score', ascending=False)
        
        return potential_buyers.head(top_k)['user_id'].tolist()

class SecondHandRecommendationSystem:
    """二手设备推荐系统"""
    
    def __init__(self):
        self.model = None
        self.user_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, user_data, device_data, interaction_data, epochs=30):
        """训练推荐系统"""
        logger.info("开始训练推荐系统...")
        
        # 合并数据
        merged_data = interaction_data.merge(device_data, on='device_id', how='left')
        
        # 编码特征
        merged_data['user_encoded'] = self.user_encoder.fit_transform(merged_data['user_id'])
        merged_data['device_encoded'] = self.device_encoder.fit_transform(merged_data['device_id'])
        merged_data['brand_encoded'] = self.brand_encoder.fit_transform(merged_data['brand'])
        
        # 创建模型
        n_users = len(self.user_encoder.classes_)
        n_devices = len(self.device_encoder.classes_)
        n_brands = len(self.brand_encoder.classes_)
        
        self.model = SecondHandDeviceRecommender(n_users, n_devices, n_brands)
        
        # 训练模型
        self._train_model(merged_data, epochs)
        self.is_trained = True
        
        logger.info("训练完成!")
    
    def _train_model(self, data, epochs):
        from torch.utils.data import DataLoader, TensorDataset
        
        user_ids = torch.LongTensor(data['user_encoded'].values)
        device_ids = torch.LongTensor(data['device_encoded'].values)
        brand_ids = torch.LongTensor(data['brand_encoded'].values)
        ratings = torch.FloatTensor(data['rating'].values)
        
        dataset = TensorDataset(user_ids, device_ids, brand_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                user_batch, device_batch, brand_batch, rating_batch = batch
                
                optimizer.zero_grad()
                predictions = self.model(user_batch, device_batch, brand_batch)
                loss = criterion(predictions, rating_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def recommend_similar_devices(self, device_id, top_k=10):
        """推荐相似设备"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        try:
            device_encoded = self.device_encoder.transform([device_id])[0]
        except ValueError:
            return []
        
        device_embeddings = self.model.device_embedding.weight
        target_embedding = device_embeddings[device_encoded].unsqueeze(0)
        similarities = F.cosine_similarity(target_embedding, device_embeddings, dim=1)
        
        similarities[device_encoded] = -1
        top_k_indices = torch.topk(similarities, top_k).indices
        
        similar_devices = []
        for idx in top_k_indices:
            similar_device_id = self.device_encoder.inverse_transform([idx.item()])[0]
            similar_devices.append(similar_device_id)
        
        return similar_devices

def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    
    # 用户数据
    n_users = 500
    users = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(18, 65, n_users)
    })
    
    # 设备数据
    n_devices = 200
    brands = ['苹果', '华为', '小米', '三星', '联想']
    
    devices = pd.DataFrame({
        'device_id': range(n_devices),
        'brand': np.random.choice(brands, n_devices),
        'price': np.random.uniform(500, 5000, n_devices),
        'condition': np.random.choice(['全新', '九成新', '八成新'], n_devices)
    })
    
    # 交互数据
    n_interactions = 2000
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'device_id': np.random.randint(0, n_devices, n_interactions),
        'rating': np.random.randint(1, 6, n_interactions)
    })
    
    return users, devices, interactions

def main():
    """主函数"""
    logger.info("=== 二手设备推荐系统演示 ===")
    
    # 生成示例数据
    users, devices, interactions = generate_sample_data()
    
    logger.info(f"用户数量: {len(users)}")
    logger.info(f"设备数量: {len(devices)}")
    logger.info(f"交互数量: {len(interactions)}")
    
    # 创建推荐系统
    recommender = SecondHandRecommendationSystem()
    
    # 训练模型
    recommender.train(users, devices, interactions, epochs=20)
    
    # 测试推荐
    test_device_id = 0
    similar_devices = recommender.recommend_similar_devices(test_device_id, top_k=5)
    
    logger.info(f"与设备 {test_device_id} 相似的设备: {similar_devices}")
    
    # 显示设备信息
    test_device = devices[devices['device_id'] == test_device_id].iloc[0]
    logger.info(f"测试设备信息: 品牌={test_device['brand']}, 价格={test_device['price']:.0f}")
    
    for device_id in similar_devices[:3]:
        device_info = devices[devices['device_id'] == device_id].iloc[0]
        logger.info(f"相似设备 {device_id}: 品牌={device_info['brand']}, 价格={device_info['price']:.0f}")

if __name__ == "__main__":
    main() 