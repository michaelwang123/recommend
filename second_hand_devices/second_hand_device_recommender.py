"""
二手设备推荐系统
==================

这个模块实现了一个完整的二手设备推荐系统，包括：
1. 基于深度学习的协同过滤推荐
2. 基于机器学习的价格预测
3. 基于地理位置的邻近推荐
4. 买家卖家智能匹配

主要技术：
- PyTorch深度学习框架
- 嵌入层(Embedding)进行特征学习
- 多层感知机(MLP)进行非线性建模
- 随机森林进行价格预测
- K-means聚类进行用户分群
"""

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
    """
    二手设备特征提取器
    =================
    
    这个类负责将设备的原始数据转换为机器学习模型可以使用的特征。
    主要功能包括：
    1. 类别特征编码（品牌、型号、成色等）
    2. 数值特征处理（价格、使用时间、存储等）
    3. 特征标准化和归一化
    
    用途：主要用于价格预测模型的特征工程
    """
    
    def __init__(self):
        """
        初始化特征提取器
        
        创建各种编码器和标准化器：
        - brand_encoder: 品牌标签编码器（苹果→0, 华为→1, 小米→2 等）
        - model_encoder: 型号标签编码器
        - condition_encoder: 成色标签编码器（全新→0, 九成新→1 等）
        - category_encoder: 类别标签编码器（手机→0, 笔记本→1 等）
        - scaler: 数值特征标准化器（将特征缩放到相同范围）
        """
        self.brand_encoder = LabelEncoder()      # 品牌编码器
        self.model_encoder = LabelEncoder()      # 型号编码器
        self.condition_encoder = LabelEncoder()  # 成色编码器
        self.category_encoder = LabelEncoder()   # 类别编码器
        self.scaler = StandardScaler()           # 标准化器
        
    def extract_features(self, device_data):
        """
        提取设备特征
        ===========
        
        将原始设备数据转换为机器学习特征向量
        
        参数:
            device_data (pandas.DataFrame): 包含设备信息的数据框
                必须包含: brand, category, condition, price 等列
                
        返回:
            dict: 特征字典，包含编码后的特征和标准化后的特征矩阵
        """
        features = {}
        
        # === 类别特征编码 ===
        # 将文本类别转换为数字编码，便于机器学习处理
        
        # 品牌特征：苹果、华为、小米等 → 0, 1, 2等
        features['brand'] = self.brand_encoder.fit_transform(device_data['brand'])
        
        # 型号特征处理（如果没有型号，用品牌代替）
        if 'model' in device_data.columns:
            features['model'] = self.model_encoder.fit_transform(device_data['model'])
        else:
            # 降级处理：没有型号信息时使用品牌作为型号
            features['model'] = self.model_encoder.fit_transform(device_data['brand'])
        
        # 类别特征：手机、笔记本、平板等 → 0, 1, 2等
        features['category'] = self.category_encoder.fit_transform(device_data['category'])
        
        # 成色特征：全新、九成新、八成新等 → 0, 1, 2等
        features['condition'] = self.condition_encoder.fit_transform(device_data['condition'])
        
        # === 数值特征处理 ===
        # 直接使用数值特征，但需要处理缺失值
        numerical_features = ['price', 'age_months', 'storage_gb', 'ram_gb', 'screen_size']
        for feature in numerical_features:
            if feature in device_data.columns:
                # 用0填充缺失值（也可以用均值等其他策略）
                features[feature] = device_data[feature].fillna(0).values
        
        # === 特征标准化 ===
        # 将所有特征组合成矩阵，然后标准化到相同的量级
        # 这对于机器学习模型的训练很重要
        feature_matrix = np.column_stack([features[key] for key in features.keys()])
        features['normalized'] = self.scaler.fit_transform(feature_matrix)
        
        return features

class SecondHandDeviceRecommender(nn.Module):
    """
    二手设备推荐深度学习模型
    =====================
    
    这是推荐系统的核心神经网络模型，使用深度学习技术进行协同过滤推荐。
    
    模型架构：
    1. 嵌入层(Embedding Layer)：将离散的用户ID、设备ID、品牌ID转换为密集向量
    2. 特征融合层：将三个嵌入向量拼接起来
    3. 深度神经网络：通过多层感知机学习复杂的用户-设备交互模式
    4. 输出层：预测用户对设备的评分
    
    技术特点：
    - 使用嵌入层进行表示学习
    - 多层感知机捕捉非线性关系
    - Dropout防止过拟合
    - Xavier初始化保证训练稳定性
    """
    
    def __init__(self, n_users, n_devices, n_brands, embedding_dim=64):
        """
        初始化推荐模型
        
        参数:
            n_users (int): 用户总数
            n_devices (int): 设备总数  
            n_brands (int): 品牌总数
            embedding_dim (int): 嵌入向量维度，默认64
        """
        super(SecondHandDeviceRecommender, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # === 嵌入层定义 ===
        # 将离散的ID转换为连续的向量表示，这是深度学习推荐系统的核心
        
        # 用户嵌入：每个用户ID对应一个64维向量
        # 这个向量会学习到用户的偏好特征
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        
        # 设备嵌入：每个设备ID对应一个64维向量
        # 这个向量会学习到设备的特征表示
        self.device_embedding = nn.Embedding(n_devices, embedding_dim)
        
        # 品牌嵌入：每个品牌ID对应一个64维向量
        # 这个向量会学习到品牌的特征表示
        self.brand_embedding = nn.Embedding(n_brands, embedding_dim)
        
        # === 深度神经网络定义 ===
        # 多层感知机(MLP)用于学习用户-设备-品牌之间的复杂交互模式
        self.deep_layers = nn.Sequential(
            # 第一层：输入层 (64*3=192维) → 隐藏层 (128维)
            nn.Linear(embedding_dim * 3, 128),  # 全连接层
            nn.ReLU(),                          # ReLU激活函数，引入非线性
            nn.Dropout(0.2),                    # 20%的神经元随机失活，防止过拟合
            
            # 第二层：隐藏层 (128维) → 隐藏层 (64维)
            nn.Linear(128, 64),                 # 全连接层
            nn.ReLU(),                          # ReLU激活函数
            nn.Dropout(0.2),                    # 20%的神经元随机失活
            
            # 第三层：输出层 (64维) → 输出 (1维)
            nn.Linear(64, 1)                    # 最终输出用户对设备的评分预测
        )
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重
        
        使用Xavier初始化方法，确保训练开始时梯度的方差适中，
        有助于网络更好地收敛。
        """
        # 初始化嵌入层权重
        for embedding in [self.user_embedding, self.device_embedding, self.brand_embedding]:
            nn.init.xavier_uniform_(embedding.weight)
        
        # 初始化深度网络权重
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # 权重用Xavier初始化
                nn.init.constant_(layer.bias, 0)       # 偏置初始化为0
    
    def forward(self, user_ids, device_ids, brand_ids):
        """
        前向传播函数
        
        这是神经网络的核心计算过程，定义了从输入到输出的完整计算流程。
        
        参数:
            user_ids (torch.Tensor): 用户ID张量，形状为[batch_size]
            device_ids (torch.Tensor): 设备ID张量，形状为[batch_size]  
            brand_ids (torch.Tensor): 品牌ID张量，形状为[batch_size]
            
        返回:
            torch.Tensor: 预测的评分，形状为[batch_size]
        """
        # === 第一步：嵌入查找 ===
        # 将离散的ID转换为连续的向量表示
        
        # 用户嵌入：用户ID → 64维向量
        user_emb = self.user_embedding(user_ids)      # [batch_size, 64]
        
        # 设备嵌入：设备ID → 64维向量  
        device_emb = self.device_embedding(device_ids)  # [batch_size, 64]
        
        # 品牌嵌入：品牌ID → 64维向量
        brand_emb = self.brand_embedding(brand_ids)    # [batch_size, 64]
        
        # === 第二步：特征融合 ===
        # 将三个64维向量拼接成一个192维向量
        combined = torch.cat([user_emb, device_emb, brand_emb], dim=1)  # [batch_size, 192]
        
        # === 第三步：深度网络前向传播 ===
        # 通过多层感知机学习复杂的交互模式
        output = self.deep_layers(combined).squeeze()  # [batch_size, 1] → [batch_size]
        
        return output

class PriceRecommender:
    """
    价格推荐器
    =========
    
    这个类使用机器学习技术为二手设备提供价格预测和推荐。
    
    核心功能：
    1. 基于设备特征训练价格预测模型
    2. 为新设备预测合理的价格区间
    3. 考虑市场波动给出价格建议
    
    技术特点：
    - 使用随机森林回归算法
    - 不依赖用户交互数据，纯粹基于设备属性
    - 提供价格区间而非单点预测，更实用
    
    应用场景：
    - 卖家定价参考
    - 买家价格评估
    - 市场价格分析
    """
    
    def __init__(self):
        """
        初始化价格推荐器
        
        创建空的价格模型和特征提取器
        """
        self.price_model = None                           # 价格预测模型（随机森林）
        self.feature_extractor = DeviceFeatureExtractor()  # 特征提取器
    
    def train_price_model(self, device_data):
        """
        训练价格预测模型
        
        使用设备数据训练随机森林回归模型来预测价格。
        这是一个监督学习过程，使用设备特征作为输入，价格作为标签。
        
        参数:
            device_data (pandas.DataFrame): 设备数据，必须包含价格列
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # === 特征提取 ===
        # 将原始设备数据转换为机器学习特征
        features = self.feature_extractor.extract_features(device_data)
        X = features['normalized']  # 标准化后的特征矩阵
        y = device_data['price'].values  # 价格标签
        
        # === 模型训练 ===
        # 使用随机森林回归算法
        # 随机森林的优点：
        # 1. 处理非线性关系能力强
        # 2. 对特征重要性敏感
        # 3. 不容易过拟合
        # 4. 可以处理缺失值
        self.price_model = RandomForestRegressor(
            n_estimators=100,    # 100棵决策树
            random_state=42      # 随机种子，确保结果可重复
        )
        self.price_model.fit(X, y)
        
        logger.info("价格预测模型训练完成")
    
    def recommend_price(self, device_info):
        """
        推荐价格
        
        为给定的设备信息预测合理的价格区间。
        
        参数:
            device_info (dict): 设备信息字典，包含品牌、类别、成色等信息
            
        返回:
            dict: 包含最小价格、推荐价格、最大价格的字典
            
        示例:
            device_info = {
                'brand': '苹果',
                'category': '手机', 
                'condition': '九成新',
                'age_months': 12,
                'storage_gb': 128,
                'ram_gb': 8,
                'screen_size': 6.1
            }
        """
        if self.price_model is None:
            raise ValueError("价格模型未训练")
        
        # === 特征提取 ===
        # 将设备信息转换为模型可以理解的特征向量
        features = self.feature_extractor.extract_features(pd.DataFrame([device_info]))
        X = features['normalized']
        
        # === 价格预测 ===
        # 使用训练好的随机森林模型预测价格
        predicted_price = self.price_model.predict(X)[0]
        
        # === 价格区间计算 ===
        # 考虑市场波动和预测不确定性，给出价格区间
        # 这比单点预测更实用，给买卖双方留有协商空间
        price_range = {
            'min_price': predicted_price * 0.9,        # 最低价格：预测价格的90%
            'recommended_price': predicted_price,       # 推荐价格：预测价格
            'max_price': predicted_price * 1.1         # 最高价格：预测价格的110%
        }
        
        return price_range

class LocationRecommender:
    """
    地理位置推荐器
    ===========
    
    这个类基于地理位置信息为用户推荐附近的二手设备。
    
    核心功能：
    1. 计算用户与设备之间的地理距离
    2. 过滤指定距离范围内的设备
    3. 按距离排序推荐最近的设备
    
    技术特点：
    - 使用大地测量学方法计算真实地球距离
    - 支持自定义距离阈值
    - 不依赖用户交互数据，纯粹基于地理位置
    
    应用场景：
    - 本地化推荐
    - 减少物流成本
    - 提高交易成功率
    """
    
    def __init__(self, max_distance_km=50):
        """
        初始化地理位置推荐器
        
        参数:
            max_distance_km (float): 最大推荐距离，单位为公里，默认50公里
        """
        self.max_distance_km = max_distance_km
    
    def calculate_distance(self, loc1, loc2):
        """
        计算两点间距离
        
        使用大地测量学方法计算地球表面两点间的真实距离。
        这比简单的欧几里得距离更准确，特别是对于较长的距离。
        
        参数:
            loc1 (tuple): 第一个位置的(纬度, 经度)
            loc2 (tuple): 第二个位置的(纬度, 经度)
            
        返回:
            float: 两点间距离，单位为公里
            
        示例:
            # 计算北京天安门到上海外滩的距离
            beijing = (39.9042, 116.4074)
            shanghai = (31.2304, 121.4737)
            distance = calculate_distance(beijing, shanghai)
        """
        return geodesic(loc1, loc2).kilometers
    
    def recommend_nearby_devices(self, user_location, device_data, top_k=10):
        """
        推荐附近的设备
        
        找出用户指定距离范围内的所有设备，并按距离从近到远排序。
        
        参数:
            user_location (tuple): 用户位置的(纬度, 经度)
            device_data (pandas.DataFrame): 设备数据，必须包含latitude和longitude列
            top_k (int): 返回的设备数量，默认10个
            
        返回:
            list: 附近设备列表，每个元素包含设备ID、距离、设备信息
            
        示例:
            user_location = (39.9042, 116.4074)  # 北京天安门
            nearby_devices = recommend_nearby_devices(user_location, devices_df, top_k=5)
        """
        nearby_devices = []
        
        # === 遍历所有设备 ===
        for idx, device in device_data.iterrows():
            # 获取设备的地理位置
            device_location = (device['latitude'], device['longitude'])
            
            # 计算用户与设备之间的距离
            distance = self.calculate_distance(user_location, device_location)
            
            # === 距离过滤 ===
            # 只推荐在指定距离范围内的设备
            if distance <= self.max_distance_km:
                nearby_devices.append({
                    'device_id': device['device_id'],  # 设备ID
                    'distance': distance,              # 距离（公里）
                    'device_info': device              # 完整设备信息
                })
        
        # === 距离排序 ===
        # 按距离从近到远排序，距离越近的设备排在前面
        nearby_devices.sort(key=lambda x: x['distance'])
        
        # 返回前top_k个最近的设备
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
    """
    二手设备推荐系统主类
    ==================
    
    这是整个推荐系统的核心类，集成了深度学习模型和各种推荐功能。
    
    主要功能：
    1. 训练深度学习推荐模型
    2. 为用户推荐设备
    3. 推荐相似设备
    4. 处理数据编码和预处理
    
    工作流程：
    1. 数据预处理：编码用户ID、设备ID、品牌ID
    2. 模型训练：使用深度学习模型学习用户偏好
    3. 推荐生成：基于训练好的模型生成推荐结果
    
    技术栈：
    - 深度学习模型：SecondHandDeviceRecommender
    - 数据编码：LabelEncoder
    - 相似度计算：余弦相似度
    """
    
    def __init__(self):
        """
        初始化推荐系统
        
        创建各种组件和编码器
        """
        self.model = None                        # 深度学习模型
        self.user_encoder = LabelEncoder()       # 用户ID编码器
        self.device_encoder = LabelEncoder()     # 设备ID编码器  
        self.brand_encoder = LabelEncoder()      # 品牌ID编码器
        self.is_trained = False                  # 训练状态标志
    
    def train(self, user_data, device_data, interaction_data, epochs=30):
        """
        训练推荐系统
        
        这是整个推荐系统的核心训练过程，包括数据预处理、模型创建和训练。
        
        参数:
            user_data (pandas.DataFrame): 用户数据，包含user_id等信息
            device_data (pandas.DataFrame): 设备数据，包含device_id、brand等信息
            interaction_data (pandas.DataFrame): 交互数据，包含user_id、device_id、rating等
            epochs (int): 训练轮数，默认30
            
        训练流程:
        1. 数据合并：将交互数据与设备数据关联
        2. 特征编码：将文本ID转换为数字编码
        3. 模型创建：根据数据规模创建深度学习模型
        4. 模型训练：使用梯度下降优化模型参数
        """
        logger.info("开始训练推荐系统...")
        
        # === 第一步：数据合并 ===
        # 将交互数据与设备数据关联，获得完整的特征信息
        # 例如：用户1对设备5的评分 + 设备5的品牌信息
        merged_data = interaction_data.merge(device_data, on='device_id', how='left')
        
        # === 第二步：特征编码 ===
        # 将文本形式的ID转换为数字编码，便于神经网络处理
        
        # 用户ID编码：将用户ID转换为0到n_users-1的连续整数
        merged_data['user_encoded'] = self.user_encoder.fit_transform(merged_data['user_id'])
        
        # 设备ID编码：将设备ID转换为0到n_devices-1的连续整数
        merged_data['device_encoded'] = self.device_encoder.fit_transform(merged_data['device_id'])
        
        # 品牌ID编码：将品牌名称转换为0到n_brands-1的连续整数
        merged_data['brand_encoded'] = self.brand_encoder.fit_transform(merged_data['brand'])
        
        # === 第三步：模型创建 ===
        # 根据数据规模确定模型参数
        n_users = len(self.user_encoder.classes_)    # 用户总数
        n_devices = len(self.device_encoder.classes_)  # 设备总数
        n_brands = len(self.brand_encoder.classes_)   # 品牌总数
        
        # 创建深度学习推荐模型
        self.model = SecondHandDeviceRecommender(n_users, n_devices, n_brands)
        
        # === 第四步：模型训练 ===
        # 使用准备好的数据训练模型
        self._train_model(merged_data, epochs)
        self.is_trained = True
        
        logger.info("训练完成!")
    
    def _train_model(self, data, epochs):
        """
        内部训练方法
        
        这是实际的神经网络训练过程，包含数据准备、训练循环和优化。
        
        参数:
            data (pandas.DataFrame): 预处理后的训练数据
            epochs (int): 训练轮数
            
        训练过程:
        1. 数据准备：转换为PyTorch张量
        2. 数据加载：创建数据加载器进行批处理
        3. 优化器设置：使用Adam优化器
        4. 训练循环：反复进行前向传播、损失计算、反向传播
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        # === 第一步：数据准备 ===
        # 将pandas数据转换为PyTorch张量
        user_ids = torch.LongTensor(data['user_encoded'].values)   # 用户ID张量
        device_ids = torch.LongTensor(data['device_encoded'].values)  # 设备ID张量
        brand_ids = torch.LongTensor(data['brand_encoded'].values)   # 品牌ID张量
        ratings = torch.FloatTensor(data['rating'].values)          # 评分张量
        
        # === 评分标准化 ===
        # 将评分标准化到[0,1]范围，有助于神经网络训练稳定
        ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
        
        # === 第二步：数据加载器 ===
        # 创建数据集和数据加载器，支持批处理和数据打乱
        dataset = TensorDataset(user_ids, device_ids, brand_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        # === 第三步：优化器设置 ===
        # 使用Adam优化器，这是深度学习中常用的优化算法
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.0001,           # 学习率设置较低，保证训练稳定
            weight_decay=1e-5    # L2正则化，防止过拟合
        )
        criterion = nn.MSELoss()  # 均方误差损失函数
        
        # === 第四步：训练循环 ===
        self.model.train()  # 设置为训练模式
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            # 遍历所有批次
            for batch in dataloader:
                user_batch, device_batch, brand_batch, rating_batch = batch
                
                # === 前向传播 ===
                optimizer.zero_grad()  # 清空梯度
                predictions = self.model(user_batch, device_batch, brand_batch)
                
                # === 预测值处理 ===
                # 使用sigmoid函数将预测值限制在[0,1]范围内
                predictions = torch.sigmoid(predictions)
                
                # === 损失计算 ===
                loss = criterion(predictions, rating_batch)
                
                # === 异常检查 ===
                # 检查损失是否为NaN或无穷大，如果是则跳过这个批次
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Loss is {loss.item()}, skipping batch")
                    continue
                
                # === 反向传播 ===
                loss.backward()  # 计算梯度
                
                # === 梯度裁剪 ===
                # 防止梯度爆炸，将梯度范数限制在1.0以内
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # === 参数更新 ===
                optimizer.step()  # 根据梯度更新参数
                
                # === 统计信息 ===
                total_loss += loss.item()
                batch_count += 1
            
            # === 训练进度输出 ===
            avg_loss = total_loss / max(batch_count, 1)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    
    def recommend_top_n_devices_for_user(self, user_id, device_data, interaction_data=None, top_n=10, exclude_interacted=True, brand_weight=0.02):
        """
        为指定用户推荐TopN个设备（改进版）
        
        基于训练好的深度学习模型，为指定用户生成个性化的设备推荐。
        相比原版函数，这个版本：
        1. 正确处理每个设备的品牌信息
        2. 可以过滤掉用户已经交互过的设备
        3. 提供更详细的设备信息
        4. 支持评分标准化
        
        参数:
            user_id: 用户ID
            device_data (pandas.DataFrame): 设备数据，包含device_id、brand等信息
            interaction_data (pandas.DataFrame): 交互数据，用于过滤已交互设备
            top_n (int): 推荐设备数量，默认10个
            exclude_interacted (bool): 是否排除用户已交互的设备，默认True
            brand_weight (float): 品牌偏好权重，默认0.02（2%），设为0则完全依赖模型学习
            
        返回:
            list: 推荐结果列表，每个元素包含设备ID、评分、设备详细信息
            
        示例:
            recommendations = recommender.recommend_top_n_devices_for_user(
                user_id=12, 
                device_data=devices_df,
                interaction_data=interactions_df,
                top_n=5,
                exclude_interacted=True
            )
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # === 第一步：用户ID编码 ===
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except ValueError:
            logger.warning(f"用户ID {user_id} 不在训练数据中")
            return []
        
        # === 第二步：获取用户已交互的设备（用于过滤）===
        interacted_devices = set()
        if exclude_interacted and interaction_data is not None:
            user_interactions = interaction_data[interaction_data['user_id'] == user_id]
            interacted_devices = set(user_interactions['device_id'].values)
        
        # === 第三步：获取用户历史偏好品牌 ===
        user_brand_preferences = {}
        if interaction_data is not None:
            user_interactions = interaction_data[interaction_data['user_id'] == user_id]
            if not user_interactions.empty:
                # 合并设备信息以获取品牌
                interactions_with_brands = user_interactions.merge(
                    device_data[['device_id', 'brand']], 
                    on='device_id', 
                    how='left'
                )
                # 统计用户对各品牌的偏好（基于交互次数和平均评分）
                brand_stats = interactions_with_brands.groupby('brand').agg({
                    'rating': ['count', 'mean']
                }).reset_index()
                brand_stats.columns = ['brand', 'interaction_count', 'avg_rating']
                
                # 计算品牌偏好权重（交互次数 * 平均评分）
                for _, row in brand_stats.iterrows():
                    brand = row['brand']
                    weight = row['interaction_count'] * row['avg_rating'] / 5.0  # 归一化到[0,1]
                    user_brand_preferences[brand] = weight
        
        # === 第四步：批量预测所有设备评分 ===
        self.model.eval()  # 设置为评估模式
        device_scores = []
        
        with torch.no_grad():  # 禁用梯度计算，节省内存和计算
            user_tensor = torch.LongTensor([user_encoded])
            
            # 遍历所有设备
            for _, device_row in device_data.iterrows():
                device_id = device_row['device_id']
                
                # 如果需要排除已交互设备，则跳过
                if exclude_interacted and device_id in interacted_devices:
                    continue
                
                try:
                    # 编码设备ID和品牌ID
                    device_encoded = self.device_encoder.transform([device_id])[0]
                    brand_encoded = self.brand_encoder.transform([device_row['brand']])[0]
                    
                    # 转换为张量
                    device_tensor = torch.LongTensor([device_encoded])
                    brand_tensor = torch.LongTensor([brand_encoded])
                    
                    # === 模型预测 ===
                    score = self.model(user_tensor, device_tensor, brand_tensor)
                    
                    # 使用sigmoid函数将评分标准化到[0,1]范围
                    score = torch.sigmoid(score).item()
                    
                    # === 添加品牌偏好权重（学习用，可配置）===
                    brand = device_row['brand']
                    if brand_weight > 0 and brand in user_brand_preferences:
                        # 如果用户历史上偏好这个品牌，给予少量额外权重
                        # 权重可配置：brand_weight=0.02（2%），brand_weight=0则完全依赖模型学习
                        brand_bonus = user_brand_preferences[brand] * brand_weight
                        score = min(1.0, score + brand_bonus)  # 确保分数不超过1.0
                    
                    # 保存预测结果
                    device_scores.append({
                        'device_id': device_id,
                        'score': score,
                        'device_info': device_row.to_dict()
                    })
                    
                except ValueError:
                    # 如果设备ID或品牌不在训练数据中，跳过
                    continue
                except Exception as e:
                    # 其他异常也跳过，避免程序崩溃
                    logger.warning(f"预测设备 {device_id} 时出错: {e}")
                    continue
        
        # === 第四步：排序和返回TopN ===
        # 按评分从高到低排序
        device_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回前top_n个设备
        top_recommendations = device_scores[:top_n]
        
        # === 第五步：添加排名信息 ===
        for i, rec in enumerate(top_recommendations):
            rec['rank'] = i + 1
        
        logger.info(f"为用户 {user_id} 生成了 {len(top_recommendations)} 个推荐")
        
        return top_recommendations
    
    def get_user_recommendation_summary(self, user_id, device_data, interaction_data=None, top_n=10):
        """
        获取用户推荐摘要
        
        为用户生成推荐结果的详细摘要，包括推荐设备信息和统计分析。
        
        参数:
            user_id: 用户ID
            device_data (pandas.DataFrame): 设备数据
            interaction_data (pandas.DataFrame): 交互数据
            top_n (int): 推荐设备数量，默认10个
            
        返回:
            dict: 包含推荐结果和统计信息的字典
        """
        # 获取推荐结果
        recommendations = self.recommend_top_n_devices_for_user(
            user_id, device_data, interaction_data, top_n
        )
        
        if not recommendations:
            return {
                'user_id': user_id,
                'recommendations': [],
                'summary': {
                    'total_recommendations': 0,
                    'avg_score': 0,
                    'brand_distribution': {},
                    'price_range': {'min': 0, 'max': 0, 'avg': 0}
                }
            }
        
        # === 统计分析 ===
        scores = [rec['score'] for rec in recommendations]
        brands = [rec['device_info']['brand'] for rec in recommendations]
        prices = [rec['device_info']['price'] for rec in recommendations]
        
        # 推荐结果中的品牌分布统计
        brand_counts = pd.Series(brands).value_counts().to_dict()
        
        # 价格统计
        price_stats = {
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / len(prices)
        }
        
        # 评分统计
        avg_score = sum(scores) / len(scores)
        
        # === 获取用户历史偏好的品牌（基于交互数据）===
        user_favorite_brand = "未知"
        if interaction_data is not None:
            user_interactions = interaction_data[interaction_data['user_id'] == user_id]
            if not user_interactions.empty:
                # 合并设备信息以获取品牌
                interactions_with_brands = user_interactions.merge(
                    device_data[['device_id', 'brand']], 
                    on='device_id', 
                    how='left'
                )
                # 统计用户历史交互的品牌分布
                historical_brands = interactions_with_brands['brand'].value_counts()
                if not historical_brands.empty:
                    user_favorite_brand = historical_brands.index[0]  # 最常交互的品牌
        
        # 构建摘要
        summary = {
            'user_id': user_id,
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'avg_score': avg_score,
                'brand_distribution': brand_counts,
                'price_range': price_stats,
                'top_brand': max(brand_counts.items(), key=lambda x: x[1])[0],  # 推荐结果中最多的品牌
                'user_favorite_brand': user_favorite_brand,  # 用户历史偏好的品牌
                'score_range': {'min': min(scores), 'max': max(scores)}
            }
        }
        
        return summary
    
    def print_user_interaction_history(self, user_id, device_data, interaction_data):
        """
        打印用户的历史交互设备记录
        
        使用logger打印指定用户的所有历史交互记录，包括设备详细信息、
        交互评分、交互时间等。这有助于了解用户的购买偏好和行为模式。
        
        参数:
            user_id: 用户ID
            device_data (pandas.DataFrame): 设备数据，包含device_id、brand、price等信息
            interaction_data (pandas.DataFrame): 交互数据，包含user_id、device_id、rating等信息
            
        功能:
            - 查找用户的所有历史交互记录
            - 关联设备详细信息
            - 按评分排序显示
            - 提供统计摘要
        """
        if interaction_data is None:
            logger.warning(f"交互数据为空，无法查询用户 {user_id} 的历史记录")
            return
        
        # === 第一步：查找用户的历史交互记录 ===
        user_interactions = interaction_data[interaction_data['user_id'] == user_id].copy()
        
        if len(user_interactions) == 0:
            logger.info(f"用户 {user_id} 没有历史交互记录")
            return
        
        # === 第二步：关联设备详细信息 ===
        # 将交互数据与设备数据合并，获取完整的设备信息
        user_history = user_interactions.merge(device_data, on='device_id', how='left')
        
        # 按评分从高到低排序
        user_history = user_history.sort_values('rating', ascending=False)
        
        # === 第三步：使用logger打印详细历史记录 ===
        logger.info(f"\n=== 用户 {user_id} 的历史交互设备记录 ===")
        logger.info(f"总交互设备数量: {len(user_history)}")
        
        # 打印每个交互记录的详细信息
        for idx, (_, record) in enumerate(user_history.iterrows(), 1):
            device_id = record['device_id']
            rating = record['rating']
            brand = record.get('brand', '未知品牌')
            price = record.get('price', 0)
            condition = record.get('condition', '未知成色')
            
            logger.info(f"第{idx}条记录: 设备{device_id} | 评分: {rating} | "
                       f"品牌: {brand} | 价格: {price:.0f}元 | 成色: {condition}")
        
        # === 第四步：统计分析 ===
        # 计算用户的购买偏好统计
        avg_rating = user_history['rating'].mean()
        max_rating = user_history['rating'].max()
        min_rating = user_history['rating'].min()
        
        # 品牌偏好统计
        brand_counts = user_history['brand'].value_counts()
        most_liked_brand = brand_counts.index[0] if len(brand_counts) > 0 else "无"
        
        # 价格偏好统计
        avg_price = user_history['price'].mean()
        max_price = user_history['price'].max()
        min_price = user_history['price'].min()
        
        # 成色偏好统计
        condition_counts = user_history['condition'].value_counts()
        preferred_condition = condition_counts.index[0] if len(condition_counts) > 0 else "无"
        
        # === 第五步：打印统计摘要 ===
        logger.info(f"\n=== 用户 {user_id} 的购买偏好分析 ===")
        logger.info(f"平均评分: {avg_rating:.2f} (范围: {min_rating} - {max_rating})")
        logger.info(f"最喜欢的品牌: {most_liked_brand} (共{brand_counts.iloc[0]}次交互)")
        logger.info(f"平均价格: {avg_price:.0f}元 (范围: {min_price:.0f} - {max_price:.0f}元)")
        logger.info(f"偏好成色: {preferred_condition} (共{condition_counts.iloc[0]}次选择)")
        
        # 品牌分布详情
        logger.info(f"\n品牌偏好分布:")
        for brand, count in brand_counts.head(5).items():
            percentage = (count / len(user_history)) * 100
            logger.info(f"  {brand}: {count}次 ({percentage:.1f}%)")
        
        # 评分分布
        rating_distribution = user_history['rating'].value_counts().sort_index()
        logger.info(f"\n评分分布:")
        for rating, count in rating_distribution.items():
            percentage = (count / len(user_history)) * 100
            logger.info(f"  {rating}分: {count}次 ({percentage:.1f}%)")
        
        # === 第六步：高评分设备推荐 ===
        # 找出用户评分最高的设备，作为偏好参考
        high_rated_devices = user_history[user_history['rating'] >= 4]
        if len(high_rated_devices) > 0:
            logger.info(f"\n用户高评分设备 (评分>=4):")
            for _, device in high_rated_devices.head(3).iterrows():
                logger.info(f"  设备{device['device_id']}: {device['brand']} | "
                           f"评分{device['rating']} | 价格{device['price']:.0f}元")
        
        logger.info("=" * 50)
    
    def recommend_similar_devices(self, device_id, top_k=10):
        """
        推荐相似设备
        
        基于设备嵌入向量的相似度计算，找出与目标设备最相似的其他设备。
        
        参数:
            device_id: 目标设备ID
            top_k (int): 返回相似设备数量，默认10个
            
        返回:
            list: 相似设备ID列表，按相似度从高到低排序
            
        相似度计算原理:
        1. 提取设备嵌入向量：从训练好的模型中获取设备的向量表示
        2. 计算余弦相似度：比较目标设备与所有设备的向量相似度
        3. 排序返回：返回相似度最高的前k个设备
        
        技术细节:
        - 使用余弦相似度而非欧几里得距离，对向量长度不敏感
        - 排除目标设备本身，避免推荐重复设备
        - 基于深度学习学到的语义相似性，而非表面特征
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # === 第一步：设备ID编码 ===
        try:
            device_encoded = self.device_encoder.transform([device_id])[0]
        except ValueError:
            # 如果设备ID不在训练数据中，返回空列表
            return []
        
        # === 第二步：提取设备嵌入向量 ===
        # 从训练好的模型中获取所有设备的嵌入向量
        device_embeddings = self.model.device_embedding.weight  # [n_devices, embedding_dim]
        
        # 获取目标设备的嵌入向量
        target_embedding = device_embeddings[device_encoded].unsqueeze(0)  # [1, embedding_dim]
        
        # === 第三步：计算相似度 ===
        # 使用余弦相似度计算目标设备与所有设备的相似度
        # 余弦相似度范围[-1, 1]，值越大表示越相似
        similarities = F.cosine_similarity(target_embedding, device_embeddings, dim=1)
        
        # === 第四步：排除自己 ===
        # 将目标设备的相似度设为-1，确保不会推荐自己
        similarities[device_encoded] = -1
        
        # === 第五步：获取最相似的设备 ===
        # 找出相似度最高的前top_k个设备
        top_k_indices = torch.topk(similarities, top_k).indices
        
        # === 第六步：转换为设备ID ===
        similar_devices = []
        for idx in top_k_indices:
            try:
                # 将编码转换回原始设备ID
                similar_device_id = self.device_encoder.inverse_transform([idx.item()])[0]
                similar_devices.append(similar_device_id)
            except:
                # 如果转换失败，跳过这个设备
                continue
        
        return similar_devices

def generate_sample_data():
    """
    生成更真实的示例数据
    
    改进点：
    1. 为每个用户生成品牌偏好
    2. 根据用户偏好生成交互数据
    3. 偏好品牌的评分更高
    4. 价格偏好与品牌偏好相关
    5. 增加用户年龄对品牌偏好的影响
    """
    np.random.seed(42)
    
    # 用户数据
    n_users = 500
    users = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(18, 65, n_users)
    })
    
    # 设备数据 - 更真实的品牌分布和价格范围
    n_devices = 200
    brands = ['苹果', '华为', '小米', '三星', '联想']
    
    # 不同品牌的价格范围更真实
    brand_price_ranges = {
        '苹果': (3000, 8000),
        '华为': (2000, 6000),
        '三星': (1500, 5000),
        '小米': (1000, 4000),
        '联想': (800, 3500)
    }
    
    devices_data = []
    for device_id in range(n_devices):
        brand = np.random.choice(brands)
        price_min, price_max = brand_price_ranges[brand]
        price = np.random.uniform(price_min, price_max)
        
        devices_data.append({
            'device_id': device_id,
            'brand': brand,
            'price': price,
            'condition': np.random.choice(['全新', '九成新', '八成新'], p=[0.2, 0.5, 0.3])
        })
    
    devices = pd.DataFrame(devices_data)
    
    # === 生成用户品牌偏好 ===
    logger.info("生成用户品牌偏好...")
    user_brand_preferences = {}
    
    for user_id in range(n_users):
        user_age = users.loc[user_id, 'age']
        
        # 根据年龄影响品牌偏好
        if user_age < 25:
            # 年轻用户偏好小米、华为
            preferred_brands = np.random.choice(['小米', '华为', '三星'], 
                                              size=np.random.randint(1, 3), 
                                              replace=False,
                                              p=[0.4, 0.4, 0.2])
        elif user_age < 35:
            # 中年用户偏好华为、三星、苹果
            preferred_brands = np.random.choice(['华为', '三星', '苹果'], 
                                              size=np.random.randint(1, 3), 
                                              replace=False,
                                              p=[0.4, 0.3, 0.3])
        else:
            # 成熟用户偏好苹果、联想、华为
            preferred_brands = np.random.choice(['苹果', '联想', '华为'], 
                                              size=np.random.randint(1, 2), 
                                              replace=False,
                                              p=[0.5, 0.3, 0.2])
        
        user_brand_preferences[user_id] = preferred_brands
    
    # === 根据用户偏好生成交互数据 ===
    logger.info("根据用户偏好生成交互数据...")
    interactions_data = []
    
    for user_id in range(n_users):
        preferred_brands = user_brand_preferences[user_id]
        user_age = users.loc[user_id, 'age']
        
        # 根据年龄决定交互数量（年轻人更活跃）
        if user_age < 30:
            n_user_interactions = np.random.randint(5, 12)
        elif user_age < 45:
            n_user_interactions = np.random.randint(3, 8)
        else:
            n_user_interactions = np.random.randint(2, 6)
        
        for _ in range(n_user_interactions):
            # 80%概率选择偏好品牌的设备
            if np.random.random() < 0.8:
                # 选择偏好品牌的设备
                preferred_devices = devices[devices['brand'].isin(preferred_brands)]
                if len(preferred_devices) > 0:
                    device = preferred_devices.sample(1).iloc[0]
                    # 偏好品牌的评分更高（3-5分）
                    rating = np.random.randint(3, 6)
                    
                    # 如果是最喜欢的品牌，评分更高
                    if device['brand'] == preferred_brands[0]:
                        rating = np.random.randint(4, 6)
                else:
                    # 如果没有偏好品牌的设备，随机选择
                    device = devices.sample(1).iloc[0]
                    rating = np.random.randint(2, 5)
            else:
                # 20%概率选择非偏好品牌的设备
                non_preferred_devices = devices[~devices['brand'].isin(preferred_brands)]
                if len(non_preferred_devices) > 0:
                    device = non_preferred_devices.sample(1).iloc[0]
                    # 非偏好品牌的评分较低（1-3分）
                    rating = np.random.randint(1, 4)
                else:
                    device = devices.sample(1).iloc[0]
                    rating = np.random.randint(1, 4)
            
            interactions_data.append({
                'user_id': user_id,
                'device_id': device['device_id'],
                'rating': rating
            })
    
    interactions = pd.DataFrame(interactions_data)
    
    # === 数据统计信息 ===
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 数据生成完成!")
    logger.info(f"{'='*50}")
    logger.info(f"👥 用户数量: {len(users):,}")
    logger.info(f"📱 设备数量: {len(devices):,}")
    logger.info(f"🔄 交互数量: {len(interactions):,}")
    logger.info(f"📈 平均每用户交互数: {len(interactions) / len(users):.1f}")
    
    # 品牌分布统计
    brand_dist = interactions.merge(devices, on='device_id')['brand'].value_counts()
    logger.info(f"\n🏷️ 品牌交互分布:")
    for brand, count in brand_dist.items():
        percentage = (count / len(interactions)) * 100
        logger.info(f"  📱 {brand}: {count:,}次 ({percentage:.1f}%)")
    
    # 评分分布统计
    rating_dist = interactions['rating'].value_counts().sort_index()
    logger.info(f"\n⭐ 评分分布:")
    for rating, count in rating_dist.items():
        percentage = (count / len(interactions)) * 100
        stars = "⭐" * int(rating)
        logger.info(f"  {stars} {rating}分: {count:,}次 ({percentage:.1f}%)")
    
    logger.info(f"{'='*50}")
    
    return users, devices, interactions

def analyze_generated_data(users, devices, interactions):
    """
    分析生成的数据质量和用户偏好分布
    
    这个函数用于验证生成的数据是否符合预期的用户偏好模式
    """
    logger.info("\n=== 数据质量分析 ===")
    
    # 合并数据以便分析
    interaction_with_device = interactions.merge(devices, on='device_id')
    
    # 1. 用户品牌偏好分析
    logger.info("1. 用户品牌偏好分析:")
    user_brand_stats = interaction_with_device.groupby(['user_id', 'brand']).agg({
        'rating': ['count', 'mean']
    }).reset_index()
    user_brand_stats.columns = ['user_id', 'brand', 'interaction_count', 'avg_rating']
    
    # 找出每个用户最喜欢的品牌
    user_favorite_brands = user_brand_stats.loc[user_brand_stats.groupby('user_id')['avg_rating'].idxmax()]
    
    # 统计品牌忠诚度
    brand_loyalty = user_favorite_brands['brand'].value_counts()
    logger.info(f"  品牌忠诚度分布: {dict(brand_loyalty)}")
    
    # 2. 评分质量分析
    logger.info("2. 评分质量分析:")
    
    # 计算每个用户对偏好品牌vs非偏好品牌的平均评分
    user_brand_preference_scores = []
    
    for user_id in interactions['user_id'].unique():
        user_interactions = interaction_with_device[interaction_with_device['user_id'] == user_id]
        
        if len(user_interactions) < 2:
            continue
            
        # 找出用户最常交互的品牌
        brand_counts = user_interactions['brand'].value_counts()
        if len(brand_counts) == 0:
            continue
            
        favorite_brand = brand_counts.index[0]
        
        # 计算对偏好品牌和非偏好品牌的平均评分
        preferred_ratings = user_interactions[user_interactions['brand'] == favorite_brand]['rating']
        non_preferred_ratings = user_interactions[user_interactions['brand'] != favorite_brand]['rating']
        
        if len(preferred_ratings) > 0 and len(non_preferred_ratings) > 0:
            user_brand_preference_scores.append({
                'user_id': user_id,
                'favorite_brand': favorite_brand,
                'preferred_avg_rating': preferred_ratings.mean(),
                'non_preferred_avg_rating': non_preferred_ratings.mean(),
                'rating_difference': preferred_ratings.mean() - non_preferred_ratings.mean()
            })
    
    if user_brand_preference_scores:
        preference_df = pd.DataFrame(user_brand_preference_scores)
        avg_preferred_rating = preference_df['preferred_avg_rating'].mean()
        avg_non_preferred_rating = preference_df['non_preferred_avg_rating'].mean()
        avg_rating_difference = preference_df['rating_difference'].mean()
        
        logger.info(f"  偏好品牌平均评分: {avg_preferred_rating:.2f}")
        logger.info(f"  非偏好品牌平均评分: {avg_non_preferred_rating:.2f}")
        logger.info(f"  评分差异: {avg_rating_difference:.2f}")
        
        # 统计有多少用户对偏好品牌评分更高
        positive_preference_users = len(preference_df[preference_df['rating_difference'] > 0])
        total_users = len(preference_df)
        logger.info(f"  偏好品牌评分更高的用户比例: {positive_preference_users}/{total_users} ({positive_preference_users/total_users*100:.1f}%)")
    
    # 3. 年龄与品牌偏好关系分析
    logger.info("3. 年龄与品牌偏好关系:")
    
    # 合并用户年龄信息
    user_age_brand = user_favorite_brands.merge(users[['user_id', 'age']], on='user_id')
    
    # 按年龄段分析品牌偏好
    age_groups = [
        (18, 25, "年轻用户"),
        (25, 35, "中年用户"),
        (35, 65, "成熟用户")
    ]
    
    for min_age, max_age, group_name in age_groups:
        group_users = user_age_brand[(user_age_brand['age'] >= min_age) & (user_age_brand['age'] < max_age)]
        if len(group_users) > 0:
            brand_dist = group_users['brand'].value_counts()
            logger.info(f"  {group_name} ({min_age}-{max_age}岁): {dict(brand_dist)}")
    
    # 4. 价格偏好分析
    logger.info("4. 价格偏好分析:")
    
    # 计算每个品牌的平均价格
    brand_avg_prices = devices.groupby('brand')['price'].mean().sort_values(ascending=False)
    logger.info(f"  品牌平均价格排序: {dict(brand_avg_prices.round(0))}")
    
    # 计算用户交互的平均价格
    user_avg_prices = interaction_with_device.groupby('user_id')['price'].mean()
    logger.info(f"  用户平均交互价格: {user_avg_prices.mean():.0f}元 (标准差: {user_avg_prices.std():.0f}元)")
    
    return {
        'brand_loyalty': dict(brand_loyalty),
        'preference_scores': user_brand_preference_scores,
        'age_brand_preference': user_age_brand,
        'brand_avg_prices': dict(brand_avg_prices)
    }

def user_history_demo():
    """
    用户历史记录演示
    
    这个函数专门演示如何查看和分析用户的历史交互记录：
    1. 生成测试数据
    2. 训练推荐模型
    3. 查看用户历史交互记录
    4. 分析用户购买偏好
    """
    print("=== 用户历史记录演示 ===")
    
    # 生成测试数据
    users, devices, interactions = generate_sample_data()
    
    # 创建并训练推荐系统
    recommender = SecondHandRecommendationSystem()
    print("正在训练推荐模型...")
    recommender.train(users, devices, interactions, epochs=5)
    
    # 选择几个不同的用户进行演示
    test_users = [12, 25, 50, 100]
    
    for user_id in test_users:
        print(f"\n{'='*60}")
        print(f"正在查看用户 {user_id} 的历史记录...")
        
        # 打印用户历史交互记录
        recommender.print_user_interaction_history(
            user_id=user_id,
            device_data=devices,
            interaction_data=interactions
        )
        
        # 简单暂停，让用户能看清楚每个用户的信息
        input("按回车键继续查看下一个用户...")
    
    print("\n用户历史记录演示完成！")


def simple_recommendation_demo():
    """
    简单的推荐演示
    
    这个函数展示了如何使用新的用户推荐功能：
    1. 生成测试数据
    2. 训练推荐模型
    3. 为用户推荐Top5设备
    4. 显示推荐结果
    """
    print("=== 简单推荐演示 ===")
    
    # 生成测试数据
    users, devices, interactions = generate_sample_data()
    
    # 创建并训练推荐系统
    recommender = SecondHandRecommendationSystem()
    print("正在训练推荐模型...")
    recommender.train(users, devices, interactions, epochs=10)
    
    # 为用户推荐设备
    user_id = 25
    
    # 首先打印用户历史交互记录
    print(f"\n=== 用户 {user_id} 的历史记录 ===")
    recommender.print_user_interaction_history(
        user_id=user_id,
        device_data=devices,
        interaction_data=interactions
    )
    
    # 然后进行个性化推荐
    print(f"\n为用户 {user_id} 推荐设备:")
    recommendations = recommender.recommend_top_n_devices_for_user(
        user_id=user_id,
        device_data=devices,
        interaction_data=interactions,
        top_n=5,
        exclude_interacted=True
    )
    
    if recommendations:
        print(f"\n推荐结果 (Top 5):")
        for rec in recommendations:
            device_info = rec['device_info']
            print(f"第{rec['rank']}名: 设备{rec['device_id']} | "
                  f"品牌: {device_info['brand']} | "
                  f"价格: {device_info['price']:.0f}元 | "
                  f"成色: {device_info['condition']} | "
                  f"推荐评分: {rec['score']:.3f}")
    else:
        print("没有找到合适的推荐设备")
    
    # 显示推荐摘要
    summary = recommender.get_user_recommendation_summary(
        user_id=user_id,
        device_data=devices,
        interaction_data=interactions,
        top_n=5
    )
    
    print(f"\n推荐摘要:")
    print(f"- 推荐设备数量: {summary['summary']['total_recommendations']}")
    print(f"- 平均推荐评分: {summary['summary']['avg_score']:.3f}")
    print(f"- 最受推荐的品牌: {summary['summary']['top_brand']}")
    print(f"- 价格范围: {summary['summary']['price_range']['min']:.0f} - {summary['summary']['price_range']['max']:.0f}元")
    print(f"- 平均价格: {summary['summary']['price_range']['avg']:.0f}元")
    

def brand_weight_comparison_demo():
    """
    品牌权重对比演示
    
    这个函数展示不同品牌权重设置对推荐结果的影响，
    帮助理解模型学习vs规则增强的平衡点。
    """
    logger.info("\n" + "="*60)
    logger.info("=== 品牌权重对比演示 ===")
    logger.info("="*60)
    
    # 生成测试数据
    users, devices, interactions = generate_sample_data()
    
    # 创建并训练推荐系统
    recommender = SecondHandRecommendationSystem()
    logger.info("正在训练推荐模型...")
    recommender.train(users, devices, interactions, epochs=20)
    
    # 选择一个有明确品牌偏好的用户进行测试
    test_user_id = 12
    
    # 显示用户历史记录
    logger.info(f"\n=== 测试用户 {test_user_id} 的历史记录 ===")
    recommender.print_user_interaction_history(
        user_id=test_user_id,
        device_data=devices,
        interaction_data=interactions
    )
    
    # 测试不同的品牌权重设置
    weight_settings = [
        (0.0, "完全依赖模型学习"),
        (0.01, "微量品牌偏好增强（1%）"),
        (0.02, "少量品牌偏好增强（2%）"),
        (0.05, "中等品牌偏好增强（5%）"),
        (0.1, "强品牌偏好增强（10%）")
    ]
    
    logger.info(f"\n=== 不同品牌权重设置的推荐结果对比 ===")
    
    for weight, description in weight_settings:
        logger.info(f"\n📊 {description} (brand_weight={weight})")
        logger.info("-" * 50)
        
        recommendations = recommender.recommend_top_n_devices_for_user(
            user_id=test_user_id,
            device_data=devices,
            interaction_data=interactions,
            top_n=5,
            exclude_interacted=True,
            brand_weight=weight
        )
        
        if recommendations:
            # 统计推荐结果中的品牌分布
            brands = [rec['device_info']['brand'] for rec in recommendations]
            brand_counts = pd.Series(brands).value_counts()
            
            logger.info(f"推荐品牌分布: {dict(brand_counts)}")
            
            # 显示详细推荐结果
            for i, rec in enumerate(recommendations, 1):
                device_info = rec['device_info']
                logger.info(f"  {i}. 设备{rec['device_id']} | 品牌:{device_info['brand']} | "
                           f"价格:{device_info['price']:.0f}元 | 评分:{rec['score']:.3f}")
        else:
            logger.info("  没有推荐结果")
    
    logger.info(f"\n=== 权重设置建议 ===")
    logger.info("📝 学习要点:")
    logger.info("  • brand_weight=0.0: 完全依赖深度学习模型，适合高质量训练数据")
    logger.info("  • brand_weight=0.01-0.02: 轻微增强，平衡模型学习和用户偏好")
    logger.info("  • brand_weight=0.05-0.1: 强化用户偏好，适合数据稀疏场景")
    logger.info("  • 实际应用中，可以根据用户交互数据量动态调整权重")

def main():
    """
    主函数 - 演示推荐系统的完整功能
    
    这个函数展示了如何使用二手设备推荐系统：
    1. 生成模拟数据
    2. 训练推荐模型
    3. 测试推荐功能
    4. 显示推荐结果
    
    这是一个完整的端到端演示，展示了推荐系统的核心功能。
    """
    logger.info("=== 二手设备推荐系统演示 ===")
    
    # === 第一步：生成演示数据 ===
    # 创建模拟的用户、设备和交互数据
    users, devices, interactions = generate_sample_data()
    
    logger.info(f"用户数量: {len(users)}")
    logger.info(f"设备数量: {len(devices)}")
    logger.info(f"交互数量: {len(interactions)}")
    
    # === 数据质量分析 ===
    analyze_generated_data(users, devices, interactions)
    
    # === 第二步：创建推荐系统 ===
    # 初始化推荐系统实例
    recommender = SecondHandRecommendationSystem()
    
    # === 第三步：训练模型 ===
    # 使用准备好的数据训练深度学习模型
    recommender.train(users, devices, interactions, epochs=20)
    
    # === 第四步：测试推荐功能 ===
    
    # 测试1：打印用户历史交互记录
    test_user_id = 12
    recommender.print_user_interaction_history(
        user_id=test_user_id,
        device_data=devices,
        interaction_data=interactions
    )
    
    # 测试2：用户个性化推荐（使用减少的权重）
    user_recommendations = recommender.recommend_top_n_devices_for_user(
        user_id=test_user_id, 
        device_data=devices, 
        interaction_data=interactions, 
        top_n=5,
        exclude_interacted=True,
        brand_weight=0.02  # 使用减少的权重
    )
    
    logger.info(f"\n=== 为用户 {test_user_id} 推荐的Top5设备 ===")
    for rec in user_recommendations:
        device_info = rec['device_info']
        logger.info(f"排名{rec['rank']}: 设备{rec['device_id']} | 品牌: {device_info['brand']} | "
                   f"价格: {device_info['price']:.0f}元 | 评分: {rec['score']:.3f}")
    
    # 测试3：推荐摘要
    recommendation_summary = recommender.get_user_recommendation_summary(
        user_id=test_user_id,
        device_data=devices,
        interaction_data=interactions,
        top_n=5
    )
    
    logger.info(f"\n=== 用户 {test_user_id} 推荐摘要 ===")
    summary = recommendation_summary['summary']
    logger.info(f"推荐设备数量: {summary['total_recommendations']}")
    logger.info(f"平均评分: {summary['avg_score']:.3f}")
    logger.info(f"用户历史偏好品牌: {summary['user_favorite_brand']}")
    logger.info(f"推荐结果中最多品牌: {summary['top_brand']}")
    logger.info(f"价格范围: {summary['price_range']['min']:.0f} - {summary['price_range']['max']:.0f}元")
    logger.info(f"平均价格: {summary['price_range']['avg']:.0f}元")
    
    # 测试4：相似设备推荐
    test_device_id = 0
    similar_devices = recommender.recommend_similar_devices(test_device_id, top_k=5)
    
    logger.info(f"\n=== 与设备 {test_device_id} 相似的设备 ===")
    target_device = devices[devices['device_id'] == test_device_id].iloc[0]
    logger.info(f"目标设备: 品牌={target_device['brand']}, 价格={target_device['price']:.0f}元")
    
    for i, device_id in enumerate(similar_devices, 1):
        device_info = devices[devices['device_id'] == device_id].iloc[0]
        logger.info(f"相似设备{i}: 设备{device_id} | 品牌={device_info['brand']}, 价格={device_info['price']:.0f}元")
    
    # === 第五步：品牌权重对比演示 ===
    brand_weight_comparison_demo()


if __name__ == "__main__":
    main()
    
    # 也可以运行其他演示
    # simple_recommendation_demo()    # 简单推荐演示
    # user_history_demo()            # 用户历史记录演示 