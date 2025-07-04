#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手设备推荐系统 - 基础使用示例

这个示例展示了如何使用二手设备推荐系统的基本功能：
1. 数据准备
2. 模型训练
3. 推荐生成
4. 结果评估
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from second_hand_device_recommender import SecondHandDeviceRecommender, SecondHandRecommendationSystem

def generate_sample_data():
    """生成示例数据"""
    print("📊 生成示例数据...")
    
    # 生成用户数据
    users = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 65, 100),
        'city': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], 100),
        'latitude': np.random.uniform(39.0, 41.0, 100),
        'longitude': np.random.uniform(116.0, 120.0, 100)
    })
    
    # 生成设备数据
    brands = ['苹果', '华为', '小米', '三星', 'OPPO', 'vivo', '联想', '戴尔']
    categories = ['手机', '笔记本', '平板', '智能手表', '耳机']
    conditions = ['全新', '九成新', '八成新', '七成新']
    
    devices = pd.DataFrame({
        'device_id': range(1, 501),
        'brand': np.random.choice(brands, 500),
        'category': np.random.choice(categories, 500),
        'condition': np.random.choice(conditions, 500),
        'price': np.random.randint(500, 10000, 500),
        'age_months': np.random.randint(0, 60, 500),
        'storage_gb': np.random.choice([64, 128, 256, 512, 1024], 500),
        'ram_gb': np.random.choice([4, 6, 8, 12, 16], 500),
        'screen_size': np.random.uniform(5.0, 17.0, 500)
    })
    
    # 生成交互数据
    interactions = []
    for _ in range(2000):
        user_id = np.random.randint(1, 101)
        device_id = np.random.randint(1, 501)
        interaction_type = np.random.choice(['view', 'like', 'purchase'], p=[0.7, 0.2, 0.1])
        rating = np.random.randint(1, 6) if interaction_type in ['like', 'purchase'] else None
        
        interactions.append({
            'user_id': user_id,
            'device_id': device_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
        })
    
    interactions = pd.DataFrame(interactions)
    
    return users, devices, interactions

def basic_recommendation_demo():
    """基础推荐演示"""
    print("🚀 开始基础推荐演示...")
    
    # 1. 生成示例数据
    users, devices, interactions = generate_sample_data()
    
    # 2. 创建推荐系统
    recommender = SecondHandRecommendationSystem()
    
    # 3. 训练模型
    print("\n🎯 训练推荐模型...")
    recommender.train(users, devices, interactions)
    
    # 4. 为用户推荐设备
    print("\n💡 为用户生成个性化推荐...")
    user_id = 1
    recommendations = recommender.recommend_for_user(user_id, k=5)
    print(f"为用户 {user_id} 推荐的设备:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. 设备ID: {rec['device_id']}, 评分: {rec['score']:.3f}")
    
    # 5. 查找相似设备
    print("\n🔍 查找相似设备...")
    device_id = 1
    similar_devices = recommender.recommend_similar_devices(device_id, k=5)
    print(f"与设备 {device_id} 相似的设备:")
    for i, sim in enumerate(similar_devices, 1):
        print(f"  {i}. 设备ID: {sim['device_id']}, 相似度: {sim['similarity']:.3f}")
    
    # 6. 基于地理位置的推荐
    print("\n📍 基于地理位置的推荐...")
    location_recs = recommender.recommend_by_location(user_id, k=5)
    print(f"为用户 {user_id} 推荐的附近设备:")
    for i, rec in enumerate(location_recs, 1):
        distance = rec.get('distance', 0)
        print(f"  {i}. 设备ID: {rec['device_id']}, 距离: {distance:.1f}km")
    
    # 7. 评估推荐质量
    print("\n📊 评估推荐质量...")
    metrics = recommender.evaluate_recommendations(interactions)
    print("推荐质量指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return recommender

def price_recommendation_demo():
    """价格推荐演示"""
    print("\n💰 价格推荐演示...")
    
    # 创建一个简单的价格推荐器
    class PriceRecommender:
        def __init__(self):
            self.price_model = None
        
        def train(self, devices):
            # 简单的价格模型：基于设备特征预测价格
            self.device_features = devices.groupby(['brand', 'category', 'condition']).agg({
                'price': ['mean', 'std', 'min', 'max'],
                'age_months': 'mean'
            }).reset_index()
            
        def recommend_price(self, brand, category, condition, age_months):
            # 查找相似设备的价格统计
            similar = self.device_features[
                (self.device_features['brand'] == brand) & 
                (self.device_features['category'] == category) & 
                (self.device_features['condition'] == condition)
            ]
            
            if len(similar) > 0:
                base_price = similar[('price', 'mean')].iloc[0]
                # 根据设备年龄调整价格
                age_factor = max(0.5, 1 - age_months / 60)  # 5年后价格降到50%
                recommended_price = base_price * age_factor
                
                return {
                    'recommended_price': recommended_price,
                    'price_range': {
                        'min': similar[('price', 'min')].iloc[0] * age_factor,
                        'max': similar[('price', 'max')].iloc[0] * age_factor
                    },
                    'market_average': base_price
                }
            else:
                return None
    
    # 使用价格推荐器
    users, devices, interactions = generate_sample_data()
    price_recommender = PriceRecommender()
    price_recommender.train(devices)
    
    # 为新设备推荐价格
    device_specs = {
        'brand': '苹果',
        'category': '手机',
        'condition': '八成新',
        'age_months': 12
    }
    
    price_rec = price_recommender.recommend_price(**device_specs)
    if price_rec:
        print(f"设备规格: {device_specs}")
        print(f"推荐价格: ¥{price_rec['recommended_price']:.0f}")
        print(f"价格区间: ¥{price_rec['price_range']['min']:.0f} - ¥{price_rec['price_range']['max']:.0f}")
        print(f"市场均价: ¥{price_rec['market_average']:.0f}")

def main():
    """主函数"""
    print("🎉 二手设备推荐系统 - 基础使用示例")
    print("=" * 50)
    
    # 基础推荐演示
    recommender = basic_recommendation_demo()
    
    # 价格推荐演示
    price_recommendation_demo()
    
    print("\n✅ 基础使用示例完成！")
    print("\n📚 接下来可以尝试:")
    print("  1. 运行 advanced_features.py 查看高级功能")
    print("  2. 运行 deployment_example.py 了解部署方案")
    print("  3. 查看 config.yaml 自定义配置")

if __name__ == "__main__":
    main() 