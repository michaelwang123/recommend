#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手设备推荐系统 - 演示脚本

展示推荐系统的主要功能
"""

import pandas as pd
import numpy as np
from second_hand_device_recommender import (
    SecondHandRecommendationSystem, 
    PriceRecommender, 
    LocationRecommender
)

print("🎉 二手设备推荐系统演示")
print("=" * 50)

# 1. 生成演示数据
print("📊 生成演示数据...")
users = pd.DataFrame({
    'user_id': range(10),
    'age': [25, 30, 35, 28, 32, 29, 31, 26, 33, 27],
    'city': ['北京', '上海', '广州', '深圳', '杭州', '南京', '苏州', '武汉', '成都', '重庆']
})

devices = pd.DataFrame({
    'device_id': range(20),
    'brand': ['苹果', '华为', '小米', '三星', '联想'] * 4,
    'category': ['手机', '手机', '笔记本', '平板', '手机'] * 4,
    'condition': ['九成新', '八成新', '全新', '七成新', '九成新'] * 4,
    'price': [4500, 3200, 6800, 2800, 1900, 5500, 4200, 7200, 3800, 2200,
              4800, 3500, 7000, 3000, 2100, 5200, 4000, 6900, 3600, 2400],
    'age_months': [6, 12, 3, 18, 24, 8, 15, 2, 20, 30, 7, 14, 4, 16, 25, 9, 13, 5, 19, 28],
    'storage_gb': [128, 64, 256, 128, 64, 256, 128, 512, 64, 128, 256, 128, 512, 64, 128, 256, 128, 512, 64, 256],
    'ram_gb': [8, 6, 16, 8, 4, 16, 8, 32, 6, 8, 16, 8, 32, 4, 8, 16, 8, 32, 6, 16],
    'screen_size': [6.1, 5.5, 14.0, 10.9, 5.0, 13.3, 6.5, 15.6, 5.8, 11.0, 
                   6.2, 5.7, 14.5, 10.5, 5.2, 13.8, 6.3, 15.4, 5.9, 11.5]
})

interactions = pd.DataFrame({
    'user_id': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4] * 5,
    'device_id': np.random.randint(0, 20, 100),
    'interaction_type': ['view', 'like', 'purchase'] * 33 + ['view'],
    'rating': np.random.randint(3, 6, 100),
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='H')
})

print(f"✅ 数据生成完成: {len(users)}个用户, {len(devices)}个设备, {len(interactions)}个交互")

# 2. 训练推荐系统
print("\n🚀 训练推荐系统...")
recommender = SecondHandRecommendationSystem()
recommender.train(users, devices, interactions, epochs=10)

# 3. 用户推荐演示
print("\n💡 用户推荐演示")
print("-" * 30)
for user_id in [0, 1, 2]:
    print(f"\n用户 {user_id} 的推荐:")
    user_info = users[users['user_id'] == user_id].iloc[0]
    print(f"  用户信息: {user_info['age']}岁, {user_info['city']}")
    
    recommendations = recommender.recommend_for_user(user_id, k=3)
    
    for i, rec in enumerate(recommendations, 1):
        device_info = devices[devices['device_id'] == rec['device_id']].iloc[0]
        print(f"  {i}. {device_info['brand']} {device_info['category']} - "
              f"{device_info['price']}元 ({device_info['condition']}) "
              f"[评分: {rec['score']:.3f}]")

# 4. 相似设备推荐演示
print("\n🔍 相似设备推荐演示")
print("-" * 30)
target_devices = [0, 5, 10]
for device_id in target_devices:
    device_info = devices[devices['device_id'] == device_id].iloc[0]
    print(f"\n目标设备: {device_info['brand']} {device_info['category']} - "
          f"{device_info['price']}元 ({device_info['condition']})")
    
    similar_devices = recommender.recommend_similar_devices(device_id, top_k=3)
    
    for i, sim_device_id in enumerate(similar_devices, 1):
        sim_device_info = devices[devices['device_id'] == sim_device_id].iloc[0]
        print(f"  {i}. {sim_device_info['brand']} {sim_device_info['category']} - "
              f"{sim_device_info['price']}元 ({sim_device_info['condition']})")

# 5. 价格推荐演示
print("\n💰 价格推荐演示")
print("-" * 30)
price_recommender = PriceRecommender()
price_recommender.train_price_model(devices)

test_devices = [
    {'brand': '苹果', 'category': '手机', 'condition': '九成新', 'age_months': 6, 
     'storage_gb': 128, 'ram_gb': 8, 'screen_size': 6.1, 'price': 4500},
    {'brand': '华为', 'category': '手机', 'condition': '八成新', 'age_months': 12, 
     'storage_gb': 64, 'ram_gb': 6, 'screen_size': 5.5, 'price': 3200},
    {'brand': '小米', 'category': '笔记本', 'condition': '全新', 'age_months': 3, 
     'storage_gb': 256, 'ram_gb': 16, 'screen_size': 14.0, 'price': 6800}
]

for device in test_devices:
    print(f"\n{device['brand']} {device['category']} ({device['condition']}):")
    price_info = price_recommender.recommend_price(device)
    print(f"  实际价格: {device['price']}元")
    print(f"  推荐价格: {price_info['recommended_price']:.0f}元")
    print(f"  价格区间: {price_info['min_price']:.0f}-{price_info['max_price']:.0f}元")

# 6. 地理位置推荐演示
print("\n🌍 地理位置推荐演示")
print("-" * 30)

# 添加地理位置数据
np.random.seed(42)
devices['latitude'] = np.random.uniform(39.8, 40.2, len(devices))   # 北京地区
devices['longitude'] = np.random.uniform(116.2, 116.6, len(devices))

location_recommender = LocationRecommender(max_distance_km=10)
user_location = (40.0, 116.4)  # 北京市中心

print(f"用户位置: {user_location}")
nearby_devices = location_recommender.recommend_nearby_devices(
    user_location, devices, top_k=5
)

print("附近设备推荐:")
for i, device in enumerate(nearby_devices, 1):
    device_info = device['device_info']
    print(f"  {i}. {device_info['brand']} {device_info['category']} - "
          f"{device_info['price']}元 (距离: {device['distance']:.1f}km)")

print("\n🎉 演示完成！")
print("=" * 50)
print("推荐系统已成功运行，所有功能正常！")
print("您可以:")
print("1. 运行 python test_recommender.py 进行完整测试")
print("2. 运行 python examples/deployment_example.py 启动Web服务")
print("3. 查看 PROJECT_SUMMARY.md 了解详细功能")
print("4. 查看 QUICK_START.md 了解快速开始指南") 