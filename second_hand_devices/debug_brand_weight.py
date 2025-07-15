#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
品牌权重调试脚本

用于分析为什么brand_weight < 0.02时没有联想推荐的原因
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from second_hand_device_recommender import SecondHandRecommendationSystem, generate_sample_data
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_brand_weight_effect():
    """分析品牌权重对推荐结果的影响"""
    
    print("🔍 品牌权重影响分析")
    print("=" * 60)
    
    # 生成数据并训练模型
    users, devices, interactions = generate_sample_data()
    recommender = SecondHandRecommendationSystem()
    recommender.train(users, devices, interactions, epochs=20)
    
    # 分析用户12
    test_user_id = 12
    user_interactions = interactions[interactions['user_id'] == test_user_id]
    
    print(f"\n📊 用户 {test_user_id} 的交互记录:")
    interactions_with_brands = user_interactions.merge(devices[['device_id', 'brand']], on='device_id')
    for _, row in interactions_with_brands.iterrows():
        print(f"  设备{row['device_id']} | 品牌:{row['brand']} | 评分:{row['rating']}")
    
    # 计算品牌偏好权重
    brand_stats = interactions_with_brands.groupby('brand').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    brand_stats.columns = ['brand', 'interaction_count', 'avg_rating']
    
    print(f"\n🎯 品牌偏好权重计算:")
    user_brand_preferences = {}
    for _, row in brand_stats.iterrows():
        brand = row['brand']
        weight = row['interaction_count'] * row['avg_rating'] / 5.0
        user_brand_preferences[brand] = weight
        print(f"  {brand}: {row['interaction_count']}次 × {row['avg_rating']:.2f}分 ÷ 5.0 = {weight:.3f}")
    
    # 获取所有联想和华为设备的原始评分
    print(f"\n⚖️ 模型原始评分对比:")
    
    # 模拟推荐过程
    recommender.model.eval()
    user_encoded = recommender.user_encoder.transform([test_user_id])[0]
    user_tensor = torch.LongTensor([user_encoded])
    
    # 分析联想和华为设备的评分
    lenovo_scores = []
    huawei_scores = []
    
    for _, device_row in devices.iterrows():
        device_id = device_row['device_id']
        brand = device_row['brand']
        
        # 跳过已交互的设备
        if device_id in user_interactions['device_id'].values:
            continue
            
        try:
            device_encoded = recommender.device_encoder.transform([device_id])[0]
            brand_encoded = recommender.brand_encoder.transform([brand])[0]
            
            device_tensor = torch.LongTensor([device_encoded])
            brand_tensor = torch.LongTensor([brand_encoded])
            
            with torch.no_grad():
                score = recommender.model(user_tensor, device_tensor, brand_tensor)
                score = torch.sigmoid(score).item()
                
                if brand == '联想':
                    lenovo_scores.append((device_id, score))
                elif brand == '华为':
                    huawei_scores.append((device_id, score))
        except:
            continue
    
    # 排序并显示前5个
    lenovo_scores.sort(key=lambda x: x[1], reverse=True)
    huawei_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 联想设备原始评分 (Top 5):")
    for i, (device_id, score) in enumerate(lenovo_scores[:5], 1):
        print(f"  {i}. 设备{device_id}: {score:.4f}")
    
    print(f"\n📈 华为设备原始评分 (Top 5):")
    for i, (device_id, score) in enumerate(huawei_scores[:5], 1):
        print(f"  {i}. 设备{device_id}: {score:.4f}")
    
    # 分析不同权重设置的影响
    print(f"\n🎚️ 不同权重设置的影响分析:")
    
    if '联想' in user_brand_preferences:
        lenovo_preference = user_brand_preferences['联想']
        
        print(f"\n联想品牌偏好权重: {lenovo_preference:.3f}")
        
        # 计算不同brand_weight设置下的额外分数
        for brand_weight in [0.01, 0.02, 0.05]:
            brand_bonus = lenovo_preference * brand_weight
            print(f"\nbrand_weight={brand_weight}:")
            print(f"  额外分数: {lenovo_preference:.3f} × {brand_weight} = {brand_bonus:.4f}")
            
            # 显示加权后的联想设备评分
            if lenovo_scores:
                best_lenovo_score = lenovo_scores[0][1]
                adjusted_score = min(1.0, best_lenovo_score + brand_bonus)
                print(f"  最佳联想设备调整后评分: {best_lenovo_score:.4f} + {brand_bonus:.4f} = {adjusted_score:.4f}")
                
                # 与最佳华为设备对比
                if huawei_scores:
                    best_huawei_score = huawei_scores[0][1]
                    print(f"  最佳华为设备评分: {best_huawei_score:.4f}")
                    print(f"  联想是否能超过华为: {'是' if adjusted_score > best_huawei_score else '否'}")
    
    # 关键洞察
    print(f"\n💡 关键洞察:")
    print(f"  1. 模型原始评分差异决定了需要多大的权重才能改变排序")
    print(f"  2. 如果华为设备的原始评分比联想高很多，需要较大权重才能逆转")
    print(f"  3. 品牌权重的作用是微调，不是颠覆性改变")
    print(f"  4. 当权重太小时，无法弥补模型学习到的评分差异")

if __name__ == "__main__":
    analyze_brand_weight_effect() 