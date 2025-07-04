#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手设备推荐系统 - 测试脚本

快速测试推荐系统的核心功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
from second_hand_device_recommender import SecondHandRecommendationSystem, PriceRecommender

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data():
    """生成测试数据"""
    logger.info("📊 生成测试数据...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 用户数据
    n_users = 100
    users = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(20, 60, n_users),
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], n_users)
    })
    
    # 设备数据
    n_devices = 50
    brands = ['苹果', '华为', '小米', '三星', '联想']
    categories = ['手机', '笔记本', '平板']
    conditions = ['全新', '九成新', '八成新', '七成新']
    
    devices = pd.DataFrame({
        'device_id': range(n_devices),
        'brand': np.random.choice(brands, n_devices),
        'category': np.random.choice(categories, n_devices),
        'condition': np.random.choice(conditions, n_devices),
        'price': np.random.uniform(1000, 8000, n_devices),
        'age_months': np.random.randint(1, 36, n_devices),
        'storage_gb': np.random.choice([64, 128, 256, 512], n_devices),
        'ram_gb': np.random.choice([4, 8, 16, 32], n_devices),
        'screen_size': np.random.uniform(5.0, 15.6, n_devices)
    })
    
    # 交互数据
    n_interactions = 500
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'device_id': np.random.randint(0, n_devices, n_interactions),
        'interaction_type': np.random.choice(['view', 'like', 'purchase'], n_interactions),
        'rating': np.random.randint(1, 6, n_interactions),
        'timestamp': pd.date_range('2023-01-01', periods=n_interactions, freq='H')
    })
    
    logger.info(f"✅ 生成数据完成 - 用户: {len(users)}, 设备: {len(devices)}, 交互: {len(interactions)}")
    return users, devices, interactions

def test_recommendation_system():
    """测试推荐系统"""
    logger.info("🎯 开始测试推荐系统...")
    
    # 生成测试数据
    users, devices, interactions = generate_test_data()
    
    # 创建推荐系统
    recommender = SecondHandRecommendationSystem()
    
    try:
        # 训练模型
        logger.info("🚀 开始训练模型...")
        recommender.train(users, devices, interactions, epochs=20)
        
        # 测试用户推荐
        logger.info("💡 测试用户推荐...")
        test_user_id = 1
        user_recommendations = recommender.recommend_for_user(test_user_id, k=5)
        
        logger.info(f"为用户 {test_user_id} 推荐的设备:")
        for i, rec in enumerate(user_recommendations, 1):
            device_info = devices[devices['device_id'] == rec['device_id']].iloc[0]
            logger.info(f"  {i}. 设备ID: {rec['device_id']}, 品牌: {device_info['brand']}, "
                       f"类别: {device_info['category']}, 价格: {device_info['price']:.0f}元, "
                       f"评分: {rec['score']:.3f}")
        
        # 测试相似设备推荐
        logger.info("🔍 测试相似设备推荐...")
        test_device_id = 0
        similar_devices = recommender.recommend_similar_devices(test_device_id, top_k=5)
        
        test_device = devices[devices['device_id'] == test_device_id].iloc[0]
        logger.info(f"目标设备: ID={test_device_id}, 品牌={test_device['brand']}, "
                   f"类别={test_device['category']}, 价格={test_device['price']:.0f}元")
        
        logger.info("相似设备推荐:")
        for i, device_id in enumerate(similar_devices, 1):
            device_info = devices[devices['device_id'] == device_id].iloc[0]
            logger.info(f"  {i}. 设备ID: {device_id}, 品牌: {device_info['brand']}, "
                       f"类别: {device_info['category']}, 价格: {device_info['price']:.0f}元")
        
        logger.info("✅ 推荐系统测试完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 推荐系统测试失败: {str(e)}")
        return False

def test_price_recommender():
    """测试价格推荐器"""
    logger.info("💰 开始测试价格推荐器...")
    
    try:
        # 生成测试数据
        users, devices, interactions = generate_test_data()
        
        # 创建价格推荐器
        price_recommender = PriceRecommender()
        
        # 训练价格模型
        logger.info("🚀 训练价格预测模型...")
        price_recommender.train_price_model(devices)
        
        # 测试价格推荐
        test_device = {
            'brand': '苹果',
            'category': '手机',
            'condition': '九成新',
            'age_months': 12,
            'storage_gb': 128,
            'ram_gb': 8,
            'screen_size': 6.1,
            'price': 5000  # 实际价格用于训练
        }
        
        price_info = price_recommender.recommend_price(test_device)
        
        logger.info("价格推荐结果:")
        logger.info(f"  最低价格: {price_info['min_price']:.0f}元")
        logger.info(f"  推荐价格: {price_info['recommended_price']:.0f}元")
        logger.info(f"  最高价格: {price_info['max_price']:.0f}元")
        
        logger.info("✅ 价格推荐器测试完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 价格推荐器测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    logger.info("🎉 二手设备推荐系统 - 测试开始")
    logger.info("=" * 50)
    
    # 测试推荐系统
    rec_success = test_recommendation_system()
    
    logger.info("=" * 50)
    
    # 测试价格推荐器
    price_success = test_price_recommender()
    
    logger.info("=" * 50)
    
    # 测试结果总结
    if rec_success and price_success:
        logger.info("🎉 所有测试通过!")
        logger.info("推荐系统已成功运行，可以开始使用!")
    else:
        logger.error("❌ 部分测试失败，请检查错误信息")
    
    logger.info("测试完成!")

if __name__ == "__main__":
    main() 