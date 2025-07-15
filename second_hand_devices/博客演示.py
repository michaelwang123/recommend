#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统品牌偏好权重机制技术博客演示
=====================================

本脚本演示了技术博客中提到的核心概念和实现细节
"""

import pandas as pd
import numpy as np
from second_hand_device_recommender import SecondHandRecommendationSystem, generate_sample_data
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主演示函数"""
    print("=" * 70)
    print("🎯 深度学习推荐系统中的品牌偏好权重机制演示")
    print("=" * 70)
    
    print("\n📖 本演示基于技术博客：《深度学习推荐系统中的品牌偏好权重机制：原理、实现与优化》")
    print("📁 博客文件：推荐系统品牌偏好权重机制技术博客.md")
    
    # 1. 系统架构演示
    print("\n" + "="*50)
    print("1️⃣ 系统架构演示")
    print("="*50)
    
    print("🏗️ 推荐系统架构组件：")
    print("   • 嵌入层：将用户ID、设备ID、品牌ID转换为64维向量")
    print("   • 深度网络：3层MLP (192→128→64→1)")
    print("   • 品牌偏好增强：基于用户历史行为的规则增强")
    
    # 2. 数据生成演示
    print("\n" + "="*50)
    print("2️⃣ 改进的数据生成策略")
    print("="*50)
    
    users, devices, interactions = generate_sample_data()
    
    print(f"📊 生成的数据统计：")
    print(f"   • 用户数量: {len(users):,}")
    print(f"   • 设备数量: {len(devices):,}")
    print(f"   • 交互数量: {len(interactions):,}")
    print(f"   • 平均每用户交互数: {len(interactions) / len(users):.1f}")
    
    # 3. 品牌偏好权重计算演示
    print("\n" + "="*50)
    print("3️⃣ 品牌偏好权重计算演示")
    print("="*50)
    
    # 选择用户12进行演示
    test_user_id = 12
    user_interactions = interactions[interactions['user_id'] == test_user_id]
    interactions_with_brands = user_interactions.merge(devices[['device_id', 'brand']], on='device_id')
    
    print(f"👤 用户 {test_user_id} 的交互记录：")
    for _, row in interactions_with_brands.iterrows():
        print(f"   • 设备{row['device_id']} | 品牌:{row['brand']} | 评分:{row['rating']}")
    
    # 计算品牌偏好权重
    brand_stats = interactions_with_brands.groupby('brand').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    brand_stats.columns = ['brand', 'interaction_count', 'avg_rating']
    
    print(f"\n🧮 品牌偏好权重计算：")
    print(f"   公式：品牌偏好权重 = (交互次数 × 平均评分) ÷ 5.0")
    
    for _, row in brand_stats.iterrows():
        brand = row['brand']
        count = row['interaction_count']
        avg_rating = row['avg_rating']
        weight = count * avg_rating / 5.0
        print(f"   • {brand}: {count}次 × {avg_rating:.2f}分 ÷ 5.0 = {weight:.3f}")
    
    # 4. 模型训练演示
    print("\n" + "="*50)
    print("4️⃣ 深度学习模型训练")
    print("="*50)
    
    recommender = SecondHandRecommendationSystem()
    print("🚀 开始训练推荐模型...")
    recommender.train(users, devices, interactions, epochs=10)
    print("✅ 模型训练完成")
    
    # 5. 权重效果对比演示
    print("\n" + "="*50)
    print("5️⃣ 权重效果对比演示")
    print("="*50)
    
    print(f"🎚️ 不同权重设置的推荐结果对比（用户{test_user_id}）：")
    
    weight_settings = [
        (0.0, "完全依赖模型学习"),
        (0.01, "微量品牌偏好增强（1%）"),
        (0.02, "少量品牌偏好增强（2%）"),
        (0.05, "中等品牌偏好增强（5%）")
    ]
    
    for weight, description in weight_settings:
        print(f"\n📊 {description} (brand_weight={weight})")
        print("-" * 40)
        
        recommendations = recommender.recommend_top_n_devices_for_user(
            user_id=test_user_id,
            device_data=devices,
            interaction_data=interactions,
            top_n=3,
            exclude_interacted=True,
            brand_weight=weight
        )
        
        if recommendations:
            brands = [rec['device_info']['brand'] for rec in recommendations]
            brand_counts = pd.Series(brands).value_counts()
            print(f"推荐品牌分布: {dict(brand_counts)}")
            
            for i, rec in enumerate(recommendations, 1):
                device_info = rec['device_info']
                print(f"  {i}. 设备{rec['device_id']} | 品牌:{device_info['brand']} | "
                      f"价格:{device_info['price']:.0f}元 | 评分:{rec['score']:.3f}")
    
    # 6. 核心洞察总结
    print("\n" + "="*50)
    print("6️⃣ 核心洞察总结")
    print("="*50)
    
    print("💡 关键发现：")
    print("   1. 模型学习了全局用户行为模式")
    print("   2. 评分差异决定了需要多大的权重才能改变排序")
    print("   3. 品牌权重的作用是微调，不是颠覆性改变")
    print("   4. 只有当额外分数 > 原始评分差异时，权重才有效")
    
    print("\n🎯 最佳实践建议：")
    print("   • brand_weight=0.0: 高质量训练数据，充分用户交互")
    print("   • brand_weight=0.02: 平衡点，推荐使用")
    print("   • brand_weight=0.05: 数据稀疏，需要明显偏好体现")
    print("   • brand_weight=0.1: 冷启动或个性化要求极高")
    
    print("\n" + "="*70)
    print("🎉 演示完成！")
    print("📚 详细内容请查看：推荐系统品牌偏好权重机制技术博客.md")
    print("🔗 包含完整的Mermaid图表和技术实现细节")
    print("="*70)

if __name__ == "__main__":
    main() 