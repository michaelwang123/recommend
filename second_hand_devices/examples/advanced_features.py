#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手设备推荐系统 - 高级功能示例

这个示例展示了推荐系统的高级功能：
1. 多目标推荐优化
2. 实时推荐更新
3. 个性化价格推荐
4. 社交推荐功能
5. 趋势分析和预测
6. A/B 测试框架
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from second_hand_device_recommender import SecondHandDeviceRecommender, SecondHandRecommendationSystem

class MultiObjectiveRecommender(nn.Module):
    """多目标推荐器 - 同时优化相关性、多样性和新颖性"""
    
    def __init__(self, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(1000, embedding_dim)
        self.item_embedding = nn.Embedding(1000, embedding_dim)
        
        # 多目标预测头
        self.relevance_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.diversity_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.novelty_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接特征
        combined = torch.cat([user_emb, item_emb], dim=1)
        
        # 多目标预测
        relevance = self.relevance_head(combined)
        diversity = self.diversity_head(combined)
        novelty = self.novelty_head(combined)
        
        return relevance, diversity, novelty

class RealTimeRecommender:
    """实时推荐系统 - 支持流式数据更新"""
    
    def __init__(self, base_recommender):
        self.base_recommender = base_recommender
        self.recent_interactions = []
        self.user_profiles = {}
        self.item_profiles = {}
        self.update_threshold = 10  # 累积多少次交互后更新模型
        
    def add_interaction(self, user_id, item_id, interaction_type, rating=None):
        """添加新的交互数据"""
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'timestamp': datetime.now()
        }
        
        self.recent_interactions.append(interaction)
        
        # 更新用户画像
        self._update_user_profile(user_id, item_id, interaction_type, rating)
        
        # 检查是否需要更新推荐模型
        if len(self.recent_interactions) >= self.update_threshold:
            self._update_model()
    
    def _update_user_profile(self, user_id, item_id, interaction_type, rating):
        """更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'preferences': {},
                'recent_items': [],
                'interaction_count': 0
            }
        
        profile = self.user_profiles[user_id]
        profile['recent_items'].append(item_id)
        profile['interaction_count'] += 1
        
        # 只保留最近50个交互
        if len(profile['recent_items']) > 50:
            profile['recent_items'] = profile['recent_items'][-50:]
    
    def _update_model(self):
        """更新推荐模型"""
        print(f"📈 更新推荐模型，处理了 {len(self.recent_interactions)} 个新交互")
        
        # 这里可以实现增量学习逻辑
        # 为了简化，我们只是清空缓存
        self.recent_interactions = []
    
    def get_real_time_recommendations(self, user_id, k=10):
        """获取实时推荐"""
        base_recs = self.base_recommender.recommend_for_user(user_id, k=k*2)
        
        # 基于用户最近行为调整推荐
        if user_id in self.user_profiles:
            recent_items = self.user_profiles[user_id]['recent_items']
            # 过滤掉最近已经交互的物品
            filtered_recs = [rec for rec in base_recs if rec['device_id'] not in recent_items]
            return filtered_recs[:k]
        
        return base_recs[:k]

class PersonalizedPriceRecommender:
    """个性化价格推荐器 - 基于用户画像推荐价格"""
    
    def __init__(self):
        self.price_sensitivity_model = None
        self.user_price_profiles = {}
    
    def analyze_user_price_behavior(self, user_id, interactions):
        """分析用户价格行为"""
        user_interactions = interactions[interactions['user_id'] == user_id]
        
        if len(user_interactions) > 0:
            # 计算用户的价格偏好
            purchased_items = user_interactions[user_interactions['interaction_type'] == 'purchase']
            
            if len(purchased_items) > 0:
                avg_price = purchased_items['price'].mean()
                price_std = purchased_items['price'].std()
                max_price = purchased_items['price'].max()
                min_price = purchased_items['price'].min()
                
                # 价格敏感度分析
                price_sensitivity = self._calculate_price_sensitivity(user_interactions)
                
                self.user_price_profiles[user_id] = {
                    'avg_price': avg_price,
                    'price_std': price_std,
                    'max_price': max_price,
                    'min_price': min_price,
                    'price_sensitivity': price_sensitivity,
                    'purchase_frequency': len(purchased_items)
                }
    
    def _calculate_price_sensitivity(self, interactions):
        """计算价格敏感度"""
        # 简单的价格敏感度计算：高价格物品的购买率
        high_price_threshold = interactions['price'].quantile(0.75)
        high_price_items = interactions[interactions['price'] >= high_price_threshold]
        
        if len(high_price_items) > 0:
            purchase_rate = len(high_price_items[high_price_items['interaction_type'] == 'purchase']) / len(high_price_items)
            return 1 - purchase_rate  # 敏感度与购买率成反比
        
        return 0.5  # 默认中等敏感度
    
    def recommend_personalized_price(self, user_id, item_id, base_price):
        """为用户推荐个性化价格"""
        if user_id not in self.user_price_profiles:
            return base_price
        
        profile = self.user_price_profiles[user_id]
        
        # 基于用户价格偏好调整
        if profile['price_sensitivity'] > 0.7:  # 高价格敏感
            discount_factor = 0.9
        elif profile['price_sensitivity'] < 0.3:  # 低价格敏感
            discount_factor = 1.1
        else:  # 中等价格敏感
            discount_factor = 1.0
        
        # 考虑用户历史价格区间
        if base_price > profile['max_price']:
            discount_factor *= 0.85  # 超出历史最高价，给予更多折扣
        
        personalized_price = base_price * discount_factor
        
        return {
            'personalized_price': personalized_price,
            'original_price': base_price,
            'discount_factor': discount_factor,
            'user_price_profile': profile
        }

class SocialRecommender:
    """社交推荐器 - 基于社交关系的推荐"""
    
    def __init__(self):
        self.social_graph = {}
        self.influence_scores = {}
    
    def build_social_graph(self, users, interactions):
        """构建社交图谱"""
        # 基于共同购买行为构建社交关系
        user_items = interactions.groupby('user_id')['device_id'].apply(set).to_dict()
        
        for user1 in user_items:
            self.social_graph[user1] = []
            
            for user2 in user_items:
                if user1 != user2:
                    # 计算共同购买的物品数量
                    common_items = len(user_items[user1] & user_items[user2])
                    total_items = len(user_items[user1] | user_items[user2])
                    
                    if total_items > 0:
                        similarity = common_items / total_items
                        if similarity > 0.1:  # 相似度阈值
                            self.social_graph[user1].append({
                                'user_id': user2,
                                'similarity': similarity
                            })
    
    def calculate_influence_scores(self, interactions):
        """计算用户影响力得分"""
        user_stats = interactions.groupby('user_id').agg({
            'device_id': 'count',  # 交互次数
            'rating': 'mean'  # 平均评分
        }).reset_index()
        
        for _, row in user_stats.iterrows():
            user_id = row['user_id']
            interaction_count = row['device_id']
            avg_rating = row['rating'] if not pd.isna(row['rating']) else 3.0
            
            # 影响力 = 交互次数 * 平均评分
            influence = interaction_count * avg_rating
            self.influence_scores[user_id] = influence
    
    def recommend_by_social_influence(self, user_id, interactions, k=10):
        """基于社交影响力推荐"""
        if user_id not in self.social_graph:
            return []
        
        # 获取社交网络中的推荐
        social_recommendations = {}
        
        for friend in self.social_graph[user_id]:
            friend_id = friend['user_id']
            friend_similarity = friend['similarity']
            friend_influence = self.influence_scores.get(friend_id, 1.0)
            
            # 获取朋友喜欢的物品
            friend_items = interactions[
                (interactions['user_id'] == friend_id) & 
                (interactions['interaction_type'].isin(['like', 'purchase']))
            ]['device_id'].values
            
            for item_id in friend_items:
                if item_id not in social_recommendations:
                    social_recommendations[item_id] = 0
                
                # 推荐分数 = 朋友相似度 * 朋友影响力
                social_recommendations[item_id] += friend_similarity * friend_influence
        
        # 排序并返回top-k
        sorted_recs = sorted(social_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        return [{'device_id': item_id, 'social_score': score} for item_id, score in sorted_recs[:k]]

class TrendAnalyzer:
    """趋势分析器 - 分析和预测市场趋势"""
    
    def __init__(self):
        self.trend_data = {}
        self.seasonal_patterns = {}
    
    def analyze_category_trends(self, interactions, devices, time_window_days=30):
        """分析品类趋势"""
        # 合并设备信息
        interaction_with_devices = interactions.merge(devices, on='device_id')
        
        # 按时间窗口分析趋势
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window_days)
        
        # 过滤时间范围
        recent_interactions = interaction_with_devices[
            interaction_with_devices['timestamp'] >= start_date
        ]
        
        # 分析品类趋势
        category_trends = recent_interactions.groupby(['category', 'interaction_type']).size().reset_index()
        category_trends.columns = ['category', 'interaction_type', 'count']
        
        # 计算增长率
        for category in category_trends['category'].unique():
            category_data = category_trends[category_trends['category'] == category]
            
            # 简单的趋势计算（实际应用中应该使用更复杂的时间序列分析）
            total_interactions = category_data['count'].sum()
            purchase_rate = category_data[category_data['interaction_type'] == 'purchase']['count'].sum() / total_interactions
            
            self.trend_data[category] = {
                'total_interactions': total_interactions,
                'purchase_rate': purchase_rate,
                'trend_score': total_interactions * purchase_rate
            }
    
    def predict_hot_categories(self, k=5):
        """预测热门品类"""
        if not self.trend_data:
            return []
        
        sorted_trends = sorted(self.trend_data.items(), key=lambda x: x[1]['trend_score'], reverse=True)
        
        return [{'category': category, 'trend_score': data['trend_score']} 
                for category, data in sorted_trends[:k]]
    
    def analyze_seasonal_patterns(self, interactions, devices):
        """分析季节性模式"""
        interaction_with_devices = interactions.merge(devices, on='device_id')
        
        # 按月份分组
        interaction_with_devices['month'] = pd.to_datetime(interaction_with_devices['timestamp']).dt.month
        
        seasonal_data = interaction_with_devices.groupby(['month', 'category']).size().reset_index()
        seasonal_data.columns = ['month', 'category', 'count']
        
        # 计算季节性指数
        for category in seasonal_data['category'].unique():
            category_data = seasonal_data[seasonal_data['category'] == category]
            
            if len(category_data) > 0:
                avg_count = category_data['count'].mean()
                seasonal_index = category_data['count'] / avg_count
                
                self.seasonal_patterns[category] = {
                    'monthly_data': category_data[['month', 'count']].to_dict('records'),
                    'seasonal_index': seasonal_index.to_list(),
                    'peak_month': category_data.loc[category_data['count'].idxmax(), 'month']
                }

class ABTestFramework:
    """A/B测试框架 - 用于推荐算法的A/B测试"""
    
    def __init__(self):
        self.test_groups = {}
        self.test_results = {}
    
    def create_test(self, test_name, algorithm_a, algorithm_b, traffic_split=0.5):
        """创建A/B测试"""
        self.test_groups[test_name] = {
            'algorithm_a': algorithm_a,
            'algorithm_b': algorithm_b,
            'traffic_split': traffic_split,
            'results_a': [],
            'results_b': []
        }
    
    def assign_user_to_group(self, test_name, user_id):
        """将用户分配到测试组"""
        if test_name not in self.test_groups:
            return None
        
        # 使用用户ID的哈希值来确定分组（确保一致性）
        user_hash = hash(str(user_id)) % 100
        threshold = self.test_groups[test_name]['traffic_split'] * 100
        
        return 'A' if user_hash < threshold else 'B'
    
    def get_recommendation(self, test_name, user_id, k=10):
        """获取A/B测试的推荐结果"""
        if test_name not in self.test_groups:
            return None
        
        group = self.assign_user_to_group(test_name, user_id)
        test_config = self.test_groups[test_name]
        
        if group == 'A':
            return test_config['algorithm_a'].recommend_for_user(user_id, k=k)
        else:
            return test_config['algorithm_b'].recommend_for_user(user_id, k=k)
    
    def record_result(self, test_name, user_id, metric_name, metric_value):
        """记录测试结果"""
        if test_name not in self.test_groups:
            return
        
        group = self.assign_user_to_group(test_name, user_id)
        test_config = self.test_groups[test_name]
        
        result = {
            'user_id': user_id,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'timestamp': datetime.now()
        }
        
        if group == 'A':
            test_config['results_a'].append(result)
        else:
            test_config['results_b'].append(result)
    
    def analyze_test_results(self, test_name):
        """分析A/B测试结果"""
        if test_name not in self.test_groups:
            return None
        
        test_config = self.test_groups[test_name]
        results_a = test_config['results_a']
        results_b = test_config['results_b']
        
        if not results_a or not results_b:
            return None
        
        # 计算平均指标
        metrics_a = {}
        metrics_b = {}
        
        for result in results_a:
            metric_name = result['metric_name']
            if metric_name not in metrics_a:
                metrics_a[metric_name] = []
            metrics_a[metric_name].append(result['metric_value'])
        
        for result in results_b:
            metric_name = result['metric_name']
            if metric_name not in metrics_b:
                metrics_b[metric_name] = []
            metrics_b[metric_name].append(result['metric_value'])
        
        # 计算统计显著性
        analysis = {}
        for metric_name in metrics_a.keys():
            if metric_name in metrics_b:
                avg_a = np.mean(metrics_a[metric_name])
                avg_b = np.mean(metrics_b[metric_name])
                
                # 简单的统计显著性检验（实际应用中应该使用更严格的统计检验）
                improvement = (avg_b - avg_a) / avg_a * 100
                
                analysis[metric_name] = {
                    'avg_a': avg_a,
                    'avg_b': avg_b,
                    'improvement': improvement,
                    'sample_size_a': len(metrics_a[metric_name]),
                    'sample_size_b': len(metrics_b[metric_name])
                }
        
        return analysis

def advanced_features_demo():
    """高级功能演示"""
    print("🚀 高级功能演示开始...")
    
    # 生成示例数据
    users = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 65, 100),
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], 100)
    })
    
    devices = pd.DataFrame({
        'device_id': range(1, 501),
        'brand': np.random.choice(['苹果', '华为', '小米', '三星'], 500),
        'category': np.random.choice(['手机', '笔记本', '平板'], 500),
        'price': np.random.randint(1000, 10000, 500)
    })
    
    interactions = pd.DataFrame({
        'user_id': np.random.randint(1, 101, 2000),
        'device_id': np.random.randint(1, 501, 2000),
        'interaction_type': np.random.choice(['view', 'like', 'purchase'], 2000),
        'rating': np.random.randint(1, 6, 2000),
        'timestamp': pd.date_range('2023-01-01', periods=2000, freq='H')
    })
    
    # 合并价格信息
    interactions = interactions.merge(devices[['device_id', 'price']], on='device_id')
    
    # 1. 实时推荐演示
    print("\n📡 实时推荐系统演示...")
    base_recommender = SecondHandRecommendationSystem()
    base_recommender.train(users, devices, interactions)
    
    real_time_recommender = RealTimeRecommender(base_recommender)
    
    # 模拟实时交互
    for i in range(5):
        user_id = np.random.randint(1, 101)
        item_id = np.random.randint(1, 501)
        real_time_recommender.add_interaction(user_id, item_id, 'view')
    
    # 获取实时推荐
    real_time_recs = real_time_recommender.get_real_time_recommendations(1, k=5)
    print(f"用户1的实时推荐: {len(real_time_recs)} 个结果")
    
    # 2. 个性化价格推荐演示
    print("\n💰 个性化价格推荐演示...")
    price_recommender = PersonalizedPriceRecommender()
    
    # 分析用户价格行为
    for user_id in range(1, 11):
        price_recommender.analyze_user_price_behavior(user_id, interactions)
    
    # 获取个性化价格推荐
    personalized_price = price_recommender.recommend_personalized_price(1, 1, 5000)
    if personalized_price:
        print(f"用户1的个性化价格: ¥{personalized_price['personalized_price']:.0f}")
        print(f"原价: ¥{personalized_price['original_price']:.0f}")
        print(f"折扣系数: {personalized_price['discount_factor']:.2f}")
    
    # 3. 社交推荐演示
    print("\n👥 社交推荐演示...")
    social_recommender = SocialRecommender()
    social_recommender.build_social_graph(users, interactions)
    social_recommender.calculate_influence_scores(interactions)
    
    social_recs = social_recommender.recommend_by_social_influence(1, interactions, k=5)
    print(f"用户1的社交推荐: {len(social_recs)} 个结果")
    
    # 4. 趋势分析演示
    print("\n📈 趋势分析演示...")
    trend_analyzer = TrendAnalyzer()
    trend_analyzer.analyze_category_trends(interactions, devices)
    
    hot_categories = trend_analyzer.predict_hot_categories(k=3)
    print("热门品类预测:")
    for category_info in hot_categories:
        print(f"  {category_info['category']}: 趋势分数 {category_info['trend_score']:.2f}")
    
    # 5. A/B测试演示
    print("\n🧪 A/B测试演示...")
    ab_test = ABTestFramework()
    
    # 创建测试（使用两个不同的推荐算法）
    recommender_a = SecondHandRecommendationSystem()
    recommender_b = SecondHandRecommendationSystem()
    
    ab_test.create_test('algorithm_comparison', recommender_a, recommender_b)
    
    # 模拟测试结果
    for user_id in range(1, 21):
        ab_test.record_result('algorithm_comparison', user_id, 'ctr', np.random.uniform(0.1, 0.3))
        ab_test.record_result('algorithm_comparison', user_id, 'conversion', np.random.uniform(0.01, 0.05))
    
    # 分析测试结果
    test_analysis = ab_test.analyze_test_results('algorithm_comparison')
    if test_analysis:
        print("A/B测试分析结果:")
        for metric, result in test_analysis.items():
            print(f"  {metric}: A组 {result['avg_a']:.3f}, B组 {result['avg_b']:.3f}")
            print(f"    提升: {result['improvement']:.2f}%")

def main():
    """主函数"""
    print("🎉 二手设备推荐系统 - 高级功能示例")
    print("=" * 50)
    
    advanced_features_demo()
    
    print("\n✅ 高级功能演示完成！")
    print("\n🔧 这些高级功能可以帮助您：")
    print("  1. 实现更精准的个性化推荐")
    print("  2. 提供实时响应的推荐服务")
    print("  3. 基于社交关系增强推荐效果")
    print("  4. 分析市场趋势指导运营决策")
    print("  5. 通过A/B测试优化推荐算法")

if __name__ == "__main__":
    main() 