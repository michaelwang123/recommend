#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL数据生成器
在MySQL中创建表并生成50000条不连续的用户ID和物品ID数据
保持与generate_user_behavior_data函数相同的用户分布
"""

import mysql.connector
import numpy as np
import pandas as pd
from datetime import datetime
import random

class MySQLDataGenerator:
    def __init__(self):
        # 数据库配置
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'test',
            'password': 'test',
            'database': 'testdb'
        }
        
        # 设置随机种子以获得可重复的结果
        np.random.seed(42)
        random.seed(42)
        
        self.connection = None
        self.cursor = None
    
    def connect_to_database(self):
        """连接到MySQL数据库"""
        try:
            print("🔌 连接到MySQL数据库...")
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("✅ 数据库连接成功")
            return True
        except mysql.connector.Error as err:
            print(f"❌ 数据库连接失败: {err}")
            return False
    
    def create_table(self):
        """创建用户行为数据表"""
        print("🏗️ 创建数据表...")
        
        # 删除已存在的表
        drop_table_sql = "DROP TABLE IF EXISTS user_behavior"
        self.cursor.execute(drop_table_sql)
        
        # 创建新表
        create_table_sql = """
        CREATE TABLE user_behavior (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(20) NOT NULL,
            item_id VARCHAR(20) NOT NULL,
            rating DECIMAL(3,2) NOT NULL,
            user_type VARCHAR(20) NOT NULL,
            item_category VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user_id (user_id),
            INDEX idx_item_id (item_id),
            INDEX idx_user_type (user_type),
            INDEX idx_item_category (item_category)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        self.cursor.execute(create_table_sql)
        self.connection.commit()
        print("✅ 数据表创建成功")
    
    def generate_non_continuous_ids(self):
        """生成不连续的用户ID和物品ID"""
        print("🎲 生成不连续的ID映射...")
        
        # 生成科技爱好者的用户ID (对应原始0-299)
        tech_user_ids = []
        for i in range(300):
            # 生成不连续的ID，如: 10001, 10234, 10567, ...
            user_id = f"U{10000 + i * 7 + random.randint(1, 50)}"
            tech_user_ids.append(user_id)
        
        # 生成时尚爱好者的用户ID (对应原始300-599)
        fashion_user_ids = []
        for i in range(300):
            user_id = f"U{20000 + i * 11 + random.randint(1, 80)}"
            fashion_user_ids.append(user_id)
        
        # 生成运动爱好者的用户ID (对应原始600-999)
        sport_user_ids = []
        for i in range(400):
            user_id = f"U{30000 + i * 13 + random.randint(1, 100)}"
            sport_user_ids.append(user_id)
        
        # 生成科技产品ID (对应原始0-199)
        tech_item_ids = []
        for i in range(200):
            item_id = f"TECH{1000 + i * 5 + random.randint(1, 30)}"
            tech_item_ids.append(item_id)
        
        # 生成时尚产品ID (对应原始200-399)
        fashion_item_ids = []
        for i in range(200):
            item_id = f"FASH{2000 + i * 7 + random.randint(1, 40)}"
            fashion_item_ids.append(item_id)
        
        # 生成运动产品ID (对应原始300-499)
        sport_item_ids = []
        for i in range(200):
            item_id = f"SPRT{3000 + i * 9 + random.randint(1, 50)}"
            sport_item_ids.append(item_id)
        
        print(f"✅ ID生成完成:")
        print(f"   科技用户ID数量: {len(tech_user_ids)}")
        print(f"   时尚用户ID数量: {len(fashion_user_ids)}")
        print(f"   运动用户ID数量: {len(sport_user_ids)}")
        print(f"   科技产品ID数量: {len(tech_item_ids)}")
        print(f"   时尚产品ID数量: {len(fashion_item_ids)}")
        print(f"   运动产品ID数量: {len(sport_item_ids)}")
        
        return {
            'tech_users': tech_user_ids,
            'fashion_users': fashion_user_ids,
            'sport_users': sport_user_ids,
            'tech_items': tech_item_ids,
            'fashion_items': fashion_item_ids,
            'sport_items': sport_item_ids
        }
    
    def generate_behavior_data(self, id_mappings):
        """生成用户行为数据（保持原始分布）"""
        print("📊 生成用户行为数据...")
        
        interactions = []
        
        # 科技爱好者 - 20000条记录
        print("   生成科技爱好者数据...")
        for _ in range(20000):
            user_id = np.random.choice(id_mappings['tech_users'])
            item_id = np.random.choice(id_mappings['tech_items'])
            rating = np.random.normal(4.0, 0.8)
            rating = np.clip(rating, 1, 5)
            rating = round(rating, 2)
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'user_type': 'tech',
                'item_category': 'technology'
            })
        
        # 时尚爱好者 - 20000条记录
        print("   生成时尚爱好者数据...")
        for _ in range(20000):
            user_id = np.random.choice(id_mappings['fashion_users'])
            item_id = np.random.choice(id_mappings['fashion_items'])
            rating = np.random.normal(4.2, 0.7)
            rating = np.clip(rating, 1, 5)
            rating = round(rating, 2)
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'user_type': 'fashion',
                'item_category': 'fashion'
            })
        
        # 运动爱好者 - 10000条记录
        print("   生成运动爱好者数据...")
        for _ in range(10000):
            user_id = np.random.choice(id_mappings['sport_users'])
            item_id = np.random.choice(id_mappings['sport_items'])
            rating = np.random.normal(3.8, 0.9)
            rating = np.clip(rating, 1, 5)
            rating = round(rating, 2)
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'user_type': 'sport',
                'item_category': 'sports'
            })
        
        # 打乱数据顺序
        random.shuffle(interactions)
        
        print(f"✅ 生成了 {len(interactions)} 条用户行为记录")
        
        return interactions
    
    def insert_data_to_mysql(self, interactions):
        """将数据插入到MySQL"""
        print("💾 插入数据到MySQL...")
        
        insert_sql = """
        INSERT INTO user_behavior (user_id, item_id, rating, user_type, item_category)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        # 批量插入数据
        batch_size = 1000
        total_batches = len(interactions) // batch_size + 1
        
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            batch_data = [
                (item['user_id'], item['item_id'], item['rating'], 
                 item['user_type'], item['item_category'])
                for item in batch
            ]
            
            self.cursor.executemany(insert_sql, batch_data)
            self.connection.commit()
            
            current_batch = i // batch_size + 1
            print(f"   已插入批次 {current_batch}/{total_batches}")
        
        print("✅ 数据插入完成")
    
    def verify_data(self):
        """验证插入的数据"""
        print("🔍 验证插入的数据...")
        
        # 统计总记录数
        self.cursor.execute("SELECT COUNT(*) FROM user_behavior")
        total_count = self.cursor.fetchone()[0]
        print(f"   总记录数: {total_count}")
        
        # 统计不同用户类型的记录数
        self.cursor.execute("""
            SELECT user_type, COUNT(*) as count, AVG(rating) as avg_rating
            FROM user_behavior 
            GROUP BY user_type
        """)
        user_type_stats = self.cursor.fetchall()
        
        print("   用户类型统计:")
        for user_type, count, avg_rating in user_type_stats:
            print(f"     {user_type}: {count} 条记录, 平均评分: {avg_rating:.2f}")
        
        # 统计唯一用户数和物品数
        self.cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_behavior")
        unique_users = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(DISTINCT item_id) FROM user_behavior")
        unique_items = self.cursor.fetchone()[0]
        
        print(f"   唯一用户数: {unique_users}")
        print(f"   唯一物品数: {unique_items}")
        
        # 显示一些示例数据
        self.cursor.execute("""
            SELECT user_id, item_id, rating, user_type, item_category 
            FROM user_behavior 
            LIMIT 10
        """)
        sample_data = self.cursor.fetchall()
        
        print("   示例数据:")
        for row in sample_data:
            print(f"     用户: {row[0]}, 物品: {row[1]}, 评分: {row[2]}, "
                  f"用户类型: {row[3]}, 物品类别: {row[4]}")
    
    def export_to_csv(self, filename="user_behavior_data.csv"):
        """导出数据到CSV文件"""
        print(f"📤 导出数据到 {filename}...")
        
        self.cursor.execute("""
            SELECT user_id, item_id, rating, user_type, item_category, created_at
            FROM user_behavior
            ORDER BY id
        """)
        
        data = self.cursor.fetchall()
        
        df = pd.DataFrame(data, columns=[
            'user_id', 'item_id', 'rating', 'user_type', 'item_category', 'created_at'
        ])
        
        df.to_csv(filename, index=False)
        print(f"✅ 数据已导出到 {filename}")
        
        return df
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("🔌 数据库连接已关闭")
    
    def run(self):
        """运行完整的数据生成流程"""
        print("🚀 开始MySQL数据生成流程")
        print("=" * 80)
        
        try:
            # 1. 连接数据库
            if not self.connect_to_database():
                return False
            
            # 2. 创建表
            self.create_table()
            
            # 3. 生成不连续的ID
            id_mappings = self.generate_non_continuous_ids()
            
            # 4. 生成行为数据
            interactions = self.generate_behavior_data(id_mappings)
            
            # 5. 插入数据到MySQL
            self.insert_data_to_mysql(interactions)
            
            # 6. 验证数据
            self.verify_data()
            
            # 7. 导出数据到CSV
            df = self.export_to_csv()
            
            print("\n" + "=" * 80)
            print("🎯 数据生成完成!")
            print("• 已在MySQL中创建user_behavior表")
            print("• 生成了50000条不连续ID的用户行为数据")
            print("• 保持了与原始函数相同的用户分布")
            print("• 数据已导出到CSV文件")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"❌ 数据生成过程中出现错误: {e}")
            return False
        
        finally:
            self.close_connection()

def main():
    """主函数"""
    generator = MySQLDataGenerator()
    success = generator.run()
    
    if success:
        print("\n✅ 所有操作完成!")
        print("您现在可以使用这些数据来验证推荐系统模型。")
    else:
        print("\n❌ 操作失败，请检查错误信息。")

if __name__ == "__main__":
    main() 