#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID嵌入向量获取机制详解
详细解释嵌入向量是如何从ID获取的，包括内部实现和数学原理
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.nn import functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EmbeddingMechanismExplainer:
    def __init__(self):
        self.n_users = 10
        self.embedding_dim = 4  # 为了演示使用小维度
        self.n_devices = 8
        
    def explain_lookup_table_concept(self):
        """解释查找表概念"""
        print("📋 嵌入层的本质：查找表")
        print("=" * 80)
        
        print("🔍 核心概念：嵌入层实际上是一个可学习的查找表")
        print()
        
        # 创建一个简单的嵌入层用于演示
        user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        
        print("嵌入层内部结构：")
        print("-" * 40)
        print("```")
        print(f"用户数量: {self.n_users}")
        print(f"嵌入维度: {self.embedding_dim}")
        print(f"嵌入矩阵形状: [{self.n_users}, {self.embedding_dim}]")
        print("```")
        
        # 显示嵌入矩阵
        embedding_matrix = user_embedding.weight.data
        print(f"\n嵌入矩阵内容（实际参数）:")
        print("-" * 40)
        
        print("用户ID | 嵌入向量 (4维)")
        print("-------|" + "-" * 40)
        for i in range(self.n_users):
            vector = embedding_matrix[i].numpy()
            vector_str = f"[{vector[0]:6.3f}, {vector[1]:6.3f}, {vector[2]:6.3f}, {vector[3]:6.3f}]"
            print(f"用户{i:2d}  | {vector_str}")
        
        print(f"\n💡 关键理解：")
        print("• 嵌入层就是一个矩阵，每行对应一个用户的向量")
        print("• 通过用户ID直接索引矩阵的对应行")
        print("• 这些向量在训练过程中会不断更新")
        
        return user_embedding, embedding_matrix
    
    def demonstrate_indexing_process(self):
        """演示索引过程"""
        print(f"\n🔍 索引过程详解")
        print("=" * 80)
        
        user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        embedding_matrix = user_embedding.weight.data
        
        print("步骤1：准备用户ID输入")
        print("-" * 30)
        user_ids = torch.tensor([0, 2, 5, 7])
        print(f"输入用户ID: {user_ids.tolist()}")
        print(f"Tensor形状: {user_ids.shape}")
        
        print(f"\n步骤2：通过嵌入层获取向量")
        print("-" * 30)
        print("```python")
        print("user_vectors = user_embedding(user_ids)")
        print("```")
        
        user_vectors = user_embedding(user_ids)
        print(f"输出形状: {user_vectors.shape}")
        
        print(f"\n步骤3：查看具体的索引过程")
        print("-" * 30)
        
        for i, user_id in enumerate(user_ids):
            manual_vector = embedding_matrix[user_id]
            auto_vector = user_vectors[i]
            
            print(f"用户ID {user_id}:")
            print(f"  手动索引: {manual_vector.numpy()}")
            print(f"  嵌入层输出: {auto_vector.detach().numpy()}")
            print(f"  是否相同: {torch.allclose(manual_vector, auto_vector)}")
            print()
        
        print("🎯 索引原理：")
        print("user_embedding(user_id) ≈ embedding_matrix[user_id]")
        print("本质上就是矩阵的行索引操作")
        
        return user_ids, user_vectors
    
    def show_mathematical_details(self):
        """展示数学细节"""
        print(f"\n📐 数学原理详解")
        print("=" * 80)
        
        print("🔢 One-Hot编码视角：")
        print("-" * 30)
        
        user_id = 3
        one_hot = torch.zeros(self.n_users)
        one_hot[user_id] = 1
        
        print(f"用户ID: {user_id}")
        print(f"One-Hot向量: {one_hot.numpy()}")
        
        # 创建嵌入矩阵
        user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        W = user_embedding.weight.data
        
        print(f"\n嵌入矩阵 W 形状: {W.shape}")
        print("W =")
        for i in range(W.shape[0]):
            print(f"  [{W[i, 0].item():6.3f}, {W[i, 1].item():6.3f}, {W[i, 2].item():6.3f}, {W[i, 3].item():6.3f}]")
        
        print(f"\n🧮 矩阵乘法计算：")
        print("-" * 30)
        
        # 方法1：One-hot矩阵乘法
        result_matmul = torch.matmul(one_hot, W)
        
        # 方法2：直接索引
        result_index = W[user_id]
        
        # 方法3：嵌入层
        result_embedding = user_embedding(torch.tensor([user_id]))[0]
        
        print(f"方法1 (One-Hot × W): {result_matmul.numpy()}")
        print(f"方法2 (直接索引):     {result_index.numpy()}")
        print(f"方法3 (嵌入层):      {result_embedding.detach().numpy()}")
        
        print(f"\n数学公式：")
        print("user_vector = one_hot_vector × W")
        print("其中：")
        print(f"  one_hot_vector.shape = [1, {self.n_users}]")
        print(f"  W.shape = [{self.n_users}, {self.embedding_dim}]")
        print(f"  user_vector.shape = [1, {self.embedding_dim}]")
        
        print(f"\n⚡ 优化：")
        print("实际实现中，PyTorch跳过One-Hot编码，直接使用索引")
        print("这样更高效，避免了稀疏矩阵乘法")
        
        return W, one_hot, result_matmul
    
    def demonstrate_gradient_flow(self):
        """演示梯度流动"""
        print(f"\n🔄 梯度更新机制")
        print("=" * 80)
        
        print("🎯 训练过程中嵌入向量如何更新：")
        print("-" * 40)
        
        # 创建简单的推荐模型
        class SimpleRecommendModel(nn.Module):
            def __init__(self, n_users, n_devices, embed_dim):
                super().__init__()
                self.user_embedding = nn.Embedding(n_users, embed_dim)
                self.device_embedding = nn.Embedding(n_devices, embed_dim)
                
            def forward(self, user_ids, device_ids):
                user_vectors = self.user_embedding(user_ids)
                device_vectors = self.device_embedding(device_ids)
                
                # 计算相似度得分
                scores = torch.sum(user_vectors * device_vectors, dim=1)
                return scores
        
        model = SimpleRecommendModel(self.n_users, self.n_devices, self.embedding_dim)
        
        print("模型结构:")
        print("```python")
        print("user_vectors = user_embedding(user_ids)")
        print("device_vectors = device_embedding(device_ids)")
        print("scores = sum(user_vectors * device_vectors)")
        print("```")
        
        # 记录初始参数
        initial_user_embedding = model.user_embedding.weight.data.clone()
        
        print(f"\n初始用户0的嵌入向量:")
        print(f"{initial_user_embedding[0].numpy()}")
        
        # 模拟一次训练步骤
        print(f"\n🚀 模拟训练步骤：")
        print("-" * 30)
        
        user_ids = torch.tensor([0, 1, 2])
        device_ids = torch.tensor([0, 1, 2])
        target_scores = torch.tensor([1.0, 0.0, 1.0])  # 真实标签
        
        # 前向传播
        predicted_scores = model(user_ids, device_ids)
        
        # 计算损失
        loss = F.mse_loss(predicted_scores, target_scores)
        
        print(f"输入: 用户{user_ids.tolist()}, 设备{device_ids.tolist()}")
        print(f"预测得分: {predicted_scores.detach().numpy()}")
        print(f"真实得分: {target_scores.numpy()}")
        print(f"损失: {loss.item():.4f}")
        
        # 反向传播
        loss.backward()
        
        print(f"\n📊 梯度信息：")
        print("-" * 20)
        user_grad = model.user_embedding.weight.grad
        print(f"用户0嵌入向量的梯度: {user_grad[0].numpy()}")
        
        # 参数更新
        lr = 0.1
        with torch.no_grad():
            model.user_embedding.weight -= lr * model.user_embedding.weight.grad
        
        updated_user_embedding = model.user_embedding.weight.data
        print(f"\n更新后用户0的嵌入向量:")
        print(f"{updated_user_embedding[0].numpy()}")
        
        change = updated_user_embedding[0] - initial_user_embedding[0]
        print(f"变化量: {change.numpy()}")
        print(f"更新公式: new_embedding = old_embedding - lr * gradient")
        
        return model, initial_user_embedding, updated_user_embedding
    
    def show_batch_processing(self):
        """展示批处理"""
        print(f"\n🔄 批处理机制")
        print("=" * 80)
        
        user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        
        print("🎯 单个用户 vs 批量用户处理：")
        print("-" * 40)
        
        # 单个用户
        single_user_id = torch.tensor([3])
        single_result = user_embedding(single_user_id)
        
        print(f"单个用户处理:")
        print(f"  输入: {single_user_id.tolist()}")
        print(f"  输入形状: {single_user_id.shape}")
        print(f"  输出形状: {single_result.shape}")
        print(f"  输出: {single_result.detach().numpy()}")
        
        # 批量用户
        batch_user_ids = torch.tensor([0, 3, 5, 7])
        batch_result = user_embedding(batch_user_ids)
        
        print(f"\n批量用户处理:")
        print(f"  输入: {batch_user_ids.tolist()}")
        print(f"  输入形状: {batch_user_ids.shape}")
        print(f"  输出形状: {batch_result.shape}")
        print(f"  输出:")
        for i, user_id in enumerate(batch_user_ids):
            print(f"    用户{user_id}: {batch_result[i].detach().numpy()}")
        
        print(f"\n⚡ 批处理优势：")
        print("• GPU并行处理多个用户")
        print("• 提高训练效率")
        print("• 充分利用硬件资源")
        
        return single_result, batch_result
    
    def compare_implementation_methods(self):
        """对比不同实现方法"""
        print(f"\n🔧 不同实现方法对比")
        print("=" * 80)
        
        user_ids = torch.tensor([1, 3, 5])
        
        print("方法对比（功能相同，效率不同）：")
        print("-" * 50)
        
        # 方法1：PyTorch嵌入层（推荐）
        embedding_layer = nn.Embedding(self.n_users, self.embedding_dim)
        result1 = embedding_layer(user_ids)
        
        print("方法1：PyTorch嵌入层")
        print("```python")
        print("embedding = nn.Embedding(num_users, embed_dim)")
        print("result = embedding(user_ids)")
        print("```")
        print(f"优点: 高效、自动梯度、GPU优化")
        print(f"结果形状: {result1.shape}")
        
        # 方法2：手动查找表
        W = embedding_layer.weight.data
        result2 = W[user_ids]
        
        print(f"\n方法2：手动索引")
        print("```python")
        print("W = embedding_matrix")
        print("result = W[user_ids]")
        print("```")
        print(f"优点: 简单直观")
        print(f"缺点: 手动处理梯度")
        print(f"结果形状: {result2.shape}")
        
        # 方法3：One-Hot + 矩阵乘法（不推荐）
        one_hot_batch = torch.zeros(len(user_ids), self.n_users)
        for i, uid in enumerate(user_ids):
            one_hot_batch[i, uid] = 1
        result3 = torch.matmul(one_hot_batch, W)
        
        print(f"\n方法3：One-Hot矩阵乘法")
        print("```python")
        print("one_hot = to_one_hot(user_ids)")
        print("result = one_hot @ embedding_matrix")
        print("```")
        print(f"优点: 数学原理清晰")
        print(f"缺点: 内存消耗大、计算低效")
        print(f"结果形状: {result3.shape}")
        
        # 验证结果一致性
        print(f"\n✅ 结果验证：")
        print(f"方法1 vs 方法2: {torch.allclose(result1, result2)}")
        print(f"方法1 vs 方法3: {torch.allclose(result1, result3)}")
        print(f"方法2 vs 方法3: {torch.allclose(result2, result3)}")
        
        print(f"\n🎯 推荐使用：")
        print("✅ PyTorch nn.Embedding - 最佳选择")
        print("⚠️ 手动索引 - 调试时可用")
        print("❌ One-Hot矩阵乘法 - 避免使用")
        
        return result1, result2, result3
    
    def demonstrate_memory_efficiency(self):
        """演示内存效率"""
        print(f"\n💾 内存效率分析")
        print("=" * 80)
        
        large_n_users = 10000
        large_embed_dim = 64
        
        print("🔍 大规模场景分析：")
        print("-" * 30)
        print(f"用户数量: {large_n_users:,}")
        print(f"嵌入维度: {large_embed_dim}")
        
        # 嵌入矩阵大小
        embedding_params = large_n_users * large_embed_dim
        embedding_memory_mb = embedding_params * 4 / (1024 * 1024)  # float32
        
        print(f"\n嵌入矩阵:")
        print(f"  参数数量: {embedding_params:,}")
        print(f"  内存占用: {embedding_memory_mb:.1f} MB")
        
        # 批处理内存
        batch_size = 512
        batch_memory_kb = batch_size * large_embed_dim * 4 / 1024
        
        print(f"\n批处理 (batch_size={batch_size}):")
        print(f"  输出张量大小: [{batch_size}, {large_embed_dim}]")
        print(f"  内存占用: {batch_memory_kb:.1f} KB")
        
        # One-Hot方法内存（对比）
        onehot_memory_mb = batch_size * large_n_users * 4 / (1024 * 1024)
        
        print(f"\n如果使用One-Hot方法:")
        print(f"  One-Hot矩阵大小: [{batch_size}, {large_n_users}]")
        print(f"  内存占用: {onehot_memory_mb:.1f} MB")
        print(f"  效率比较: One-Hot是嵌入层的 {onehot_memory_mb/batch_memory_kb*1024:.0f}x 内存消耗")
        
        print(f"\n💡 关键优势：")
        print("• 嵌入层只存储必要的参数矩阵")
        print("• 避免了稀疏的One-Hot表示")
        print("• 索引操作比矩阵乘法更高效")
        
        return embedding_memory_mb, batch_memory_kb, onehot_memory_mb
    
    def create_visual_summary(self):
        """创建可视化总结"""
        print(f"\n📊 可视化总结")
        print("=" * 80)
        
        print("ID嵌入向量获取流程图：")
        print("""
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │   用户ID    │───▶│   嵌入层    │───▶│  嵌入向量   │
        │   [0,1,2]   │    │  查找表机制  │    │ [4×64矩阵]  │
        └─────────────┘    └─────────────┘    └─────────────┘
                                 │
                                 ▼
                         ┌─────────────┐
                         │ 嵌入矩阵W   │
                         │[10000×64]   │
                         │可学习参数   │
                         └─────────────┘
        """)
        
        print("关键步骤：")
        print("1️⃣ 输入：用户ID张量 [batch_size]")
        print("2️⃣ 索引：在嵌入矩阵中查找对应行")
        print("3️⃣ 输出：嵌入向量 [batch_size, embed_dim]")
        print("4️⃣ 训练：通过反向传播更新嵌入矩阵")
        
        return True

def main():
    """主函数"""
    print("🔍 ID嵌入向量获取机制详解")
    print("=" * 80)
    
    explainer = EmbeddingMechanismExplainer()
    
    # 1. 查找表概念
    explainer.explain_lookup_table_concept()
    
    # 2. 索引过程演示
    explainer.demonstrate_indexing_process()
    
    # 3. 数学原理
    explainer.show_mathematical_details()
    
    # 4. 梯度更新
    explainer.demonstrate_gradient_flow()
    
    # 5. 批处理机制
    explainer.show_batch_processing()
    
    # 6. 实现方法对比
    explainer.compare_implementation_methods()
    
    # 7. 内存效率
    explainer.demonstrate_memory_efficiency()
    
    # 8. 可视化总结
    explainer.create_visual_summary()
    
    print(f"\n" + "=" * 80)
    print("🎯 核心要点总结:")
    print("• 嵌入层本质是可学习的查找表")
    print("• 通过用户ID直接索引嵌入矩阵的对应行")
    print("• 比One-Hot+矩阵乘法更高效")
    print("• 嵌入向量在训练过程中通过梯度更新")
    print("• PyTorch自动处理批处理和GPU优化")
    print("=" * 80)

if __name__ == "__main__":
    main() 