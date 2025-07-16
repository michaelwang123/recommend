import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

print("=== 用户转换为64维向量详细演示 ===")

# 步骤1：原始用户数据
print("\n1. 原始用户数据:")
users = ["张三", "李四", "王五", "赵六", "钱七"]
print(f"用户列表: {users}")

# 步骤2：用户ID编码
print("\n2. 用户ID编码:")
user_encoder = LabelEncoder()
user_encoded = user_encoder.fit_transform(users)
print(f"编码后的用户ID: {user_encoded}")
print(f"编码映射关系:")
for i, user in enumerate(users):
    print(f"  '{user}' -> {user_encoded[i]}")

# 步骤3：创建嵌入层
print("\n3. 创建嵌入层:")
n_users = len(users)  # 5个用户
embedding_dim = 64    # 64维向量
user_embedding = nn.Embedding(n_users, embedding_dim)
print(f"嵌入层参数: nn.Embedding({n_users}, {embedding_dim})")
print(f"嵌入层权重形状: {user_embedding.weight.shape}")
print(f"总参数数量: {n_users * embedding_dim}")

# 步骤4：转换过程演示
print("\n4. 转换过程演示:")
print(f"用户'张三'的转换过程:")
print(f"  原始用户名: '张三'")
print(f"  编码后ID: {user_encoded[0]}")

# 转换为PyTorch张量
user_id_tensor = torch.LongTensor([user_encoded[0]])
print(f"  PyTorch张量: {user_id_tensor}")

# 通过嵌入层转换
user_vector = user_embedding(user_id_tensor)
print(f"  64维向量形状: {user_vector.shape}")
print(f"  64维向量前10个值: {user_vector[0][:10].detach().numpy()}")

# 步骤5：批量转换
print("\n5. 批量转换演示:")
all_user_ids = torch.LongTensor(user_encoded)
all_user_vectors = user_embedding(all_user_ids)
print(f"批量输入形状: {all_user_ids.shape}")
print(f"批量输出形状: {all_user_vectors.shape}")

# 展示每个用户的向量
for i, user in enumerate(users):
    vector = all_user_vectors[i]
    print(f"用户'{user}': 向量长度={len(vector)}, 前5个值={vector[:5].detach().numpy()}")

# 步骤6：向量相似度计算（随机初始化）
print("\n6. 向量相似度计算（随机初始化）:")
user1_vector = all_user_vectors[0]  # 张三
user2_vector = all_user_vectors[1]  # 李四
similarity = torch.cosine_similarity(user1_vector, user2_vector, dim=0)
print(f"张三和李四的相似度: {similarity.item():.4f}")

print("⚠️  WARNING: 这个相似度是没有意义的！因为向量是随机初始化的")

# 步骤7：问题分析
print("\n7. 问题分析:")
print("❌ 当前问题：")
print("  • 嵌入向量是随机初始化的")
print("  • 没有反映用户真实偏好")
print("  • 相似度计算毫无意义")
print("  • 无法用于实际推荐")

# 步骤8：简单的训练示例
print("\n8. 简单的训练示例:")
print("为了让嵌入向量有意义，我们需要训练数据：")

# 模拟用户行为数据
print("\n模拟用户行为数据:")
# 假设：张三和李四喜欢科技产品，王五和赵六喜欢时尚产品，钱七喜欢运动产品
user_behavior = {
    "张三": [0, 1, 2],  # 喜欢物品0,1,2（科技产品）
    "李四": [0, 1, 3],  # 喜欢物品0,1,3（科技产品）
    "王五": [4, 5, 6],  # 喜欢物品4,5,6（时尚产品）
    "赵六": [4, 5, 7],  # 喜欢物品4,5,7（时尚产品）
    "钱七": [8, 9, 10]  # 喜欢物品8,9,10（运动产品）
}

for user, items in user_behavior.items():
    print(f"  {user}: 喜欢物品 {items}")

# 创建简单的推荐模型
print("\n创建简单推荐模型:")
class SimpleRecommender(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=8):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return torch.sum(user_emb * item_emb, dim=1)

# 准备训练数据
train_data = []
for user, items in user_behavior.items():
    user_id = user_encoder.transform([user])[0]
    for item_id in items:
        train_data.append((user_id, item_id, 1.0))  # 正样本

# 添加负样本（用户没有交互的物品）
for user, items in user_behavior.items():
    user_id = user_encoder.transform([user])[0]
    all_items = set(range(11))  # 物品0-10
    negative_items = all_items - set(items)
    for item_id in list(negative_items)[:2]:  # 只取2个负样本
        train_data.append((user_id, item_id, 0.0))  # 负样本

# 训练模型
print(f"训练数据样本数: {len(train_data)}")
model = SimpleRecommender(n_users=5, n_items=11, embedding_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

print("\n开始训练...")
for epoch in range(200):
    total_loss = 0
    for user_id, item_id, rating in train_data:
        optimizer.zero_grad()
        pred = model(torch.LongTensor([user_id]), torch.LongTensor([item_id]))
        loss = criterion(pred, torch.FloatTensor([rating]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_data):.4f}")

# 步骤9：训练后的用户嵌入向量分析
print("\n9. 训练后的用户嵌入向量分析:")
trained_user_embeddings = model.user_embedding.weight.detach()

print("训练后的用户嵌入向量:")
for i, user in enumerate(users):
    vector = trained_user_embeddings[i]
    print(f"用户'{user}': {vector.numpy()}")

# 计算训练后的用户相似度
print("\n训练后的用户相似度:")
for i, user1 in enumerate(users):
    for j, user2 in enumerate(users):
        if i < j:
            vec1 = trained_user_embeddings[i]
            vec2 = trained_user_embeddings[j]
            similarity = torch.cosine_similarity(vec1, vec2, dim=0)
            print(f"{user1} vs {user2}: {similarity.item():.4f}")

print("\n=== 关键洞察 ===")
print("✅ 训练后的嵌入向量才有意义：")
print("  • 相似偏好的用户嵌入向量更接近")
print("  • 张三和李四（都喜欢科技）相似度更高")
print("  • 王五和赵六（都喜欢时尚）相似度更高")
print("  • 不同类型用户的相似度较低")

print("\n❌ 随机初始化的嵌入向量没有意义：")
print("  • 向量值是随机的，不反映真实偏好")
print("  • 相似度计算结果毫无意义")
print("  • 无法用于实际推荐系统")

print("\n=== 转换过程总结 ===")
print("1. 用户名 → 数字ID（编码）")
print("2. 数字ID → PyTorch张量")
print("3. 张量 → 64维向量（嵌入层）")
print("4. 🔥 关键步骤：通过用户行为数据训练向量")
print("5. 训练后的向量才能用于推荐评分")

print("\n📚 实际推荐系统中的应用：")
print("  • 收集用户行为数据（点击、购买、评分等）")
print("  • 通过协同过滤或深度学习训练模型")
print("  • 学习到的嵌入向量才能反映用户偏好")
print("  • 项目中的 meaningful_user_embedding.py 提供了完整示例") 