import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class Item2VecDataset(Dataset):
    """Item2Vec训练数据集"""
    
    def __init__(self, sequences, window_size=5):
        """
        Args:
            sequences: 用户行为序列列表
            window_size: 窗口大小
        """
        self.sequences = sequences
        self.window_size = window_size
        self.data = self._prepare_data()
    
    def _prepare_data(self):
        """准备训练数据"""
        data = []
        for sequence in self.sequences:
            for i, target in enumerate(sequence):
                # 获取上下文窗口
                start = max(0, i - self.window_size)
                end = min(len(sequence), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context = sequence[j]
                        data.append((target, context))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target, context = self.data[idx]
        return torch.LongTensor([target]), torch.LongTensor([context])

class Item2Vec(nn.Module):
    """Item2Vec模型"""
    
    def __init__(self, vocab_size, embedding_dim=100):
        super(Item2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 嵌入层
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        initrange = 0.5 / self.embedding_dim
        self.target_embedding.weight.data.uniform_(-initrange, initrange)
        self.context_embedding.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, target, context):
        """前向传播"""
        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        
        # 计算相似度
        similarity = torch.sum(target_emb * context_emb, dim=2)
        return similarity
    
    def get_item_embedding(self, item_id):
        """获取物品嵌入"""
        return self.target_embedding.weight[item_id]
    
    def compute_similarity(self, item1_id, item2_id):
        """计算两个物品的相似度"""
        emb1 = self.get_item_embedding(item1_id)
        emb2 = self.get_item_embedding(item2_id)
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()

class DeepNeuralRecommender(nn.Module):
    """深度神经网络推荐系统"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[128, 64]):
        super(DeepNeuralRecommender, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 深度神经网络
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, user_ids, item_ids):
        """前向传播"""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接用户和物品嵌入
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # 通过深度神经网络
        output = self.mlp(concat_emb)
        return output.squeeze()
    
    def compute_item_similarity(self):
        """计算物品相似性矩阵"""
        item_embeddings = self.item_embedding.weight
        normalized_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        return similarity_matrix

class AttentionRecommender(nn.Module):
    """基于注意力机制的推荐系统"""
    
    def __init__(self, n_items, embedding_dim=64, n_heads=8):
        super(AttentionRecommender, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        
        # 物品嵌入
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 多头自注意力
        self.multihead_attention = nn.MultiheadAttention(
            embedding_dim, n_heads, batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(embedding_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, item_sequences):
        """
        前向传播
        Args:
            item_sequences: 物品序列 (batch_size, seq_len)
        """
        # 获取嵌入
        embeddings = self.item_embedding(item_sequences)
        
        # 自注意力
        attn_output, _ = self.multihead_attention(
            embeddings, embeddings, embeddings
        )
        
        # 池化
        pooled = torch.mean(attn_output, dim=1)
        
        # 输出
        output = self.output_layer(pooled)
        return output.squeeze()
    
    def get_item_representation(self, item_id):
        """获取物品表示"""
        return self.item_embedding.weight[item_id]

class SimilarityRecommenderSystem:
    """相似性推荐系统整合类"""
    
    def __init__(self):
        self.models = {}
        self.item_encoder = None
        self.user_encoder = None
    
    def fit_item2vec(self, user_sequences, embedding_dim=100, epochs=50):
        """训练Item2Vec模型"""
        print("训练Item2Vec模型...")
        
        # 准备数据
        all_items = set()
        for seq in user_sequences:
            all_items.update(seq)
        
        self.item_encoder = LabelEncoder()
        self.item_encoder.fit(list(all_items))
        
        # 编码序列
        encoded_sequences = []
        for seq in user_sequences:
            encoded_seq = self.item_encoder.transform(seq)
            encoded_sequences.append(encoded_seq)
        
        # 创建数据集
        dataset = Item2VecDataset(encoded_sequences)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # 创建模型
        vocab_size = len(self.item_encoder.classes_)
        model = Item2Vec(vocab_size, embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for target, context in dataloader:
                target = target.squeeze()
                context = context.squeeze()
                
                # 正样本
                pos_score = model(target, context)
                
                # 负样本
                neg_context = torch.randint(0, vocab_size, context.shape)
                neg_score = model(target, neg_context)
                
                # 计算损失
                pos_loss = criterion(pos_score, torch.ones_like(pos_score))
                neg_loss = criterion(neg_score, torch.zeros_like(neg_score))
                loss = pos_loss + neg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
        
        self.models['item2vec'] = model
        print("Item2Vec训练完成!")
    
    def recommend_similar_items(self, item_id, top_k=5, method='item2vec'):
        """推荐相似物品"""
        if method == 'item2vec' and 'item2vec' in self.models:
            model = self.models['item2vec']
            
            # 编码物品ID
            try:
                encoded_item = self.item_encoder.transform([item_id])[0]
            except:
                print(f"物品 {item_id} 不在训练集中")
                return [], []
            
            # 计算与所有物品的相似度
            similarities = []
            for i in range(len(self.item_encoder.classes_)):
                if i != encoded_item:
                    sim = model.compute_similarity(encoded_item, i)
                    similarities.append((i, sim))
            
            # 排序并返回top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_items = similarities[:top_k]
            
            # 解码物品ID
            item_ids = [self.item_encoder.inverse_transform([item[0]])[0] for item in top_items]
            scores = [item[1] for item in top_items]
            
            return item_ids, scores
        else:
            print(f"方法 {method} 不可用")
            return [], []

def generate_user_sequences(n_users=1000, n_items=100, avg_seq_len=20):
    """生成用户行为序列"""
    sequences = []
    
    for _ in range(n_users):
        seq_len = np.random.poisson(avg_seq_len)
        seq_len = max(5, min(seq_len, 50))  # 限制序列长度
        
        # 生成随机序列，但引入一些相关性
        sequence = []
        for _ in range(seq_len):
            if len(sequence) == 0:
                item = np.random.randint(0, n_items)
            else:
                # 50%概率选择相关物品，50%概率随机选择
                if np.random.random() < 0.5 and len(sequence) > 0:
                    # 选择相关物品（附近的物品ID）
                    last_item = sequence[-1]
                    item = max(0, min(n_items-1, last_item + np.random.randint(-10, 11)))
                else:
                    item = np.random.randint(0, n_items)
            
            sequence.append(item)
        
        sequences.append(sequence)
    
    return sequences

def main():
    """主函数"""
    print("=== 高级相似性推荐系统示例 ===")
    
    # 生成用户行为序列
    user_sequences = generate_user_sequences(n_users=1000, n_items=100)
    print(f"生成了 {len(user_sequences)} 个用户序列")
    print(f"示例序列: {user_sequences[0][:10]}...")
    
    # 创建推荐系统
    recommender = SimilarityRecommenderSystem()
    
    # 训练Item2Vec模型
    recommender.fit_item2vec(user_sequences, embedding_dim=64, epochs=30)
    
    # 进行推荐
    test_item = user_sequences[0][0]  # 选择第一个用户的第一个物品
    print(f"\n为物品 {test_item} 推荐相似物品:")
    
    similar_items, scores = recommender.recommend_similar_items(
        test_item, top_k=5, method='item2vec'
    )
    
    print("推荐结果:")
    for i, (item, score) in enumerate(zip(similar_items, scores)):
        print(f"  {i+1}. 物品 {item}, 相似度: {score:.4f}")
    
    # 可视化嵌入（降维到2D）
    if 'item2vec' in recommender.models:
        model = recommender.models['item2vec']
        embeddings = model.target_embedding.weight.detach().numpy()
        
        # 使用PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 绘制嵌入可视化
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
        plt.title('Item2Vec 嵌入可视化 (PCA降维)')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.grid(True)
        plt.savefig('item2vec_embeddings.png')
        plt.show()
    
    print("\n高级推荐系统演示完成!")

if __name__ == "__main__":
    main() 