#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐系统 API 服务
基于 FastAPI 构建的推荐接口服务
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from recommend import MySQLRecommendationSystem

# 请求和响应模型
class RecommendationRequest(BaseModel):
    user_id: str
    top_n: int = 10
    exclude_rated: bool = True

class RecommendationItem(BaseModel):
    rank: int
    item_id: str
    predicted_rating: float

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    total_count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_users: int
    num_items: int

# 创建 FastAPI 应用
app = FastAPI(
    title="推荐系统 API",
    description="基于 PyTorch 的个性化推荐服务",
    version="1.0.0"
)

# 全局推荐系统实例
recommender = None

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global recommender
    
    try:
        print("🚀 启动推荐系统 API 服务...")
        
        # 初始化推荐系统
        recommender = MySQLRecommendationSystem()
        
        # 加载保存的模型
        model_path = "./saved_model"
        if os.path.exists(model_path):
            success = recommender.load_model_and_mappings(model_path)
            if success:
                print("✅ 模型加载成功!")
            else:
                print("❌ 模型加载失败!")
                recommender = None
        else:
            print(f"⚠️ 模型路径不存在: {model_path}")
            recommender = None
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        recommender = None

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "推荐系统 API 服务",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    global recommender
    
    model_loaded = recommender is not None and recommender.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        num_users=recommender.num_users if model_loaded else 0,
        num_items=recommender.num_items if model_loaded else 0
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """获取用户推荐"""
    global recommender
    
    # 检查模型是否加载
    if recommender is None or recommender.model is None:
        raise HTTPException(
            status_code=503,
            detail="推荐模型未加载，请检查服务状态"
        )
    
    # 检查用户是否存在
    if request.user_id not in recommender.user_id_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"用户 {request.user_id} 不存在于训练数据中"
        )
    
    try:
        # 生成推荐
        recommendations = recommender.get_user_recommendations(
            user_id=request.user_id,
            top_n=request.top_n,
            exclude_rated=request.exclude_rated
        )
        
        # 转换为响应格式
        recommendation_items = [
            RecommendationItem(
                rank=rec['rank'],
                item_id=rec['item_id'],
                predicted_rating=round(rec['predicted_rating'], 4)
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendation_items,
            total_count=len(recommendation_items)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"推荐生成失败: {str(e)}"
        )

@app.get("/users")
async def get_users():
    """获取所有用户列表"""
    global recommender
    
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="推荐模型未加载"
        )
    
    user_ids = list(recommender.user_id_to_idx.keys())
    
    return {
        "total_users": len(user_ids),
        "user_ids": user_ids[:100],  # 限制返回前100个
        "note": "如果用户超过100个，仅显示前100个"
    }

if __name__ == "__main__":
    print("🚀 启动推荐系统 API 服务器...")
    print("📖 API 文档地址: http://localhost:8000/docs")
    print("🔍 健康检查: http://localhost:8000/health")
    
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 