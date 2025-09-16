from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from clip_client import Client
import faiss
import numpy as np
import os
import uuid
from PIL import Image
import io

# 初始化FastAPI应用
app = FastAPI(title="多模态检索API服务")

# 初始化CLIP客户端连接到本地CLIP服务
clip_client = Client("grpc://0.0.0.0:51000")

# 向量维度 - CLIP ViT-B/32模型的输出维度是512
VECTOR_DIM = 512

# 初始化FAISS索引
index = faiss.IndexFlatL2(VECTOR_DIM)  # 使用L2距离

# 存储图片ID到路径的映射
image_metadata = {}

# 1. 将图片数据写入向量库的接口
@app.post("/images/add", summary="添加图片到向量库")
async def add_image_to_vector_db(file: UploadFile = File(...)):
    try:
        # 读取图片文件
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 将图片转换为向量
        image_embedding = clip_client.encode_image([image])

        # 生成唯一ID
        image_id = str(uuid.uuid4())

        # 存储图片元数据
        image_metadata[image_id] = {
            "filename": file.filename,
            "content_type": file.content_type
        }

        # 将向量添加到FAISS索引
        index.add(np.array(image_embedding).astype('float32'))

        return {"status": "success", "image_id": image_id, "message": "图片已成功添加到向量库"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加图片失败: {str(e)}")

# 2. 语义文本比对搜索接口
@app.get("/images/search", summary="根据文本搜索相似图片")
async def search_images_by_text(text: str, top_k: int = 5):
    try:
        # 将文本转换为向量
        text_embedding = clip_client.encode_text([text])

        # 在FAISS索引中搜索相似向量
        distances, indices = index.search(np.array(text_embedding).astype('float32'), top_k)

        # 准备结果
        results = []
        for i, idx in enumerate(indices[0]):
            # 找到与索引对应的image_id
            image_id = list(image_metadata.keys())[idx] if idx < len(image_metadata) else None
            if image_id:
                results.append({
                    "image_id": image_id,
                    "filename": image_metadata[image_id]["filename"],
                    "similarity_score": float(1 / (1 + distances[0][i])),  # 转换为相似度分数
                    "rank": i + 1
                })

        return {"status": "success", "results": results, "query_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

# 3. 基于语义文本智能打标签接口
@app.post("/images/tag", summary="为图片智能打标签")
async def auto_tag_image(file: UploadFile = File(...), candidate_tags: list[str] = None):
    try:
        # 如果没有提供候选标签，使用默认标签集
        if not candidate_tags:
            candidate_tags = [
                "nature", "people", "city", "animal", "food",
                "car", "building", "mountain", "ocean", "indoor",
                "outdoor", "night", "day", "sunny", "rainy"
            ]

        # 读取图片并转换为向量
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_embedding = clip_client.encode_image([image])

        # 将候选标签转换为向量
        tag_embeddings = clip_client.encode_text(candidate_tags)

        # 计算图片向量与每个标签向量的相似度
        similarities = np.dot(image_embedding, np.array(tag_embeddings).T).flatten()

        # 按相似度排序并返回前5个标签
        top_indices = similarities.argsort()[-5:][::-1]
        top_tags = [
            {
                "tag": candidate_tags[i],
                "confidence": float(similarities[i])
            }
            for i in top_indices
        ]

        return {"status": "success", "filename": file.filename, "tags": top_tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"标签生成失败: {str(e)}")

# 启动服务时的初始化
@app.on_event("startup")
async def startup_event():
    print("多模态检索API服务已启动")
    print(f"当前向量库中的图片数量: {index.ntotal}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
