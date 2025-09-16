from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from clip_client import Client
import faiss
import numpy as np
import os
import uuid
from PIL import Image
import io

app = FastAPI(title="多模态检索API服务")

# 初始化CLIP客户端，增加重试机制
clip_client = Client("grpc://0.0.0.0:51000", timeout=30, max_retries=3)

VECTOR_DIM = 512
index = faiss.IndexFlatIP(VECTOR_DIM)  # 改用内积计算（更适合归一化向量）

image_metadata = {}
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/images/add", summary="添加图片到向量库")
async def add_image_to_vector_db(file: UploadFile = File(...)):
    try:
        # 保存图片
        image_id = str(uuid.uuid4())
        image_ext = file.filename.split(".")[-1].lower()
        image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.{image_ext}")

        with open(image_path, "wb") as f:
            f.write(await file.read())

        # 编码图片并归一化
        image_embedding = clip_client.encode([image_path])
        if image_embedding is None or len(image_embedding) == 0:
            raise ValueError("CLIP服务未能生成有效的图片向量")

        # 归一化向量（对相似度计算很重要）
        image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)

        # 存储元数据
        image_metadata[image_id] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "local_path": image_path
        }

        # 添加到索引
        index.add(np.array(image_embedding).astype("float32"))

        return {
            "status": "success",
            "image_id": image_id,
            "message": "图片已成功添加到向量库",
            "local_path": image_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加图片失败: {str(e)}")


@app.get("/images/search", summary="根据文本搜索相似图片")
async def search_images_by_text(text: str, top_k: int = 5):
    try:
        # 对中文查询进行优化，同时提供中英文提示
        enhanced_text = f"{text} 英语: {' '.join(text.split())}"  # 简单增强

        # 编码文本并归一化
        text_embedding = clip_client.encode([enhanced_text])
        if text_embedding is None or len(text_embedding) == 0:
            raise ValueError("CLIP服务未能生成有效的文本向量")

        # 归一化向量
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)

        if index.ntotal == 0:
            raise HTTPException(status_code=400, detail="向量库为空，请先添加图片")

        # 搜索相似向量（内积越大越相似）
        top_k = min(top_k, index.ntotal)
        similarities, indices = index.search(np.array(text_embedding).astype("float32"), top_k)

        # 匹配结果
        results = []
        image_ids = list(image_metadata.keys())
        for i, idx in enumerate(indices[0]):
            if idx < len(image_ids):
                img_id = image_ids[idx]
                results.append({
                    "image_id": img_id,
                    "filename": image_metadata[img_id]["filename"],
                    "similarity_score": float(similarities[0][i]),  # 直接使用内积作为相似度
                    "rank": i + 1
                })

        # 按相似度重新排序（确保正确的顺序）
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {
            "status": "success",
            "query_text": text,
            "enhanced_query": enhanced_text,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


# 其他接口保持不变...
@app.post("/images/tag", summary="为图片智能打标签")
async def auto_tag_image(
    file: UploadFile = File(...),
    candidate_tags: str = Form(None)
):
    # 保持原有实现...
    try:
        if candidate_tags:
            candidate_tags = [tag.strip() for tag in candidate_tags.split(',') if tag.strip()]

        if not candidate_tags:
            candidate_tags = [
                "nature", "people", "city", "animal", "food",
                "car", "building", "mountain", "ocean", "indoor",
                "outdoor", "night", "day", "sunny", "rainy"
            ]

        temp_image_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())

        try:
            with Image.open(temp_image_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"无效的图片文件: {str(e)}")

        image_embedding = clip_client.encode([temp_image_path])
        if image_embedding is None or len(image_embedding) == 0:
            raise ValueError("CLIP服务未能生成有效的图片向量")

        tag_embeddings = clip_client.encode(candidate_tags)
        if tag_embeddings is None or len(tag_embeddings) == 0:
            raise ValueError("CLIP服务未能生成有效的标签向量")

        similarities = np.dot(image_embedding, np.transpose(tag_embeddings)).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        top_tags = [
            {
                "tag": candidate_tags[i],
                "confidence": float(similarities[i])
            }
            for i in top_indices
        ]

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return {
            "status": "success",
            "filename": file.filename,
            "tags": top_tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"标签生成失败: {str(e)}")


@app.on_event("startup")
async def startup_event():
    print("多模态检索API服务已启动")
    print(f"当前向量库中的图片数量: {index.ntotal}")
    print(f"图片上传目录: {os.path.abspath(UPLOAD_FOLDER)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
