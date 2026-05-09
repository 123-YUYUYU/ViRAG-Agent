"""
多模态检索器模块
基于ChromaDB实现图像-文本的语义检索功能
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
from chromadb.api.types import QueryResult
import logging
from pathlib import Path


class VisualRetriever:
    """
    视觉检索器类
    基于ChromaDB实现对图像特征的高效检索
    """

    def __init__(
        self,
        collection_name: str = "adas_manual_visual_index",
        db_path: str = "./data/chroma_db",
        distance_metric: str = "cosine",
        embedding_function=None
    ):
        """
        初始化视觉检索器

        Args:
            collection_name: ChromaDB集合名称
            db_path: 数据库存储路径
            distance_metric: 距离度量方式，默认为余弦距离
            embedding_function: 嵌入函数，如果提供则使用此函数
        """
        self.collection_name = collection_name
        self.db_path = db_path
        self.distance_metric = distance_metric
        self.embedding_function = embedding_function

        # 确保数据库目录存在
        os.makedirs(db_path, exist_ok=True)

        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)  # 关闭遥测
        )

        # 获取或创建集合
        # 使用原生 API 安全获取或创建集合
        collection_params = {
            "name": collection_name,
            "metadata": {"hnsw:space": distance_metric}
        }

        # 如果提供了embedding_function，添加到参数中
        if self.embedding_function is not None:
            collection_params["embedding_function"] = self.embedding_function

        self.collection = self.client.get_or_create_collection(**collection_params)
        print(f"✓ 成功连接或创建集合: {collection_name}")

        # 检查集合中是否有数据
        count = self.collection.count()
        print(f"📊 集合中现有向量数量: {count}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        添加图像特征向量到数据库

        Args:
            embeddings: 特征向量数组，形状为(N, feature_dim)
            image_paths: 图像路径列表
            metadatas: 元数据列表
            ids: 向量ID列表，如果不提供则自动生成
        """
        assert len(embeddings) == len(image_paths) == len(metadatas), \
            "嵌入向量、图像路径和元数据的数量必须一致"

        # 如果没有提供ID，生成默认ID
        if ids is None:
            ids = [f"img_{i}" for i in range(len(embeddings))]

        print(f"正在添加 {len(embeddings)} 个向量到数据库...")

        # 添加到ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),  # ChromaDB期望列表格式
            documents=image_paths,  # 存储图像路径作为文档
            metadatas=metadatas,    # 存储元数据
            ids=ids                 # 向量ID
        )

        print(f"✓ 成功添加 {len(embeddings)} 个向量")

    def search_by_text_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        使用文本特征向量进行检索

        Args:
            query_embedding: 查询特征向量
            top_k: 返回最相似的前k个结果
            threshold: 相似度阈值，低于此值的结果会被过滤

        Returns:
            List[Dict]: 检索结果列表，包含图像路径、相似度分数和元数据
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        print(f"正在使用特征向量检索最相似的 {top_k} 个图像...")

        # 执行查询
        results: QueryResult = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["distances", "documents", "metadatas"]
        )

        # 整理结果
        search_results = []
        distances = results["distances"][0] if results["distances"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        for i in range(len(distances)):
            distance = distances[i]
            similarity = 1 - distance  # 转换为相似度（越接近1越相似）

            # 只返回超过阈值的结果
            if similarity >= threshold:
                result = {
                    "image_path": documents[i],
                    "similarity": similarity,
                    "distance": distance,
                    "metadata": metadatas[i] if metadatas else {}
                }
                search_results.append(result)

        print(f"✓ 检索完成，找到 {len(search_results)} 个符合条件的结果")
        return search_results

    def search_by_text(
        self,
        query: str,
        text_encoder,  # CLIPEncoder实例
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        使用文本查询进行检索

        Args:
            query: 查询文本
            text_encoder: 文本编码器（CLIPEncoder实例）
            top_k: 返回最相似的前k个结果
            threshold: 相似度阈值

        Returns:
            List[Dict]: 检索结果列表
        """
        print(f"正在检索文本查询: '{query}'")

        # 编码查询文本
        query_embedding = text_encoder.encode_texts([query])

        # 使用编码后的向量进行检索
        return self.search_by_text_embedding(
            query_embedding=query_embedding[0],  # 取第一个向量
            top_k=top_k,
            threshold=threshold
        )

    def search_by_image(
        self,
        query_image_path: str,
        image_encoder,  # CLIPEncoder实例
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        使用图像进行检索（以图搜图）

        Args:
            query_image_path: 查询图像路径
            image_encoder: 图像编码器（CLIPEncoder实例）
            top_k: 返回最相似的前k个结果
            threshold: 相似度阈值

        Returns:
            List[Dict]: 检索结果列表
        """
        print(f"正在检索图像: '{query_image_path}'")

        # 编码查询图像
        query_embedding = image_encoder.encode_images([query_image_path])

        # 使用编码后的向量进行检索
        return self.search_by_text_embedding(
            query_embedding=query_embedding[0],  # 取第一个向量
            top_k=top_k,
            threshold=threshold
        )

    def get_collection_stats(self) -> Dict:
        """
        获取集合统计信息

        Returns:
            Dict: 包含集合统计信息的字典
        """
        count = self.collection.count()

        # 获取所有ID以了解集合的多样性
        try:
            all_ids = self.collection.get(limit=1)["ids"]
            sample_size = min(count, 10)  # 只取样少量数据
            sample_data = self.collection.get(limit=sample_size)

            stats = {
                "total_vectors": count,
                "collection_name": self.collection_name,
                "database_path": self.db_path,
                "distance_metric": self.distance_metric,
                "sample_ids_count": len(all_ids[0]) if all_ids else 0
            }
        except:
            stats = {
                "total_vectors": count,
                "collection_name": self.collection_name,
                "database_path": self.db_path,
                "distance_metric": self.distance_metric
            }

        return stats

    def delete_collection(self):
        """删除当前集合（谨慎使用）"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️ 集合 '{self.collection_name}' 已被删除")
        except Exception as e:
            print(f"❌ 删除集合失败: {str(e)}")

    def reset_collection(self):
        """重置集合，清空所有数据"""
        try:
            self.delete_collection()
            # 重新创建集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            print(f"🔄 集合 '{self.collection_name}' 已重置")
        except Exception as e:
            print(f"❌ 重置集合失败: {str(e)}")


def build_visual_index_from_images(
    image_data: List[Tuple[str, Dict]],
    clip_encoder,
    retriever: VisualRetriever,
    batch_size: int = 32
) -> None:
    """
    从图像数据批量构建视觉索引

    Args:
        image_data: 包含(图像路径, 元数据)的列表
        clip_encoder: CLIP编码器实例
        retriever: 视觉检索器实例
        batch_size: 批处理大小
    """
    print(f"开始批量构建视觉索引，总共 {len(image_data)} 个图像...")

    total_processed = 0

    # 分批处理
    for i in range(0, len(image_data), batch_size):
        batch = image_data[i:i + batch_size]
        image_paths = [item[0] for item in batch]
        metadatas = [item[1] for item in batch]

        print(f"处理批次 {i//batch_size + 1}: {len(batch)} 个图像")

        # 批量编码图像
        try:
            batch_embeddings = clip_encoder.encode_images(
                image_paths,
                batch_size=min(batch_size, 8)  # 图像编码使用较小的batch
            )

            # 生成批次ID
            batch_ids = [f"img_{j + i}" for j in range(len(batch))]

            # 添加到检索器
            retriever.add_embeddings(
                embeddings=batch_embeddings,
                image_paths=image_paths,
                metadatas=metadatas,
                ids=batch_ids
            )

            total_processed += len(batch)
            print(f"  ✓ 批次完成，累计处理 {total_processed}/{len(image_data)}")

        except Exception as e:
            print(f"  ❌ 批次处理失败: {str(e)}")
            continue

    print(f"✅ 视觉索引构建完成，总共处理 {total_processed} 个图像")


# 使用示例
if __name__ == "__main__":
    from model.clip_encoder import CLIPEncoder

    # 初始化组件
    encoder = CLIPEncoder()
    retriever = VisualRetriever()

    print("检索器统计信息:")
    stats = retriever.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 示例：如果有图像数据，可以这样添加
    # 假设我们有一些图像路径和元数据
    sample_images = [
        img for img in os.listdir("./data/page_images/")
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ] if os.path.exists("./data/page_images/") else []

    if sample_images and len(sample_images) > 0:
        sample_img_path = os.path.join("./data/page_images/", sample_images[0])

        print(f"\n演示以图搜图功能...")
        results = retriever.search_by_image(
            query_image_path=sample_img_path,
            image_encoder=encoder,
            top_k=3
        )

        for i, result in enumerate(results):
            print(f"  结果 {i+1}:")
            print(f"    图像路径: {result['image_path']}")
            print(f"    相似度: {result['similarity']:.4f}")
            print(f"    距离: {result['distance']:.4f}")
    else:
        print("\n提示: 请先运行构建脚本生成图像数据，然后才能进行检索测试")