"""
CLIP模型编码器模块
用于提取图像和文本的特征向量
"""

import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from typing import Union, List, Optional
import os
from pathlib import Path


def _cuda_memory_snapshot() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    return (
        f"allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()}"
    )


class CLIPEncoder:
    """
    CLIP模型封装类
    提供图像和文本的特征向量提取功能
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        初始化CLIP编码器

        Args:
            model_name: CLIP模型名称，默认为"clip-ViT-B-32"
        """
        self.model_name = model_name
        print(f"正在加载CLIP模型: {model_name}")

        # 1. 强制声明这是一个 CLIP 多模态视觉模型
        clip_module = models.CLIPModel(model_name)
        
        # 2. 用专门的视觉模块来组装 SentenceTransformer
        self.model = SentenceTransformer(modules=[clip_module])

        # 检查是否有可用的GPU
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device('cuda'))
            self.device = torch.device('cuda')
            print("✓ 使用GPU加速特征提取")
        else:
            self.device = torch.device('cpu')
            print("⚠ GPU不可用，将使用CPU进行特征提取")

        # 设置为评估模式
        self.model.eval()

        print(f"✓ CLIP模型加载完成")

        print(
            f"[CLIP_DEBUG] initialized model_name={self.model_name} "
            f"device={self.device} cuda_memory {_cuda_memory_snapshot()}"
        )

    def encode_images(
        self,
        images: Union[List[Union[str, Image.Image]], Union[str, Image.Image]],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        批量提取图像特征向量

        Args:
            images: 图像路径列表或PIL Image对象列表
            batch_size: 批处理大小

        Returns:
            np.ndarray: 特征向量数组，形状为(N, feature_dim)
        """
        # 确保输入是列表形式
        if isinstance(images, (str, Image.Image)):
            images = [images]

        # 处理图像路径为PIL Image对象
        pil_images = []
        for img in images:
            if isinstance(img, str):
                # 如果是字符串，认为是文件路径
                pil_img = Image.open(img)
                pil_images.append(pil_img)
            elif isinstance(img, Image.Image):
                # 如果已经是PIL Image对象，直接添加
                pil_images.append(img)
            else:
                raise ValueError(f"不支持的图像类型: {type(img)}")

        print(f"正在提取 {len(pil_images)} 张图像的特征向量...")

        # 使用模型进行编码
        with torch.no_grad():  # 不需要计算梯度
            embeddings = self.model.encode(
                pil_images,
                batch_size=batch_size,
                convert_to_tensor=False,  # 返回numpy数组而不是tensor
                normalize_embeddings=True  # 归一化特征向量
            )

        embeddings = np.array(embeddings)
        print(f"✓ 图像特征提取完成，输出形状: {embeddings.shape}")
        return embeddings

    def encode_texts(
        self,
        texts: Union[List[str], str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        批量提取文本特征向量

        Args:
            texts: 文本列表或单个文本
            batch_size: 批处理大小

        Returns:
            np.ndarray: 特征向量数组，形状为(N, feature_dim)
        """
        # 确保输入是列表形式
        if isinstance(texts, str):
            texts = [texts]

        print(f"正在提取 {len(texts)} 条文本的特征向量...")

        # 使用模型进行编码
        with torch.no_grad():  # 不需要计算梯度
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,  # 返回numpy数组而不是tensor
                normalize_embeddings=True  # 归一化特征向量
            )

        embeddings = np.array(embeddings)
        print(f"✓ 文本特征提取完成，输出形状: {embeddings.shape}")
        return embeddings

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        计算查询向量与候选向量之间的余弦相似度

        Args:
            query_embedding: 查询特征向量，形状为(feature_dim,)
            candidate_embeddings: 候选特征向量，形状为(N, feature_dim)

        Returns:
            np.ndarray: 相似度分数数组，形状为(N,)
        """
        # 确保输入维度正确
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if candidate_embeddings.ndim == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)

        # 计算余弦相似度（对于已经归一化的向量，就是点积）
        similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

        return similarities

    def get_feature_dimension(self) -> int:
        """
        获取特征向量的维度

        Returns:
            int: 特征向量维度
        """
        # 使用一个简单的测试文本获取特征维度
        test_embedding = self.encode_texts(["test"])
        return test_embedding.shape[1]

    @staticmethod
    def load_image_safe(image_path: str) -> Optional[Image.Image]:
        """
        安全地加载图像，处理可能的错误

        Args:
            image_path: 图像文件路径

        Returns:
            PIL Image对象或None（如果加载失败）
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"加载图像失败 {image_path}: {str(e)}")
            return None


def preprocess_image_for_clip(image: Image.Image) -> Image.Image:
    """
    为CLIP模型预处理图像（虽然SentenceTransformer内部会自动处理，
    但这里提供手动预处理选项）

    Args:
        image: 输入PIL Image

    Returns:
        预处理后的PIL Image
    """
    # CLIP模型通常期望RGB图像
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


# 示例使用
if __name__ == "__main__":
    import sys

    # 初始化编码器
    encoder = CLIPEncoder()

    if len(sys.argv) > 1:
        # 如果提供了图像路径，进行图像编码测试
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            embedding = encoder.encode_images([img_path])
            print(f"图像特征向量维度: {embedding.shape}")
            print(f"特征向量范数: {np.linalg.norm(embedding[0]):.4f}")
        else:
            print(f"图像文件不存在: {img_path}")
    else:
        # 简单测试
        print(f"CLIP模型特征维度: {encoder.get_feature_dimension()}")

        # 测试文本编码
        texts = ["工业车辆ADAS系统", "叉车安全装置"]
        text_embeddings = encoder.encode_texts(texts)
        print(f"文本特征向量形状: {text_embeddings.shape}")

        # 测试图像编码（如果目录中有图像）
        sample_images = [
            img for img in os.listdir("./data/page_images/")
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ] if os.path.exists("./data/page_images/") else []

        if sample_images:
            sample_img_path = os.path.join("./data/page_images/", sample_images[0])
            img_embeddings = encoder.encode_images([sample_img_path])
            print(f"图像特征向量形状: {img_embeddings.shape}")
        else:
            print("提示: 在 ./data/page_images/ 目录下放入一些图像文件来测试图像编码功能")
