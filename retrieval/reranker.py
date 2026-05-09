from sentence_transformers import CrossEncoder
from typing import List
from data_types import DocBlock
import torch


def _cuda_memory_snapshot() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    return (
        f"allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()}"
    )


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化交叉编码重排序器

        Args:
            model_name: 重排序模型名称
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        try:
            self.device = next(self.model.model.parameters()).device
        except Exception:
            self.device = "unknown"
        print(
            f"[RERANKER_DEBUG] initialized model_name={self.model_name} "
            f"device={self.device} cuda_memory {_cuda_memory_snapshot()}"
        )

    def rerank(self, query: str, docs: List[DocBlock], top_k: int = 3) -> List[DocBlock]:
        """
        对文档进行重排序

        Args:
            query: 查询字符串
            docs: 待重排序的文档块列表
            top_k: 返回前k个结果

        Returns:
            List[DocBlock]: 重排序后的文档块列表
        """
        if not docs:
            return []
        print(f"[RERANKER_DEBUG] rerank received top_k={top_k}")

        # 准备重排序的文本对
        sentence_pairs = []
        for doc in docs:
            if doc.block_type == "text" or doc.block_type == "table":
                text_content = doc.content
            elif doc.block_type == "image":
                # 使用OCR文本或标题
                text_content = doc.metadata.get("ocr_text", "") or doc.content
            else:
                text_content = doc.content

            sentence_pairs.append([query, text_content])

        # 计算相似度得分
        scores = self.model.predict(sentence_pairs)

        # 根据得分排序
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 返回前top_k个文档
        reranked_docs = [doc for doc, score in scored_docs[:top_k]]

        print(f"[RERANKER_DEBUG] returned_evidence_count={len(reranked_docs)}")
        for idx, doc in enumerate(reranked_docs):
            image_path = doc.content if doc.block_type == "image" else doc.metadata.get("image_path")
            print(
                f"[RERANKER_DEBUG] evidence[{idx}] image_path={image_path} "
                f"has_text={bool(doc.content or doc.metadata.get('ocr_text'))} "
                f"has_metadata={bool(doc.metadata)}"
            )

        return reranked_docs
