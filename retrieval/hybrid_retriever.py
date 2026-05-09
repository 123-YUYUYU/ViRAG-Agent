import numpy as np
from typing import List, Dict
from data_types import DocBlock
from rank_bm25 import BM25Okapi
import jieba
import re


class HybridRetriever:
    def __init__(self, chroma_collection, docstore: Dict[str, DocBlock], clip_encoder=None):
        """
        初始化混合检索器

        Args:
            chroma_collection: ChromaDB集合
            docstore: id到DocBlock的映射
            clip_encoder: CLIP编码器实例，用于将文本转换为向量
        """
        self.chroma_collection = chroma_collection
        self.docstore = docstore
        self.docstore = docstore
        self.clip_encoder = clip_encoder
        self.docs_list = list(docstore.values())

        # 构建BM25索引
        self._build_bm25_index()

    def _build_bm25_index(self):
        """构建BM25索引"""
        # 准备BM25的文本语料库
        corpus = []
        for doc in self.docs_list:
            if doc.block_type == "text" or doc.block_type == "table":
                # 对文本进行分词
                tokens = jieba.lcut(doc.content)
                # 过滤掉标点符号和空白字符
                tokens = [token.strip() for token in tokens if token.strip() and not re.match(r'[^\w\s]', token)]
                corpus.append(tokens)
            elif doc.block_type == "image":
                # 使用OCR文本进行检索
                ocr_text = doc.metadata.get("ocr_text", "")
                if ocr_text:
                    tokens = jieba.lcut(ocr_text)
                    tokens = [token.strip() for token in tokens if token.strip() and not re.match(r'[^\w\s]', token)]
                    corpus.append(tokens)
                else:
                    # 如果没有OCR文本，尝试使用其他可能的文本描述
                    tokens = jieba.lcut(doc.content if doc.content else "")  # 这可能是图像路径
                    tokens = [token.strip() for token in tokens if token.strip() and not re.match(r'[^\w\s]', token)]
                    corpus.append(tokens)

        # 初始化BM25模型
        self.bm25_model = BM25Okapi(corpus)

        # 🚀 【新增验证测试代码】检查 BM25 是否真的吃到了文字！
        print("\n" + "="*40)
        print(f"🛠️ [调试] BM25 语料库初始化完毕！")
        print(f"🛠️ [调试] 总共为 {len(corpus)} 个文档建立了文字索引。")
        if corpus:
            print(f"🛠️ [调试] 抽查第 1 页的文字分词结果 (前10个词): {corpus[0][:10]}")
        print("="*40 + "\n")

    def retrieve(self, query: str, top_k: int = 5) -> List[DocBlock]:
        """
        混合检索方法

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            List[DocBlock]: 检索到的文档块列表
        """
        # 1. Dense检索 (ChromaDB) - 先用CLIP编码器将文本转为向量
        print(f"[HYBRID_RETRIEVAL_DEBUG] retrieve received top_k={top_k}")
        if self.clip_encoder:
            # 使用CLIP编码器将查询文本转换为向量
            query_embedding = self.clip_encoder.encode_texts([query])
            query_embed = query_embedding[0]  # 取第一个向量
            # 将numpy数组转换为列表格式
            query_embed_list = query_embed.tolist() if hasattr(query_embed, 'tolist') else query_embed

            chroma_results = self.chroma_collection.query(
                query_embeddings=[query_embed_list],  # 直接传入预计算的向量
                n_results=top_k * 2  # 获取更多结果用于后续融合
            )
        else:
            # 如果没有CLIP编码器，退回到原来的文本查询方式
            chroma_results = self.chroma_collection.query(
                query_texts=[query],  # 仍可能引起冲突，但作为备用
                n_results=top_k * 2
            )

        # 将Chroma结果转换为DocBlock列表
        dense_docs = []
        if 'ids' in chroma_results and chroma_results['ids']:
            for doc_id in chroma_results['ids'][0]:
                if doc_id in self.docstore:
                    dense_docs.append(self.docstore[doc_id])

        # 2. Sparse检索 (BM25)
        query_tokens = jieba.lcut(query)
        query_tokens = [token.strip() for token in query_tokens if token.strip()]

        bm25_scores = self.bm25_model.get_scores(query_tokens)
        # 获取top-k的BM25结果
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        sparse_docs = [self.docs_list[i] for i in bm25_top_indices if bm25_scores[i] > 0]

        # 3. RRF (Reciprocal Rank Fusion) 融合
        k = 60  # RRF参数，通常设为k=60

        # 计算密集检索得分
        dense_scores = {}
        for idx, doc in enumerate(dense_docs):
            rank = idx + 1
            dense_scores[doc.id] = 1 / (k + rank)

        # 计算稀疏检索得分
        sparse_scores = {}
        for idx, doc in enumerate(sparse_docs):
            rank = idx + 1
            sparse_scores[doc.id] = 1 / (k + rank)

        # 融合得分
        fused_scores = {}
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        for doc_id in all_doc_ids:
            fused_score = dense_scores.get(doc_id, 0) + sparse_scores.get(doc_id, 0)
            fused_scores[doc_id] = fused_score

        # 根据融合得分排序
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]

        # 返回对应的DocBlock
        result_docs = [self.docstore[doc_id] for doc_id in sorted_doc_ids if doc_id in self.docstore]
        print(f"[HYBRID_RETRIEVAL_DEBUG] returned_evidence_count={len(result_docs)}")
        for idx, doc in enumerate(result_docs):
            image_path = doc.content if doc.block_type == "image" else doc.metadata.get("image_path")
            print(
                f"[HYBRID_RETRIEVAL_DEBUG] evidence[{idx}] image_path={image_path} "
                f"has_text={bool(doc.content or doc.metadata.get('ocr_text'))} "
                f"has_metadata={bool(doc.metadata)}"
            )

        return result_docs
