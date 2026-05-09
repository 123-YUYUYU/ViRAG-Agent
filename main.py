"""
ADAS多模态Agent系统启动脚本
整合所有模块并提供一键启动功能
"""

import sys
import os
from pathlib import Path
import argparse
import subprocess
import time
from typing import Dict

# 原有导入
import torch


def check_dependencies():
    """检查所需依赖"""
    import config

    required_packages = [
        'torch', 'transformers',
        'sentence_transformers', 'PIL', 'opencv-python',
        'chromadb', 'gradio', 'PyMuPDF'
    ]
    if config.LLM_BACKEND == "local":
        required_packages.append('qwen_vl_utils')
    elif config.LLM_BACKEND == "qwen_api":
        required_packages.append('openai')

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'qwen_vl_utils':
                __import__('qwen_vl_utils')
            elif package == 'PyMuPDF':
                __import__('fitz')
            elif package == 'opencv-python':
                __import__('cv2')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"缺失的包: {missing_packages}")
        print("请运行: pip install -r requirements.txt")
        return False

    print("所有依赖检查成功")
    return True


def check_data_files():
    """检查所需数据文件"""
    checks = [
        ("PDF文档", "./data/manual/", "*.pdf"),
        ("图像文件", "./data/page_images/", "*.png"),
        ("向量数据库", "./data/chroma_db/", "chroma.sqlite3 或其他数据库文件")
    ]

    print("\n检查数据文件...")
    for name, path, pattern in checks:
        path_obj = Path(path)
        if path_obj.exists():
            if name == "向量数据库":
                # 检查数据库文件
                db_files = list(path_obj.glob("*"))
                if db_files:
                    print(f"数据库文件: 找到 {len(db_files)} 个文件")
                else:
                    print(f"向量数据库: 目录存在但没有数据库文件")
            else:
                files = list(path_obj.glob(pattern.replace('*', '*.*')))
                print(f"{name}: 找到 {len(files)} 个文件" if files else f"{name}: 未找到文件")
        else:
            print(f"{name}: 路径不存在 ({path})")

    return True


def start_web_interface():
    """启动Web界面"""
    print("\n启动ADAS多模态Agent Web界面...")

    try:
        # 导入并启动Gradio界面
        from apps.web_demo import launch_demo
        launch_demo()
    except Exception as e:
        print(f"Web界面启动失败: {e}")
        print("请检查模型和数据文件是否就绪")


def start_cli_interface(use_hybrid_rerank=False):
    """启动命令行界面"""
    print(f"\n启动CLI交互模式... (混合检索模式: {'开启' if use_hybrid_rerank else '关闭'})")

    try:
        # 初始化组件
        import config
        from llm.client_factory import create_vlm_client
        from retrieval.retriever import VisualRetriever
        from model.clip_encoder import CLIPEncoder
        from agent.react_agent import SimpleVQAAgent, EvidenceAwareVQAAgent
        from data_types import DocBlock
        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.reranker import Reranker
        from tools.smart_zoom_tool import SmartZoomProcessor

        # 导入修复ChromaDB维度问题所需模块
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        import chromadb

        print("\n初始化模型组件...")

        # 初始化模型
        print(f"  初始化VLM backend: {config.LLM_BACKEND}...")
        qwen_client = create_vlm_client()

        print("  初始化CLIP编码器...")
        clip_encoder = CLIPEncoder(model_name="/root/autodl-tmp/hf_cache/clip-ViT-B-32/0_CLIPModel")

        print("  初始化向量检索器...")
        # 初始化向量检索器 - 使用干净的初始化，不传embedding_function
        retriever = VisualRetriever(
            collection_name="adas_manual_visual_index",
            db_path="./data/chroma_db"
        )

        # Check database
        stats = retriever.get_collection_stats()
        print(f"  数据库统计: {stats['total_vectors']} 个向量")

        if stats['total_vectors'] == 0:
            print("  数据库为空，请先运行构建脚本")
            return

        # 如果使用混合检索，则初始化附加组件
        if use_hybrid_rerank:
            # 从ChromaDB集合构建文档存储
            print("  构建文档存储...")
            all_docs = retriever.collection.get()
            docstore = {}
            for doc_id, doc_content, doc_metadata in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
                # 为每个文档创建DocBlock
                doc_block = DocBlock(
                    id=doc_id,
                    content=doc_content,  # 这是图像路径
                    block_type="image",  # 因为我们的数据库存储图像
                    metadata=doc_metadata
                )
                docstore[doc_id] = doc_block

            print(f"  构建了 {len(docstore)} 个文档块")

            # 初始化混合检索器
            print("  初始化混合检索器...")
            hybrid_retriever = HybridRetriever(
                chroma_collection=retriever.collection,
                docstore=docstore,
                clip_encoder=clip_encoder  # 传入CLIP编码器实例
            )

            # 初始化重排序器
            print("  初始化重排序器...")
            reranker = Reranker(model_name="/root/autodl-tmp/hf_cache/BAAI/bge-reranker-base/")

            # 初始化代理
            print("  初始化Evidence-aware代理...")
            agent = EvidenceAwareVQAAgent(qwen_client)

            print("\n混合检索组件初始化完成!")
        else:
            # 初始化简单代理
            print("  初始化VQA代理...")
            agent = SimpleVQAAgent(qwen_client)

            print("\n组件初始化完成!")

        print("开始交互 (输入 'quit' 退出)\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break

                if not user_input:
                    continue

                if use_hybrid_rerank:
                    # 使用混合检索和重排序
                    print("  正在执行混合检索...")
                    retrieved_docs = hybrid_retriever.retrieve(query=user_input, top_k=5)

                    # 重排结果
                    print("  正在重排序结果...")
                    reranked_docs = reranker.rerank(query=user_input, docs=retrieved_docs, top_k=1)

                    # 生成带证据的答案
                    print("  正在生成答案...")
                    result = agent.run(query=user_input, docs=reranked_docs)

                    print(f"Assistant: {result['answer']}\n")

                    # 打印来源
                    print("Sources:")
                    for source in result['sources']:
                        page = source.get("page_num", source.get("page", "Unknown"))
                        print(f"  - Page {page}, Type: {source['type']}, Source: {source['source']}")
                    print()
                else:
                    # 检查图像上传(简单模拟)
                    images = []  # 这可以扩展以支持图像上传
                    response = agent.answer(
                        question=user_input,
                        images=images
                    )

                    print(f"Assistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\n会话被中断，再见!")
                break
            except Exception as e:
                print(f"处理请求时出错: {e}\n")

    except Exception as e:
        print(f"CLI界面启动失败: {e}")
        import traceback
        traceback.print_exc()


def start_build_pipeline():
    """启动构建管道"""
    print("\n启动视觉向量数据库构建过程...")

    try:
        from scripts.build_vector_db import main as build_main
        build_main()
    except Exception as e:
        print(f"构建过程启动失败: {e}")
        import traceback
        traceback.print_exc()


def show_system_info():
    """显示系统信息"""
    import torch

    print("\nADAS多模态Agent系统信息")
    print("="*50)
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # 检查模型组件
    print("\n模型组件状态:")
    try:
        from transformers import __version__ as tf_version
        print(f"  Transformers: {tf_version}")
    except:
        print("  Transformers: 未安装")

    try:
        from PIL import __version__ as pil_version
        print(f"  PIL/Pillow: {pil_version}")
    except:
        print("  PIL/Pillow: 未安装")

    try:
        import chromadb
        print(f"  ChromaDB: {chromadb.__version__}")
    except:
        print("  ChromaDB: 未安装")

    try:
        import gradio
        print(f"  Gradio: {gradio.__version__}")
    except:
        print("  Gradio: 未安装")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ADAS多模态Agent系统')
    parser.add_argument('mode', nargs='?', default='web',
                       choices=['web', 'cli', 'build', 'info'],
                       help='运行模式: web(网页界面), cli(命令行), build(构建数据库), info(系统信息)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Web服务器端口 (仅web模式)')
    parser.add_argument('--hybrid', action='store_true',
                       help='在CLI模式中使用混合检索和重排序')
    parser.add_argument('--simple', action='store_true',
                       help='在CLI模式中使用简单检索 (如果未指定--hybrid或--simple则为默认模式)')

    args = parser.parse_args()

    print("ADAS工业车辆多模态Agent系统")
    print("基于Visual-RAG的智能问答系统")

    # 检查依赖
    if not check_dependencies():
        return

    # 检查数据文件
    check_data_files()

    print(f"\n目标模式: {args.mode}")

    if args.mode == 'web':
        start_web_interface()
    elif args.mode == 'cli':
        use_hybrid = args.hybrid
        # 如果未指定--hybrid或--simple，则默认为简单模式
        start_cli_interface(use_hybrid_rerank=use_hybrid)
    elif args.mode == 'build':
        start_build_pipeline()
    elif args.mode == 'info':
        show_system_info()
    else:
        print(f"未知模式: {args.mode}")
        print("可用模式: web, cli, build, info")


if __name__ == "__main__":
    main()
