"""
ADAS多模态Agent网页界面
使用Gradio实现交互式对话界面
"""

import gradio as gr
import torch
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil

import config
from llm.client_factory import create_vlm_client
from retrieval.retriever import VisualRetriever
from model.clip_encoder import CLIPEncoder
from agent.react_agent import ReActAgent, SimpleVQAAgent, EvidenceAwareVQAAgent
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from data_types import DocBlock
from agent.memory import ConversationMemory
from tools.smart_zoom_tool import visualize_zoom_area
from utils.logger import (
    log_info, log_error, log_success, log_query, log_result,
    log_tool_call, log_agent_response
)


class WebDemo:
    """
    Web演示界面类
    封装Gradio界面的所有功能
    """

    def __init__(self):
        """初始化网页界面"""
        self.qwen_client = None
        self.visual_retriever = None
        self.clip_encoder = None
        self.agent = None
        self.hybrid_retriever = None
        self.reranker = None
        self.docstore = None
        self.memory = ConversationMemory(max_history_length=20)

        # 状态变量
        self.last_zoom_image = None
        self.conversation_state = []

        print("Web界面初始化完成")

    def initialize_models(self):
        """初始化模型组件"""
        try:
            log_info("Initializing model components...")

            # 初始化VLM客户端
            log_info(f"Initializing VLM backend: {config.LLM_BACKEND}...")
            self.qwen_client = create_vlm_client()

            # 初始化CLIP编码器
            log_info("Initializing CLIP encoder...")
            self.clip_encoder = CLIPEncoder(model_name="/root/autodl-tmp/hf_cache/clip-ViT-B-32/0_CLIPModel")

            # 初始化向量检索器
            log_info("Initializing vector retriever...")
            self.visual_retriever = VisualRetriever(
                collection_name="adas_manual_visual_index",
                db_path="./data/chroma_db"
            )

            # 检查向量数据库是否为空
            stats = self.visual_retriever.get_collection_stats()
            if stats["total_vectors"] == 0:
                log_error("Vector database is empty! Please run the build script first")
            else:
                log_success(f"Vector database contains {stats['total_vectors']} vectors")

                # 构建文档存储
                log_info("Building document store...")
                all_docs = self.visual_retriever.collection.get()
                self.docstore = {}
                for doc_id, doc_content, doc_metadata in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
                    # Create DocBlock for each document
                    doc_block = DocBlock(
                        id=doc_id,
                        content=doc_content,  # This is image path
                        block_type="image",  # Since our DB stores images
                        metadata=doc_metadata
                    )
                    self.docstore[doc_id] = doc_block

                log_success(f"Built {len(self.docstore)} document blocks")

                # 初始化混合检索器
                log_info("Initializing hybrid retriever...")
                self.hybrid_retriever = HybridRetriever(
                    chroma_collection=self.visual_retriever.collection,
                    docstore=self.docstore,
                    clip_encoder=self.clip_encoder  # 传入CLIP编码器实例
                )

                # 初始化重排序器
                log_info("Initializing reranker...")
                self.reranker = Reranker(model_name="/root/autodl-tmp/hf_cache/bge-reranker-base/")

            # 初始化代理
            log_info("Initializing ReAct agent...")
            self.agent = ReActAgent(
                qwen_client=self.qwen_client,
                visual_retriever=self.visual_retriever,
                clip_encoder=self.clip_encoder  # 确保传入已初始化的CLIP编码器
            )

            log_success("All model components initialized!")

            return f"SUCCESS: Model initialization complete!\n- Vector database: {stats['total_vectors']} vectors\n- Document store: {len(self.docstore) if self.docstore else 0} docs\n- Device: {self.qwen_client.device}"
        except Exception as e:
            error_msg = f"ERROR: Model initialization failed: {str(e)}"
            log_error(error_msg)
            return error_msg

    def process_query(
        self,
        user_input: str,
        uploaded_images: List,
        history: List,
        use_react: bool = False,
        use_hybrid: bool = False
    ) -> tuple:
        """
        Process user query

        Args:
            user_input: User input text
            uploaded_images: Uploaded image list
            history: Conversation history
            use_react: Whether to use ReAct framework
            use_hybrid: Whether to use hybrid retrieval and reranking

        Returns:
            tuple: (updated history, retrieved image, thinking process, clear button state)
        """
        if not user_input.strip():
            return history, None, "Please enter a question", gr.update(visible=True)

        log_query(user_input)

        try:
            # 处理上传的图像
            image_paths = []
            if uploaded_images:
                for img in uploaded_images:
                    if isinstance(img, str) and Path(img).exists():
                        image_paths.append(img)
                    else:
                        # 如果是临时文件，则复制到临时目录
                        temp_dir = Path("./temp_uploaded")
                        temp_dir.mkdir(exist_ok=True)
                        temp_path = temp_dir / f"temp_{int(time.time())}.png"

                        # 将PIL图像保存到临时路径
                        if hasattr(img, 'save'):
                            img.save(temp_path)
                        else:
                            # 假设img已经是路径
                            temp_path = img

                        image_paths.append(str(temp_path))

            # 初始化模型（如果尚未初始化）
            if self.qwen_client is None:
                init_msg = self.initialize_models()
                if "ERROR:" in init_msg:
                    return history + [(user_input, init_msg)], None, "Model initialization failed", gr.update(visible=True)

            # 根据所选模式处理
            if use_hybrid and self.hybrid_retriever and self.reranker:
                # 使用证据感知代理，结合混合检索和重排序
                evidence_agent = EvidenceAwareVQAAgent(self.qwen_client)

                # 执行混合检索
                log_info("Executing hybrid retrieval...")
                retrieved_docs = self.hybrid_retriever.retrieve(query=user_input, top_k=5)

                # 重排序结果
                log_info("Reranking results...")
                reranked_docs = self.reranker.rerank(query=user_input, docs=retrieved_docs, top_k=1)

                # 生成响应
                log_info("Generating response with evidence...")
                result = evidence_agent.run(query=user_input, docs=reranked_docs)

                response = result['answer']

                # 更新对话历史
                self.memory.add_user_message(user_input)
                self.memory.add_assistant_message(response)

                # 获取第一个检索结果的图像作为展示
                retrieval_img = None
                if result.get('sources'):
                    first_source = result['sources'][0]
                    if 'image_path' in first_source:
                        retrieval_img = first_source['image_path']
                    elif 'content' in first_source:
                        retrieval_img = first_source['content']  # This might be the image path

                thinking_process = f"Hybrid retrieval + reranking: Found {len(reranked_docs)} results"

            elif use_react and self.agent:
                # 使用ReAct代理
                response = self.agent.chat(
                    user_input=user_input,
                    images=image_paths if image_paths else None
                )

                # 更新对话历史
                self.memory.add_user_message(user_input)
                self.memory.add_assistant_message(response)

                # 解析ReAct输出以提取相关信息
                retrieval_img = None
                thinking_process = "ReAct agent processing..."

                # 简单解析响应中的图像引用
                if "results" in str(response) and isinstance(response, dict):
                    # 如果响应包含检索结果
                    try:
                        resp_dict = response if isinstance(response, dict) else eval(str(response))
                        if "results" in resp_dict:
                            if resp_dict["results"]:
                                first_result = resp_dict["results"][0]
                                retrieval_img = first_result.get("image_path")
                    except:
                        pass

            else:
                # 使用简单VQA代理
                vqa_agent = SimpleVQAAgent(self.qwen_client)
                response = vqa_agent.answer(
                    question=user_input,
                    images=image_paths if image_paths else [],
                )

                # 更新对话历史
                self.memory.add_user_message(user_input)
                self.memory.add_assistant_message(response)

                # 提取用于显示的图像路径
                retrieval_img = image_paths[0] if image_paths else None
                thinking_process = f"Simple Q&A mode, Images: {len(image_paths) if image_paths else 0}"

            # 更新Gradio历史记录
            updated_history = history + [(user_input, response)]

            log_result(response[:200] + "..." if len(response) > 200 else response)

            return (
                updated_history,
                retrieval_img,  # Retrieved image
                thinking_process,  # Thinking process
                gr.update(visible=True)  # Show clear button
            )

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            log_error(error_msg)

            updated_history = history + [(user_input, error_msg)]
            return updated_history, None, f"Error: {str(e)}", gr.update(visible=True)

    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(title="ADAS Multimodal Agent", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # ADAS Industrial Vehicle Multimodal Agent

            Intelligent assistant for industrial vehicle ADAS installation and maintenance based on Visual-RAG
            """)

            # 状态栏
            with gr.Row():
                status_display = gr.Textbox(label="Status", interactive=False)
                init_btn = gr.Button("Initialize Models", variant="primary")

            # 主交互区域
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="Conversation History", height=500)
                        user_input = gr.Textbox(label="Your Question", placeholder="Enter questions about ADAS installation, maintenance...")

                        with gr.Row():
                            image_input = gr.Gallery(
                                label="Upload Images",
                                type="filepath",
                                elem_id="gallery",
                                columns=2,
                                object_fit="contain",
                                height="auto",
                                allow_preview=True
                            )

                        with gr.Row():
                            submit_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear Chat", visible=False)
                            use_react_checkbox = gr.Checkbox(label="Use ReAct Reasoning", value=False)
                            use_hybrid_checkbox = gr.Checkbox(label="Use Hybrid Retrieval + Reranking", value=False)

                    with gr.Column(scale=1):
                        retrieved_image = gr.Image(label="Retrieved Related Pages", height=300)
                        thinking_process = gr.Textbox(label="Thinking Process", interactive=False, max_lines=5)

            with gr.Tab("Instructions"):
                gr.Markdown("""
                ## Instructions

                ### Core Functions
                - **Visual Q&A**: Directly ask questions about uploaded drawings
                - **ReAct Reasoning**: Intelligent reasoning based on thought chains
                - **Smart Retrieval**: Find related drawings based on text descriptions
                - **Hybrid Retrieval**: Combine dense (vector) and sparse (BM25) retrieval with re-ranking
                - **Smart Zoom**: Automatically enlarge detail areas

                ### Usage Tips
                1. **Upload Drawings**: Click upload button to select ADAS-related images
                2. **Smart Queries**: Use natural language to describe what you want to query
                3. **ReAct Mode**: Check to enable complex reasoning chains
                4. **Hybrid Retrieval**: Check to use combined dense/sparse retrieval with re-ranking for better results
                5. **Detail Enlargement**: Model automatically identifies and enlarges unclear areas

                ### Supported Query Types
                - Circuit wiring analysis
                - Installation step guidance
                - Maintenance suggestions
                - Troubleshooting
                """)

            with gr.Tab("System Info"):
                gr.Markdown("""
                ## System Information

                ### Core Models
                - **Vision Language Model**: Qwen/Qwen2-VL-7B-Instruct
                - **Vision Encoder**: CLIP ViT-B/32
                - **Vector Database**: ChromaDB
                - **Sparse Retrieval**: BM25 with Chinese text processing
                - **Re-ranking**: BGE ReRanker model

                ### Data Source
                - **Document Type**: ADAS installation manual PDF
                - **Processing Precision**: 300 DPI high-definition rendering
                - **Feature Dimension**: 512-dimensional CLIP features

                ### Features
                - **End-to-end Visual Q&A**
                - **Smart Area Zoom**
                - **Multi-step Reasoning Chain**
                - **Real-time Visual Retrieval**
                - **Hybrid Dense/Sparse Retrieval**
                - **Intelligent Re-ranking**
                """)

            # 事件绑定
            init_btn.click(
                fn=self.initialize_models,
                inputs=[],
                outputs=[status_display]
            )

            submit_btn.click(
                fn=self.process_query,
                inputs=[
                    user_input,
                    image_input,
                    chatbot,
                    use_react_checkbox,
                    use_hybrid_checkbox
                ],
                outputs=[
                    chatbot,
                    retrieved_image,
                    thinking_process,
                    clear_btn
                ]
            ).then(
                lambda: gr.update(value=""), None, user_input
            )

            user_input.submit(
                fn=self.process_query,
                inputs=[
                    user_input,
                    image_input,
                    chatbot,
                    use_react_checkbox,
                    use_hybrid_checkbox
                ],
                outputs=[
                    chatbot,
                    retrieved_image,
                    thinking_process,
                    clear_btn
                ]
            ).then(
                lambda: gr.update(value=""), None, user_input
            )

            clear_btn.click(
                fn=lambda: ([], None, "", gr.update(visible=False)),
                inputs=[],
                outputs=[chatbot, retrieved_image, thinking_process, clear_btn]
            )

        return interface


def launch_demo():
    """启动网页演示"""
    demo = WebDemo()

    # 创建并启动界面
    interface = demo.create_interface()

    print("\n启动ADAS多模态Agent Web界面...")
    print("数据目录: ./data/")
    print("向量数据库: ./data/chroma_db/")
    print("临时图像: ./temp_uploaded/")
    print("\n访问地址: http://localhost:7860")
    print("如需修改端口，请使用 launch(port=xxx)")
    print("按 Ctrl+C 停止服务")

    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        # server_port=7860,
        share=True,  # Set to True for public link
        show_error=True
    )


if __name__ == "__main__":
    launch_demo()
