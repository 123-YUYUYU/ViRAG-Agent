"""
ReAct代理核心逻辑实现
实现多模态Agent的思考与行动循环
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import time

from llm.qwen2_vl_client import Qwen2VLClient
from retrieval.retriever import VisualRetriever
from tools.smart_zoom_tool import SmartZoomProcessor, smart_zoom, extract_zoom_coordinates
from tools.schema import get_tool_description
from llm.prompts import get_prompt, format_react_step, format_visual_retrieval_action, format_smart_zoom_action
from utils.logger import (
    log_agent_thinking, log_agent_action, log_agent_response,
    log_tool_call, log_tool_result, log_visual_retrieval, log_smart_zoom
)
from data_types import DocBlock


def _get_doc_page(metadata: Dict[str, Any]):
    return metadata.get("page_num", metadata.get("page", "Unknown"))


def _get_doc_source(metadata: Dict[str, Any]):
    return metadata.get("source", metadata.get("original_pdf", "Unknown"))


class ReActAgent:
    """
    ReAct（Reasoning and Acting）代理类
    实现基于思维链的多模态推理与行动
    """

    def __init__(
        self,
        qwen_client: Qwen2VLClient,
        visual_retriever: VisualRetriever,
        clip_encoder=None,  # 可选的CLIP编码器，用于检索
        max_iterations: int = 10
    ):
        """
        初始化ReAct代理

        Args:
            qwen_client: Qwen2-VL客户端
            visual_retriever: 视觉检索器
            clip_encoder: CLIP编码器（用于文本检索）
            max_iterations: 最大迭代次数
        """
        self.qwen_client = qwen_client
        self.visual_retriever = visual_retriever
        self.clip_encoder = clip_encoder
        self.max_iterations = max_iterations
        self.zoom_processor = SmartZoomProcessor()

        # 对话历史
        self.conversation_history = []

        print("✅ ReAct代理初始化完成")

    def _extract_action_from_response(self, response: str) -> Optional[Dict]:
        """
        从模型响应中提取行动信息

        Args:
            response: 模型的原始响应

        Returns:
            Optional[Dict]: 行动信息字典或None
        """
        # 尝试解析<thought><action>格式
        action_pattern = r'<action>\s*(\{.*?\})\s*</action>'
        thought_pattern = r'<thought>\s*(.*?)\s*</thought>'

        action_match = re.search(action_pattern, response, re.DOTALL)
        thought_match = re.search(thought_pattern, response, re.DOTALL)

        if action_match:
            try:
                action_json = action_match.group(1)
                action_dict = json.loads(action_json)

                # 添加思考信息
                if thought_match:
                    action_dict['thought'] = thought_match.group(1).strip()

                return action_dict
            except json.JSONDecodeError:
                pass

        # 检查是否有ZOOM指令
        zoom_coords = extract_zoom_coordinates(response)
        if zoom_coords:
            # 模拟smart_zoom工具调用
            return {
                "name": "smart_zoom",
                "arguments": {
                    "image_path": self.last_image_used,  # 需要追踪最后使用的图像
                    "x": zoom_coords[0],
                    "y": zoom_coords[1],
                    "width": zoom_coords[2],
                    "height": zoom_coords[3]
                }
            }

        return None

    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """
        执行指定的工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            str: 工具执行结果
        """
        log_tool_call(tool_name, arguments)

        try:
            if tool_name == "visual_retrieval":
                return self._execute_visual_retrieval(arguments)
            elif tool_name == "smart_zoom":
                return self._execute_smart_zoom(arguments)
            else:
                result = f"错误：未知工具 '{tool_name}'"
                log_tool_result(tool_name, result)
                return result

        except Exception as e:
            error_msg = f"工具执行错误: {str(e)}"
            log_tool_result(tool_name, error_msg)
            return error_msg

    def _execute_visual_retrieval(self, arguments: Dict) -> str:
        """
        执行视觉检索工具

        Args:
            arguments: 检索参数

        Returns:
            str: 检索结果
        """
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 5)
        threshold = arguments.get("threshold", 0.3)

        log_visual_retrieval(query, top_k, 0)  # 暂时记录0，稍后更新

        if not self.clip_encoder:
            result = "错误：CLIP编码器未初始化，无法执行文本检索"
            log_tool_result("visual_retrieval", result)
            return result

        try:
            results = self.visual_retriever.search_by_text(
                query=query,
                text_encoder=self.clip_encoder,
                top_k=top_k,
                threshold=threshold
            )

            log_visual_retrieval(query, top_k, len(results))

            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "image_path": result["image_path"],
                    "similarity": round(result["similarity"], 4),
                    "page": result["metadata"].get("page", "Unknown"),
                    "source": result["metadata"].get("source", "Unknown")
                })

            result_str = json.dumps({
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }, ensure_ascii=False, indent=2)

            log_tool_result("visual_retrieval", f"找到{len(results)}个结果")
            return result_str

        except Exception as e:
            error_msg = f"视觉检索失败: {str(e)}"
            log_tool_result("visual_retrieval", error_msg)
            return error_msg

    def _execute_smart_zoom(self, arguments: Dict) -> str:
        """
        执行智能缩放工具

        Args:
            arguments: 缩放参数

        Returns:
            str: 缩放结果
        """
        image_path = arguments.get("image_path", "")
        x = arguments.get("x", 0.0)
        y = arguments.get("y", 0.0)
        width = arguments.get("width", 0.5)
        height = arguments.get("height", 0.5)
        scale_factor = arguments.get("scale_factor", 2.0)

        log_smart_zoom(image_path, (x, y, width, height), scale_factor)

        try:
            # 使用智能缩放处理器
            zoomed_image, output_path = self.zoom_processor.process_zoom_request(
                image_path=image_path,
                coordinates=(x, y, width, height),
                scale_factor=scale_factor,
                output_dir="./data/zoomed_regions/"
            )

            result = {
                "status": "success",
                "original_image": image_path,
                "zoomed_image": output_path,
                "coordinates": [x, y, width, height],
                "scale_factor": scale_factor
            }

            result_str = json.dumps(result, ensure_ascii=False, indent=2)
            log_tool_result("smart_zoom", f"缩放图像保存至: {output_path}")
            return result_str

        except Exception as e:
            error_msg = f"智能缩放失败: {str(e)}"
            log_tool_result("smart_zoom", error_msg)
            return error_msg

    def chat(
        self,
        user_input: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        执行多轮对话

        Args:
            user_input: 用户输入
            images: 图像列表
            system_prompt: 系统提示词

        Returns:
            str: 代理响应
        """
        start_time = time.time()

        # 设置默认系统提示词
        if system_prompt is None:
            system_prompt = get_prompt("system")

        # 初始化对话历史
        if not self.conversation_history:
            self.conversation_history = [{
                "role": "system",
                "content": system_prompt
            }]

        # 记录用户输入
        user_content = [{"type": "text", "text": user_input}]
        if images:
            for img_path in images:
                user_content.insert(0, {"type": "image", "image": img_path})
                # 记录最后一个图像路径（用于ZOOM指令）
                self.last_image_used = img_path

        self.conversation_history.append({
            "role": "user",
            "content": user_content
        })

        # 执行ReAct循环
        iteration = 0
        current_response = ""
        tool_result = ""

        while iteration < self.max_iterations:
            iteration += 1
            log_agent_thinking(f"ReAct迭代 #{iteration}")

            # 构建当前上下文
            current_context = self._build_context(user_input, tool_result)

            try:
                # 获取模型响应
                response = self.qwen_client.generate_single_response(
                    text=current_context,
                    images=images,
                    system_prompt=system_prompt
                )

                log_agent_response(response[:200] + "..." if len(response) > 200 else response)

                # 检查是否包含工具调用
                action_info = self._extract_action_from_response(response)

                if action_info and "name" in action_info:
                    # 记录思考过程
                    if "thought" in action_info:
                        log_agent_thinking(action_info["thought"])

                    # 执行工具
                    tool_name = action_info["name"]
                    arguments = action_info.get("arguments", {})

                    log_agent_action(f"执行工具: {tool_name}", str(arguments)[:200])

                    tool_result = self._execute_tool(tool_name, arguments)

                    # 将工具结果添加到对话历史
                    self.conversation_history.append({
                        "role": "user",
                        "content": f"工具执行结果:\n{tool_result}"
                    })

                    current_response = tool_result
                else:
                    # 没有更多工具调用，返回最终结果
                    current_response = response
                    log_agent_response(f"迭代#{iteration}结束，返回最终答案")
                    break

            except Exception as e:
                error_msg = f"ReAct循环第{iteration}次迭代出错: {str(e)}"
                log_tool_result("react_loop", error_msg)

                # 添加错误恢复逻辑
                recovery_prompt = get_prompt("error_recovery",
                                           error_message=str(e),
                                           task=user_input)

                current_response = self.qwen_client.generate_single_response(
                    text=recovery_prompt,
                    images=images,
                    system_prompt=system_prompt
                )

                break

        # 记录最终响应到对话历史
        self.conversation_history.append({
            "role": "assistant",
            "content": current_response
        })

        end_time = time.time()
        log_agent_response(f"总耗时: {end_time - start_time:.2f}秒")

        return current_response

    def _build_context(self, initial_query: str, tool_result: str = "") -> str:
        """
        构建当前迭代的上下文

        Args:
            initial_query: 初始查询
            tool_result: 工具执行结果

        Returns:
            str: 构建的上下文
        """
        if tool_result:
            context = f"""之前的查询: {initial_query}
工具执行结果: {tool_result}

请基于以上信息继续处理任务或给出最终答案。"""
        else:
            context = initial_query

        return context

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
        print("对话历史已重置")

    def get_conversation_history(self) -> List[Dict]:
        """
        获取对话历史

        Returns:
            List[Dict]: 对话历史列表
        """
        return self.conversation_history


class SimpleVQAAgent:
    """
    简单的视觉问答代理
    不使用ReAct框架，直接进行问答
    """

    def __init__(self, qwen_client: Qwen2VLClient):
        """
        初始化简单VQA代理

        Args:
            qwen_client: Qwen2-VL客户端
        """
        self.qwen_client = qwen_client

    def answer(
        self,
        question: str,
        images: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        直接回答问题（支持ZOOM指令）

        Args:
            question: 问题
            images: 图像列表
            system_prompt: 系统提示词

        Returns:
            str: 答案
        """
        if system_prompt is None:
            system_prompt = get_prompt("system")

        response = self.qwen_client.generate_single_response(
            text=question,
            images=images,
            system_prompt=system_prompt
        )

        # 检查是否包含ZOOM指令
        zoom_coords = self.qwen_client.check_for_zoom_command(response)

        if zoom_coords:
            print(f"检测到ZOOM指令: {zoom_coords}")

            # 执行智能缩放
            if images:
                try:
                    zoomed_img, zoom_path = smart_zoom(
                        image_path=images[0],  # 使用第一张图
                        x=zoom_coords[0],
                        y=zoom_coords[1],
                        width=zoom_coords[2],
                        height=zoom_coords[3],
                        output_path=f"./data/zoomed_regions/temp_zoom_{int(time.time())}.png"
                    )

                    # 用缩放后的图像重新询问
                    zoom_response = self.qwen_client.generate_single_response(
                        text=question,
                        images=[zoom_path],
                        system_prompt=system_prompt
                    )

                    return f"[已执行智能缩放]\n原响应: {response}\n缩放后响应: {zoom_response}"
                except Exception as e:
                    print(f"智能缩放执行失败: {e}")

        return response


class EvidenceAwareVQAAgent:
    """
    Evidence-aware 多模态问答代理
    基于DocBlock证据进行多模态问答
    """

    def __init__(self, qwen_client: Qwen2VLClient):
        """
        初始化Evidence-aware VQA代理

        Args:
            qwen_client: Qwen2-VL客户端
        """
        self.qwen_client = qwen_client

    def build_multimodal_messages(self, query: str, docs: List[DocBlock]) -> List[Dict]:
        """
        构建多模态消息

        Args:
            query: 查询问题
            docs: 检索到的DocBlock列表

        Returns:
            List[Dict]: 符合Qwen2-VL格式的消息列表
        """
        # 分离不同类型的内容
        images = []
        texts = []

        for doc in docs:
            if doc.block_type == "image":
                images.append({"type": "image", "image": doc.content})
            else:
                # 添加来源信息
                page = _get_doc_page(doc.metadata)
                text_with_source = f"[第{page}页 {doc.block_type.upper()}内容]\n{doc.content}"
                texts.append(text_with_source)

        # 组装证据文本
        evidence_text = "\n".join(texts)

        # 构建系统提示
        system_prompt = f"""你是工业ADAS专家，请严格基于证据回答。
【证据】
{evidence_text}
要求：
1）优先使用图像信息
2）必须引用页码
3）如果证据不足，明确说明"""

        # 构建用户消息
        user_content = [
            {"type": "text", "text": f"{system_prompt}\n\n问题：{query}"}
        ]

        # 添加图像内容
        user_content.extend(images)

        messages = [
            {
                "role": "user",
                "content": user_content
            }
        ]

        print(f"[AGENT_DEBUG] qwen_vl_image_count={len(images)}")
        for idx, image in enumerate(images):
            print(f"[AGENT_DEBUG] image[{idx}] path={image.get('image')}")
        print(f"[AGENT_DEBUG] combined_text_length={len(user_content[0]['text'])}")
        print(f"[AGENT_DEBUG] has_history={bool(getattr(self, 'conversation_history', []))}")
        print(
            f"[AGENT_DEBUG] has_ocr={any(bool(doc.metadata.get('ocr_text')) for doc in docs)} "
            f"has_metadata={any(bool(doc.metadata) for doc in docs)} "
            f"has_evidence_text={bool(evidence_text)}"
        )

        return messages

    def answer_with_evidence(
        self,
        query: str,
        docs: List[DocBlock],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        基于证据的回答

        Args:
            query: 查询问题
            docs: 检索到的DocBlock列表
            system_prompt: 系统提示词

        Returns:
            str: 答案
        """
        # 构建多模态消息
        messages = self.build_multimodal_messages(query, docs)

        # 发送消息给模型
        response = self.qwen_client.chat(messages)

        return response

    def run(self, query: str, docs: List[DocBlock]) -> Dict:
        """
        运行完整的Evidence-aware问答流程

        Args:
            query: 查询问题
            docs: 检索到的DocBlock列表

        Returns:
            Dict: 包含答案和来源信息的结果
        """
        # 获取答案
        answer = self.answer_with_evidence(query, docs)

        # 构建来源信息
        sources = [
            {
                "page": _get_doc_page(d.metadata),
                "page_num": _get_doc_page(d.metadata),
                "type": d.block_type,
                "source": _get_doc_source(d.metadata)
            } for d in docs
        ]

        return {
            "answer": answer,
            "sources": sources
        }


if __name__ == "__main__":
    # 测试代理初始化
    print("测试ReAct代理初始化...")

    # 这里只是一个演示，实际使用需要先初始化各组件
    print("代理模块加载完成")
    print("注意：完整功能需要启动时初始化Qwen2-VL模型、CLIP编码器和向量数据库")
