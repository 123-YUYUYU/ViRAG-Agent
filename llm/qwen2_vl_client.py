"""
Qwen2-VL模型客户端封装
负责加载模型、处理多模态输入和生成响应
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Union, Optional
import warnings
from PIL import Image
import base64
from io import BytesIO
import re

try:
    from config import QWEN_VL_MIN_PIXELS, QWEN_VL_MAX_PIXELS
except ImportError:
    QWEN_VL_MIN_PIXELS = 256 * 28 * 28
    QWEN_VL_MAX_PIXELS = 768 * 28 * 28


def _cuda_memory_snapshot() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    return (
        f"allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()}"
    )


def _shape_of(value):
    return tuple(value.shape) if hasattr(value, "shape") else None


class Qwen2VLClient:
    """
    Qwen2-VL模型客户端封装类
    提供模型加载、推理和多模态输入处理功能
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 1024
    ):
        """
        初始化Qwen2-VL客户端

        Args:
            model_name: 模型名称
            device: 设备名称，如果为None则自动选择
            torch_dtype: 模型精度
            max_new_tokens: 生成的最大token数
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens

        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"正在初始化Qwen2-VL模型: {model_name}")
        print(f"设备: {self.device}, 精度: {torch_dtype}")

        # 使用 AutoProcessor 替代 AutoTokenizer
        print(
            "[VLM_DEBUG] processor min_pixels="
            f"{QWEN_VL_MIN_PIXELS}, max_pixels={QWEN_VL_MAX_PIXELS}"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=QWEN_VL_MIN_PIXELS,
            max_pixels=QWEN_VL_MAX_PIXELS
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",  # 自动分配到可用设备
            trust_remote_code=True
        )

        # 设置为评估模式
        self.model.eval()

        print(f"✅ Qwen2-VL模型加载完成")

    def prepare_multimodal_input(
        self,
        text: str,
        images: List[Union[str, Image.Image]]
    ) -> List[Dict]:
        """
        准备多模态输入格式

        Args:
            text: 文本输入
            images: 图像路径或PIL Image对象列表

        Returns:
            List[Dict]: 格式化的多模态输入
        """
        inputs = []

        # 添加图像
        for img in images:
            if isinstance(img, str):
                # 如果是字符串，认为是文件路径
                inputs.append({
                    "type": "image",
                    "image": img
                })
            elif isinstance(img, Image.Image):
                # 如果是PIL Image，转换为base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                inputs.append({
                    "type": "image",
                    "image": f"data:image/png;base64,{img_str}"
                })

        # 添加文本
        inputs.append({
            "type": "text",
            "text": text
        })

        return inputs

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        top_p: float = 0.8,
        do_sample: bool = True
    ) -> str:
        """
        执行对话生成

        Args:
            messages: 对话消息列表，格式为[{"role": "...", "content": "..."}]
            temperature: 温度参数
            top_p: Top-P采样参数
            do_sample: 是否采样

        Returns:
            str: 生成的回复
        """
        # 直接使用 messages 作为 processor 的输入，确保格式正确
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 提取所有图像用于处理
        images = []
        image_debug_sources = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_source = item.get("image")
                        if isinstance(image_source, str):
                            # 如果是文件路径，加载图像
                            if image_source.startswith("data:image"):
                                # Base64图像数据
                                header, encoded = image_source.split(",", 1)
                                decoded = base64.b64decode(encoded)
                                img = Image.open(BytesIO(decoded))
                                images.append(img)
                                image_debug_sources.append("<base64_image>")
                            else:
                                # 文件路径
                                img = Image.open(image_source)
                                images.append(img)
                                image_debug_sources.append(image_source)

        print(f"[VLM_DEBUG] before processor cuda_memory {_cuda_memory_snapshot()}")
        print(f"[VLM_DEBUG] images_count={len(images)}")
        for idx, img in enumerate(images):
            source = image_debug_sources[idx] if idx < len(image_debug_sources) else "<unknown>"
            print(
                f"[VLM_DEBUG] image[{idx}] path={source} "
                f"width={img.width} height={img.height} mode={img.mode}"
            )

        # 使用processor处理文本和图像
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        )

        print(f"[VLM_DEBUG] after processor cuda_memory {_cuda_memory_snapshot()}")
        print(f"[VLM_DEBUG] input_ids.shape={_shape_of(inputs.get('input_ids'))}")
        print(f"[VLM_DEBUG] pixel_values.shape={_shape_of(inputs.get('pixel_values'))}")
        print(f"[VLM_DEBUG] image_grid_thw={inputs.get('image_grid_thw')}")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 生成配置
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "repetition_penalty": 1.1
        }

        print("正在生成响应...")

        with torch.no_grad():
            print(f"[VLM_DEBUG] before generate cuda_memory {_cuda_memory_snapshot()}")
            # 生成输出
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
            print(f"[VLM_DEBUG] after generate cuda_memory {_cuda_memory_snapshot()}")

        # 解码输出 - 使用 processor 而不是 tokenizer
        output_text = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]

        return output_text.strip()

    def generate_single_response(
        self,
        text: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        生成单轮回复

        Args:
            text: 用户输入文本
            images: 输入图像列表
            system_prompt: 系统提示词

        Returns:
            str: 模型生成的回复
        """
        # 构造消息历史
        messages = []

        # 添加系统提示词
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 准备用户输入
        user_content = [{"type": "text", "text": text}]

        # 添加图像
        if images:
            for img in images:
                if isinstance(img, str):
                    user_content.insert(0, {"type": "image", "image": img})
                elif isinstance(img, Image.Image):
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    user_content.insert(0, {
                        "type": "image",
                        "image": f"data:image/png;base64,{img_str}"
                    })

        messages.append({
            "role": "user",
            "content": user_content
        })

        # 生成回复
        response = self.chat(messages)
        return response

    def check_for_zoom_command(self, text: str) -> Optional[tuple]:
        """
        检查文本中是否包含缩放指令

        Args:
            text: 要检查的文本

        Returns:
            Optional[tuple]: (x, y, w, h) 或 None
        """
        # 正则表达式匹配 [ZOOM: x, y, w, h] 格式
        pattern = r'\[ZOOM:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
        match = re.search(pattern, text)

        if match:
            try:
                coords = tuple(map(float, match.groups()))
                # 验证坐标在合理范围内
                x, y, w, h = coords
                if all(0 <= val <= 1 for val in [x, y, w, h]) and x + w <= 1 and y + h <= 1:
                    return coords
            except (ValueError, IndexError):
                pass

        return None

    def get_model_info(self) -> Dict:
        """
        获取模型基本信息

        Returns:
            Dict: 模型信息
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "max_new_tokens": self.max_new_tokens,
            "has_gpu": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


# 全局模型实例（可选，根据需要决定是否使用单例）
_qwen2_vl_instance = None


def get_qwen2_vl_client(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    device: Optional[str] = None
) -> Qwen2VLClient:
    """
    获取Qwen2-VL客户端实例（单例模式）

    Args:
        model_name: 模型名称
        device: 设备名称

    Returns:
        Qwen2VLClient: 模型客户端实例
    """
    global _qwen2_vl_instance
    if _qwen2_vl_instance is None:
        _qwen2_vl_instance = Qwen2VLClient(model_name=model_name, device=device)
    return _qwen2_vl_instance


if __name__ == "__main__":
    # 示例用法
    print("Qwen2-VL Client模块测试")

    # 检查CUDA可用性
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")

    # 初始化客户端（仅做演示，实际使用可能需要更长时间）
    try:
        # 为了测试目的，使用较低配置
        client = Qwen2VLClient(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            max_new_tokens=512
        )

        info = client.get_model_info()
        print(f"模型信息: {info}")

        # 测试缩放命令检测
        test_text = "图像细节不够清楚，需要放大查看 [ZOOM: 0.1, 0.2, 0.3, 0.4]"
        zoom_coords = client.check_for_zoom_command(test_text)
        print(f"缩放坐标检测: {zoom_coords}")

    except Exception as e:
        print(f"初始化测试失败: {e}")
        print("注意: 实际运行时需要下载模型文件")
