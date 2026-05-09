"""
全局配置文件
存储项目中的各种路径、模型名称和超参数
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Qwen2-VL 本地推理图像像素范围
QWEN_VL_MIN_PIXELS = 256 * 28 * 28
QWEN_VL_MAX_PIXELS = 768 * 28 * 28

# LLM 推理 backend
# 可选值：
# - "local": 使用本地 Qwen2-VL 模型推理，适合有 GPU / 本地模型环境
# - "qwen_api": 使用 DashScope Qwen API 推理，适合降低本地显存压力
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"
MANUAL_DIR = DATA_DIR / "manual"  # 原始PDF手册目录
PAGE_IMAGES_DIR = DATA_DIR / "page_images"  # 渲染后的图片目录
CHROMA_DB_DIR = DATA_DIR / "chroma_db"  # ChromaDB持久化目录

# 模型配置
CLIP_MODEL_NAME = "/root/autodl-tmp/hf_cache/clip-ViT-B-32/0_CLIPModel"  # 用于视觉特征提取的 CLIP 模型
QWEN_VL_MODEL_NAME = "/root/autodl-tmp/hf_cache/qwen/Qwen2-VL-7B-Instruct"  # local backend 使用的本地 Qwen2-VL 模型路径或 HuggingFace 名称

# qwen_api backend 配置
QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"  # API Key 环境变量名；只从环境变量或 .env 读取
QWEN_API_BASE_URL = os.getenv(
    "QWEN_API_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)  # DashScope OpenAI-compatible endpoint，可通过 QWEN_API_BASE_URL 覆盖
QWEN_API_MODEL = os.getenv(
    "QWEN_API_MODEL",
    "qwen-vl-plus"
)  # qwen_api backend 模型名；常用可选值如 "qwen-vl-plus"、"qwen-vl-max"
QWEN_API_TIMEOUT = int(
    os.getenv("QWEN_API_TIMEOUT", "120")
)  # qwen_api 请求超时时间，单位秒
QWEN_API_MAX_TOKENS = int(
    os.getenv("QWEN_API_MAX_TOKENS", "1024")
)  # qwen_api 单次回答最大输出 token 数
QWEN_API_TEMPERATURE = float(
    os.getenv("QWEN_API_TEMPERATURE", "0.1")
)  # qwen_api 采样温度；评估/技术问答建议较低，如 0.1

# Evaluation Judge 模式
# 可选值：
# - "local_llm": 使用当前 VLM client 作为本地/接口 Judge，失败时自动 fallback 到 rule
# - "rule": 只使用本地规则评估，不调用模型 Judge
EVALUATION_JUDGE_MODE = os.getenv("EVALUATION_JUDGE_MODE", "local_llm")

# 渲染参数
PDF_RENDER_DPI = 300  # PDF渲染DPI，用于生成高清图片

# 向量数据库参数
CHROMA_COLLECTION_NAME = "adas_manual_visual_index"  # ChromaDB集合名称
VECTOR_SEARCH_TOP_K = 5  # 向量检索返回的最相似结果数

# 智能缩放参数
SMART_ZOOM_SCALE_FACTOR = 2.0  # 智能缩放的放大倍数
SMART_ZOOM_MIN_SIZE = 64  # 最小缩放区域尺寸（像素）

# 系统提示词配置
SYSTEM_PROMPT = """你是一名高级车辆电工与 ADAS 专家。请仔细观察提供的图纸（注意区分线束颜色、引脚编号、拓扑关系和空间参数）来回答问题。"""

# 日志配置
LOG_LEVEL = "INFO"  # 日志级别
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

# 创建必要的目录
for dir_path in [DATA_DIR, MANUAL_DIR, PAGE_IMAGES_DIR, CHROMA_DB_DIR]:
    dir_path.mkdir(exist_ok=True)
