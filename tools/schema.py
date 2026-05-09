"""
工具模式定义
定义Agent可调用工具的参数规范和描述
"""

from typing import Dict, Any, List


def get_smart_zoom_tool_schema() -> Dict[str, Any]:
    """
    获取智能缩放工具的JSON Schema定义

    Returns:
        Dict[str, Any]: 工具的完整模式定义
    """
    return {
        "name": "smart_zoom",
        "description": "智能图像区域放大工具，用于放大图纸中的特定区域以查看细节",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "原始图像的完整路径"
                },
                "x": {
                    "type": "number",
                    "description": "起始点X坐标（归一化，0-1之间）",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "y": {
                    "type": "number",
                    "description": "起始点Y坐标（归一化，0-1之间）",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "width": {
                    "type": "number",
                    "description": "区域宽度（归一化，0-1之间）",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "height": {
                    "type": "number",
                    "description": "区域高度（归一化，0-1之间）",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "scale_factor": {
                    "type": "number",
                    "description": "放大倍数，默认为2.0",
                    "minimum": 1.0,
                    "default": 2.0
                },
                "output_path": {
                    "type": "string",
                    "description": "输出图像路径，可选"
                }
            },
            "required": ["image_path", "x", "y", "width", "height"]
        }
    }


def get_visual_retrieval_tool_schema() -> Dict[str, Any]:
    """
    获取视觉检索工具的JSON Schema定义

    Returns:
        Dict[str, Any]: 工具的完整模式定义
    """
    return {
        "name": "visual_retrieval",
        "description": "基于视觉特征的文档检索工具，可以根据文本查询找到相关的图纸页面",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的文本查询"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回最相似的结果数量，默认为5",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "threshold": {
                    "type": "number",
                    "description": "相似度过滤阈值，默认为0.3",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3
                }
            },
            "required": ["query"]
        }
    }


def get_all_tools_schemas() -> List[Dict[str, Any]]:
    """
    获取所有工具的模式定义列表

    Returns:
        List[Dict[str, Any]]: 所有工具的模式定义列表
    """
    return [
        get_smart_zoom_tool_schema(),
        get_visual_retrieval_tool_schema()
    ]


# 工具描述映射
TOOL_DESCRIPTIONS = {
    "smart_zoom": (
        "当模型认为图像中的某个区域细节不够清晰时，可以使用此工具来放大特定区域。"
        "使用格式: [ZOOM: x, y, w, h] 其中x,y是起始坐标，w,h是宽高，均为0-1之间的归一化值。"
    ),
    "visual_retrieval": (
        "当需要查找相关信息时，使用此工具根据文本描述搜索相关的图纸页面。"
        "可以帮助找到与当前问题相关的手册页面。"
    )
}


def get_tool_description(tool_name: str) -> str:
    """
    获取指定工具的描述

    Args:
        tool_name: 工具名称

    Returns:
        str: 工具描述
    """
    return TOOL_DESCRIPTIONS.get(tool_name, f"未知工具: {tool_name}")


# 专门用于提示词的工具说明
def get_tools_prompt_instruction() -> str:
    """
    获取用于模型提示词的工具使用说明

    Returns:
        str: 工具使用说明文本
    """
    instruction = """
## 工具使用说明

你可以使用以下工具来帮助回答问题：

### 1. 智能缩放工具 (smart_zoom)
当你发现图像中的文字或细节不够清晰时，可以使用此工具放大特定区域：
- 格式: [ZOOM: x, y, w, h]
- x, y: 区域左上角坐标 (0-1的归一化值)
- w, h: 区域宽高 (0-1的归一化值)
- 注意：确保所选区域完全包含在图像内 (x+w ≤ 1, y+h ≤ 1)

### 2. 视觉检索工具 (visual_retrieval)
当需要查找其他相关信息时，可以通过文本搜索找到相关的图纸页面。

使用这些工具可以提高回答的准确性。
"""
    return instruction


if __name__ == "__main__":
    # 打印所有工具的模式定义
    print("=== 所有工具的JSON Schema ===")
    for tool_schema in get_all_tools_schemas():
        print(f"\n工具名称: {tool_schema['name']}")
        print(f"描述: {tool_schema['description']}")
        print("参数:")
        for param_name, param_info in tool_schema['parameters']['properties'].items():
            print(f"  {param_name}: {param_info}")