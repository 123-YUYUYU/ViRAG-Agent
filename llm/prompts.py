"""
提示词管理模块
存储系统提示词和ReAct指令模板
"""

# 系统提示词
SYSTEM_PROMPT = """你是一名高级车辆电工与 ADAS 专家。请仔细观察提供的图纸（注意区分线束颜色、引脚编号、拓扑关系和空间参数）来回答问题。

重要提醒：
1. 如果图像中的文字或细节不够清晰，请使用 [ZOOM: x, y, w, h] 格式指定需要放大的区域
2. 坐标使用归一化值（0-1之间），x,y为左上角坐标，w,h为宽度和高度
3. 回答问题时请结合图纸中的具体信息，确保准确性"""

# ReAct提示词模板
REACT_AGENT_TEMPLATE = """你是一个基于ReAct框架的智能助手，专门用于分析工业车辆ADAS系统的安装手册。

## 工作流程
1. **思考 (THOUGHT)**: 分析当前情况和需要采取的行动
2. **行动 (ACTION)**: 执行下一步操作
3. **观察 (OBSERVATION)**: 记录执行结果
4. 重复直到任务完成

## 可用工具
- visual_retrieval: 根据文本查询搜索相关图纸页面
- smart_zoom: 放大图像中的特定区域以查看细节

## 工具使用格式
使用以下JSON格式调用工具：
```
<thought>
你的思考过程...
</thought>

<action>
{
    "name": "工具名",
    "arguments": {
        "参数名": "参数值"
    }
}
</action>
```

## 输出格式
- 当需要调用工具时：使用上述<thought><action>格式
- 当回答最终问题时：直接输出答案

## 特殊规则
- 如果图像中的细节不清晰，使用 [ZOOM: x, y, w, h] 格式指定区域
- 确保所有坐标值都在0-1范围内
- 结合视觉信息给出专业建议

当前任务：{task}

{chat_history}

助手："""

# 简化版提示词模板
SIMPLE_CHAT_TEMPLATE = """{system_prompt}

用户: {user_input}

助手: """

# 工具提示词模板
TOOLS_INSTRUCTION = """## 工具使用说明

### 1. 视觉检索工具 (visual_retrieval)
用途：根据文本描述搜索相关的图纸页面
使用方式：
<thought>
我需要查找关于"{query}"的相关图纸。
</thought>

<action>
{
    "name": "visual_retrieval",
    "arguments": {
        "query": "{query}",
        "top_k": {top_k}
    }
}
</action>

### 2. 智能缩放工具 (smart_zoom)
用途：放大图像中的特定区域以查看细节
使用方式：
<thought>
图像中的{detail}部分不够清晰，需要放大查看。
</thought>

<action>
{
    "name": "smart_zoom",
    "arguments": {
        "image_path": "{image_path}",
        "x": {x},
        "y": {y},
        "width": {width},
        "height": {height}
    }
}
</action>

注意：当图片中的文字或细节不清晰时，也可以直接使用 [ZOOM: x, y, w, h] 格式。"""

# 专家角色提示词
EXPERT_ROLE_PROMPT = """你是工业车辆ADAS系统的高级专家，具备以下能力：
1. 电气工程专业知识（特别是车辆电气系统）
2. ADAS系统安装与维护经验
3. 图纸阅读与解析能力
4. 故障诊断技能

在分析图纸时，请特别关注：
- 线束颜色与连接方式
- 电气元件标识与规格
- 安装位置与空间关系
- 安全注意事项与标准规范

回答问题时请体现专业性，并基于图纸中的实际信息。"""

# 视觉问答提示词
VISUAL_QA_TEMPLATE = """请仔细观察以下图片并回答问题：

图片内容：{image_description}
用户问题：{question}

回答要求：
1. 基于图片内容给出准确答案
2. 如果图片细节不清晰，使用 [ZOOM: x, y, w, h] 格式指出需要放大的区域
3. 体现ADAS专家的专业知识
4. 确保答案的准确性和实用性

答案："""

# 错误恢复提示词
ERROR_RECOVERY_PROMPT = """之前的尝试遇到了问题：{error_message}

请重新分析情况并尝试不同的方法。可能的解决方向：
1. 检查是否需要更多的上下文信息
2. 考虑使用不同的工具组合
3. 重新表述查询条件
4. 分解复杂问题为更简单的子问题

当前任务：{task}"""

# 上下文管理提示词
CONTEXT_MANAGEMENT_PROMPT = """## 对话上下文管理指南

### 上下文保持
- 保持对之前对话轮次的记忆
- 维持话题的连贯性
- 避免重复询问已知信息

### 上下文更新
- 当用户提供新信息时及时更新理解
- 修正之前可能的误解
- 基于最新信息调整回答策略

### 上下文清理
- 适时忘记过时的信息
- 专注于当前任务相关的内容
- 保持上下文的精简高效

当前对话上下文：
{context_history}"""


def get_prompt(template_name: str, **kwargs) -> str:
    """
    根据模板名称获取格式化后的提示词

    Args:
        template_name: 模板名称
        **kwargs: 用于格式化的参数

    Returns:
        str: 格式化后的提示词
    """
    templates = {
        "system": SYSTEM_PROMPT,
        "react_agent": REACT_AGENT_TEMPLATE,
        "simple_chat": SIMPLE_CHAT_TEMPLATE,
        "tools_instruction": TOOLS_INSTRUCTION,
        "expert_role": EXPERT_ROLE_PROMPT,
        "visual_qa": VISUAL_QA_TEMPLATE,
        "error_recovery": ERROR_RECOVERY_PROMPT,
        "context_management": CONTEXT_MANAGEMENT_PROMPT
    }

    template = templates.get(template_name)
    if template:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"缺少必需的参数: {e}")
    else:
        raise ValueError(f"未知的模板名称: {template_name}")


def format_react_step(thought: str, action: dict = None) -> str:
    """
    格式化ReAct步骤

    Args:
        thought: 思考内容
        action: 行动字典

    Returns:
        str: 格式化的ReAct步骤
    """
    result = f"<thought>\n{thought}\n</thought>\n\n"

    if action:
        import json
        result += f"<action>\n{json.dumps(action, indent=4, ensure_ascii=False)}\n</action>"

    return result


def format_visual_retrieval_action(query: str, top_k: int = 5) -> dict:
    """
    格式化视觉检索行动

    Args:
        query: 搜索查询
        top_k: 返回结果数

    Returns:
        dict: 格式化的行动
    """
    return {
        "name": "visual_retrieval",
        "arguments": {
            "query": query,
            "top_k": top_k
        }
    }


def format_smart_zoom_action(
    image_path: str,
    x: float, y: float,
    width: float, height: float
) -> dict:
    """
    格式化智能缩放行动

    Args:
        image_path: 图像路径
        x, y: 坐标
        width, height: 宽高

    Returns:
        dict: 格式化的行动
    """
    return {
        "name": "smart_zoom",
        "arguments": {
            "image_path": image_path,
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
    }


# 预定义的常用提示词变体
PROMPT_VARIANTS = {
    "detailed_analysis": SYSTEM_PROMPT + "\n\n请进行详细的技术分析，包括具体的数值、规格和安装要求。",
    "safety_focused": SYSTEM_PROMPT + "\n\n在回答时请特别强调安全注意事项和操作规范。",
    "troubleshooting": SYSTEM_PROMPT + "\n\n请从故障排除的角度分析问题，提供排查步骤和解决方案。",
    "installation_guide": SYSTEM_PROMPT + "\n\n请以安装指导的方式回答，提供详细的步骤和要点。",
    "component_identification": SYSTEM_PROMPT + "\n\n请重点识别图像中的各个组件及其功能。"
}


def get_expert_prompt(variant: str = "standard") -> str:
    """
    获取专家角色的特定变体提示词

    Args:
        variant: 变体类型

    Returns:
        str: 专家提示词
    """
    if variant == "standard":
        return SYSTEM_PROMPT
    else:
        return PROMPT_VARIANTS.get(variant, SYSTEM_PROMPT)


if __name__ == "__main__":
    # 测试提示词功能
    print("=== 系统提示词 ===")
    print(SYSTEM_PROMPT)

    print("\n=== ReAct模板示例 ===")
    sample_task = "分析继电器接线图并解释工作原理"
    sample_history = "用户: 请解释这个继电器的工作原理\n助手: [等待分析图片]"
    print(get_prompt("react_agent", task=sample_task, chat_history=sample_history))

    print("\n=== 专家角色提示词 ===")
    print(EXPERT_ROLE_PROMPT)

    print("\n=== ReAct步骤格式化示例 ===")
    action_example = format_visual_retrieval_action("继电器接线图", 3)
    print(format_react_step("我需要查找继电器接线图", action_example))