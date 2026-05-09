from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class DocBlock:
    id: str
    content: str  # 文本内容 or 图像绝对路径
    block_type: str  # "text", "image", "table"
    metadata: Dict[str, Any] = field(default_factory=dict)