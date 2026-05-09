"""
对话上下文管理模块
负责管理Agent的对话历史和上下文记忆
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import uuid


class ConversationMemory:
    """
    对话记忆管理类
    负责存储和管理对话历史、上下文和状态
    """

    def __init__(self, max_history_length: int = 50, auto_save: bool = True):
        """
        初始化对话记忆管理器

        Args:
            max_history_length: 最大历史长度
            auto_save: 是否自动保存
        """
        self.max_history_length = max_history_length
        self.auto_save = auto_save
        self.session_id = str(uuid.uuid4())

        # 当前对话历史
        self.history: List[Dict[str, Any]] = []

        # 会话元数据
        self.session_metadata = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "turn_count": 0,
            "topic_summary": "",
            "relevant_docs": []
        }

        # 确保数据目录存在
        self.memory_dir = Path("./data/memory")
        self.memory_dir.mkdir(exist_ok=True)

        print(f"对话记忆管理器初始化 (会话: {self.session_id[:8]})")

    def add_message(self, role: str, content: Any, timestamp: Optional[float] = None):
        """
        向对话历史添加消息

        Args:
            role: 角色 ('user', 'assistant', 'system', 'tool')
            content: 消息内容
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "formatted_time": datetime.fromtimestamp(timestamp).isoformat()
        }

        self.history.append(message)
        self.session_metadata["last_accessed"] = datetime.now().isoformat()
        self.session_metadata["turn_count"] += 1

        # 限制历史长度
        if len(self.history) > self.max_history_length:
            # 保留最新消息，但也保留系统消息
            system_msgs = [msg for msg in self.history if msg["role"] == "system"]
            other_msgs = [msg for msg in self.history if msg["role"] != "system"]

            # 保留所有系统消息和最新的max_history_length-other_msgs个非系统消息
            if len(other_msgs) > self.max_history_length - len(system_msgs):
                other_msgs = other_msgs[-(self.max_history_length - len(system_msgs)):]

            self.history = system_msgs + other_msgs

        if self.auto_save:
            self.save_session()

    def add_user_message(self, content: Any):
        """Add user message"""
        self.add_message("user", content)

    def add_assistant_message(self, content: Any):
        """Add assistant message"""
        self.add_message("assistant", content)

    def add_tool_message(self, content: Any):
        """Add tool message"""
        self.add_message("tool", content)

    def add_system_message(self, content: Any):
        """Add system message"""
        self.add_message("system", content)

    def get_recent_history(self, num_turns: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的对话历史

        Args:
            num_turns: 要返回的消息回合数

        Returns:
            List[Dict[str, Any]]: 最近的消息历史
        """
        return self.history[-num_turns:]

    def get_full_history(self) -> List[Dict[str, Any]]:
        """
        获取完整的对话历史

        Returns:
            List[Dict[str, Any]]: 完整的消息历史
        """
        return self.history.copy()

    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        self.session_metadata["turn_count"] = 0
        print("对话历史已清除")

    def search_history(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        在历史中搜索包含关键字的消息

        Args:
            query: 搜索关键词
            max_results: 最大结果数

        Returns:
            List[Dict[str, Any]]: 匹配的消息列表
        """
        results = []
        query_lower = query.lower()

        for message in self.history:
            content = message["content"]

            # 为搜索转换内容为字符串
            if isinstance(content, dict) or isinstance(content, list):
                content_str = json.dumps(content, ensure_ascii=False)
            else:
                content_str = str(content)

            if query_lower in content_str.lower():
                results.append(message)

                if len(results) >= max_results:
                    break

        return results

    def summarize_topic(self) -> str:
        """
        总结当前对话主题

        Returns:
            str: 对话主题摘要
        """
        if not self.history:
            return "New conversation"

        # 提取用户和助手消息
        user_msgs = [msg for msg in self.history if msg["role"] == "user"]
        assistant_msgs = [msg for msg in self.history if msg["role"] == "assistant"]

        # 简单启发式摘要
        if user_msgs:
            # 使用第一条用户消息的前几个词作为主题
            first_user_msg = user_msgs[0]["content"]
            if isinstance(first_user_msg, str):
                topic = first_user_msg.strip()[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
            else:
                topic = "Multimodal conversation"
        else:
            topic = "System conversation"

        self.session_metadata["topic_summary"] = topic
        return topic

    def get_context_window(self, max_tokens: int = 4000) -> List[Dict[str, Any]]:
        """
        获取适合模型上下文窗口的消息历史

        Args:
            max_tokens: 最大令牌数（估算）

        Returns:
            List[Dict[str, Any]]: 适合上下文的消息
        """
        # 简单的令牌估算：大约每4个字符1个令牌
        estimated_max_chars = max_tokens * 4

        # 从末尾收集消息直到达到令牌限制
        context = []
        total_chars = 0

        # 优先保留系统消息
        system_msgs = [msg for msg in self.history if msg["role"] == "system"]
        for msg in system_msgs:
            msg_content = str(msg["content"])
            total_chars += len(msg_content)
            context.insert(0, msg)  # Put system messages at the beginning

        # 从末尾添加其他消息
        for msg in reversed(self.history):
            if msg["role"] == "system":
                continue  # Already processed

            msg_content = str(msg["content"]) if not isinstance(msg["content"], (dict, list)) else json.dumps(msg["content"], ensure_ascii=False)
            msg_chars = len(msg_content)

            if total_chars + msg_chars > estimated_max_chars:
                break

            context.insert(0, msg)
            total_chars += msg_chars

        return context

    def save_session(self, session_name: Optional[str] = None):
        """
        保存当前会话

        Args:
            session_name: 会话名称（可选）
        """
        if session_name is None:
            session_name = f"session_{self.session_id[:8]}_{int(time.time())}"

        session_data = {
            "metadata": self.session_metadata,
            "history": self.history
        }

        session_file = self.memory_dir / f"{session_name}.json"

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        print(f"会话保存至: {session_file}")

    def load_session(self, session_name: str) -> bool:
        """
        加载会话

        Args:
            session_name: 会话名称

        Returns:
            bool: 加载是否成功
        """
        session_file = self.memory_dir / f"{session_name}.json"

        if not session_file.exists():
            print(f"会话文件不存在: {session_file}")
            return False

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            self.session_metadata = session_data["metadata"]
            self.history = session_data["history"]
            self.session_id = self.session_metadata["session_id"]

            print(f"会话已加载: {session_file}")
            return True

        except Exception as e:
            print(f"加载会话失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取会话统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        total_messages = len(self.history)
        user_messages = len([m for m in self.history if m["role"] == "user"])
        assistant_messages = len([m for m in self.history if m["role"] == "assistant"])
        tool_messages = len([m for m in self.history if m["role"] == "tool"])

        return {
            "session_id": self.session_id,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "tool_messages": tool_messages,
            "turn_count": self.session_metadata["turn_count"],
            "session_duration": time.time() - self._get_session_start_time(),
            "topic_summary": self.summarize_topic()
        }

    def _get_session_start_time(self) -> float:
        """获取会话开始时间戳"""
        if self.history:
            return self.history[0].get("timestamp", time.time())
        return time.time()

    def export_chatlog(self, filename: str = None) -> str:
        """
        导出聊天记录

        Args:
            filename: 输出文件名

        Returns:
            str: 导出的文件路径
        """
        if filename is None:
            filename = f"chatlog_{self.session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        filepath = self.memory_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"ADAS Multimodal Agent Chat Log\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Created: {self.session_metadata['created_at']}\n")
            f.write(f"Topic Summary: {self.session_metadata['topic_summary']}\n")
            f.write("-" * 60 + "\n\n")

            for msg in self.history:
                role = msg["role"].upper()
                timestamp = datetime.fromtimestamp(msg["timestamp"]).strftime("%H:%M:%S")

                f.write(f"[{timestamp}] {role}: ")

                content = msg["content"]
                if isinstance(content, (dict, list)):
                    f.write(json.dumps(content, ensure_ascii=False, indent=2))
                else:
                    f.write(str(content))

                f.write("\n\n")

        print(f"聊天记录导出至: {filepath}")
        return str(filepath)


class ShortTermMemory:
    """
    短期记忆类
    用于在单次对话中临时存储重要信息
    """

    def __init__(self):
        self.temp_storage = {}

    def store(self, key: str, value: Any, ttl: int = 300):
        """
        存储临时信息

        Args:
            key: 键名
            value: 值
            ttl: 生存时间（秒）
        """
        import time
        self.temp_storage[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl
        }

    def retrieve(self, key: str, default=None) -> Any:
        """
        检索临时信息

        Args:
            key: 键名
            default: 默认值

        Returns:
            Any: 存储的值或默认值
        """
        import time
        if key in self.temp_storage:
            item = self.temp_storage[key]
            # 检查是否过期
            if time.time() - item["timestamp"] < item["ttl"]:
                return item["value"]
            else:
                # 如果过期则删除
                del self.temp_storage[key]

        return default

    def clear_expired(self):
        """清除过期项目"""
        import time
        current_time = time.time()
        expired_keys = [
            key for key, item in self.temp_storage.items()
            if time.time() - item["timestamp"] >= item["ttl"]
        ]

        for key in expired_keys:
            del self.temp_storage[key]

    def clear_all(self):
        """清除所有临时存储"""
        self.temp_storage.clear()


# 全局记忆实例
global_memory = ConversationMemory()


def get_global_memory() -> ConversationMemory:
    """
    获取全局对话记忆实例

    Returns:
        ConversationMemory: 全局记忆实例
    """
    return global_memory


if __name__ == "__main__":
    # 测试对话记忆功能
    print("测试对话记忆...")

    memory = ConversationMemory(max_history_length=10)

    # 添加一些测试消息
    memory.add_user_message("Hello, please help me analyze this circuit diagram")
    memory.add_assistant_message("Sure, I see the circuit diagram, analyzing now...")
    memory.add_tool_message({"tool": "visual_retrieval", "result": "Found relevant page"})

    # 测试不同功能
    print(f"历史长度: {len(memory.get_full_history())}")
    print(f"最近2条消息: {memory.get_recent_history(2)}")
    print(f"会话统计: {memory.get_statistics()}")

    # 搜索测试
    search_results = memory.search_history("circuit")
    print(f"搜索结果数量: {len(search_results)}")

    # 保存和加载测试
    memory.save_session("test_session")

    new_memory = ConversationMemory()
    new_memory.load_session("test_session")

    print(f"加载的历史长度: {len(new_memory.get_full_history())}")