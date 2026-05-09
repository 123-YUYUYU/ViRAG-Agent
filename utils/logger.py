"""
统一日志工具
提供结构化日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerSetup:
    """
    日志设置类
    提供灵活的日志配置
    """

    def __init__(self, name: str = "adas_agent", level: str = "INFO"):
        """
        初始化日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # 防止重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 文件处理器
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        # 今天的日志文件
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"adas_agent_{today}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """获取日志记录器实例"""
        return self.logger


# 全局日志记录器实例
logger = LoggerSetup().get_logger()


def log_info(message: str):
    """记录信息级别消息"""
    logger.info(message)


def log_debug(message: str):
    """记录调试级别消息"""
    logger.debug(message)


def log_warning(message: str):
    """记录警告级别消息"""
    logger.warning(message)


def log_error(message: str):
    """记录错误级别消息"""
    logger.error(message)


def log_critical(message: str):
    """记录严重级别消息"""
    logger.critical(message)


def log_success(message: str):
    """记录成功操作"""
    logger.info(f"SUCCESS: {message}")


def log_failure(message: str):
    """记录失败操作"""
    logger.error(f"FAILURE: {message}")


def log_step(message: str):
    """记录步骤消息"""
    logger.info(f"STEP: {message}")


def log_query(query: str):
    """记录查询消息"""
    logger.info(f"QUERY: {query}")


def log_result(result: str):
    """记录结果消息"""
    logger.info(f"RESULT: {result}")


def log_tool_call(tool_name: str, params: dict):
    """记录工具调用消息"""
    logger.info(f"TOOL_CALL: {tool_name}, PARAMS: {params}")


def log_tool_result(tool_name: str, result: str):
    """记录工具结果消息"""
    logger.info(f"TOOL_RESULT: {tool_name}, RESULT: {result}")


def log_visual_retrieval(query: str, top_k: int, results_count: int):
    """记录视觉检索消息"""
    logger.info(f"VISUAL_RETRIEVAL: Query='{query}', Requested={top_k}, Returned={results_count}")


def log_smart_zoom(image_path: str, coords: tuple, scale: float):
    """记录智能缩放消息"""
    logger.info(f"SMART_ZOOM: Image='{Path(image_path).name}', Coords={coords}, Scale={scale}x")


def log_agent_thinking(thought: str):
    """记录代理思考过程"""
    logger.debug(f"AGENT_THINKING: {thought}")


def log_agent_action(action: str, details: str):
    """记录代理动作"""
    logger.info(f"AGENT_ACTION: {action} - {details}")


def log_agent_response(response: str):
    """记录代理最终响应"""
    logger.info(f"AGENT_RESPONSE: {response[:100]}...")  # Only record first 100 characters


def log_system_status(status: str):
    """记录系统状态"""
    logger.info(f"SYSTEM_STATUS: {status}")


def log_performance(metric: str, value: float, unit: str = ""):
    """记录性能指标"""
    logger.debug(f"PERFORMANCE: {metric} = {value}{unit}")


# 便捷别名
info = log_info
debug = log_debug
warning = log_warning
error = log_error
success = log_success
failure = log_failure
step = log_step
query = log_query
result = log_result
tool_call = log_tool_call
tool_result = log_tool_result
visual_retrieval = log_visual_retrieval
smart_zoom = log_smart_zoom
agent_thinking = log_agent_thinking
agent_action = log_agent_action
agent_response = log_agent_response
system_status = log_system_status
performance = log_performance


if __name__ == "__main__":
    # 测试日志功能
    print("测试日志功能...")

    log_info("This is an info log")
    log_debug("This is a debug log")
    log_warning("This is a warning log")
    log_error("This is an error log")

    log_success("Operation succeeded")
    log_failure("Operation failed")

    log_step("Performing step 1")
    log_query("Searching circuit diagram")
    log_result("Found 3 relevant pages")

    log_tool_call("smart_zoom", {"x": 0.1, "y": 0.2})
    log_visual_retrieval("Relay wiring diagram", 5, 3)
    log_smart_zoom("./test.png", (0.1, 0.2, 0.3, 0.4), 2.0)

    print(f"日志文件创建于 ./logs/ 目录")