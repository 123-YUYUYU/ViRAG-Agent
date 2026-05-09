"""
智能缩放工具模块
实现对图像特定区域的精确裁剪和放大功能
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def smart_zoom(
    image_path: str,
    x: float,
    y: float,
    width: float,
    height: float,
    scale_factor: float = 2.0,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    对图像指定区域进行智能缩放

    Args:
        image_path: 原始图像路径
        x: 起始点x坐标（归一化，0-1之间）
        y: 起始点y坐标（归一化，0-1之间）
        width: 区域宽度（归一化，0-1之间）
        height: 区域高度（归一化，0-1之间）
        scale_factor: 缩放倍数，默认2.0
        output_path: 输出路径，如果不指定则返回PIL Image对象

    Returns:
        PIL Image: 放大后的图像对象
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = image.shape[:2]

    # 将归一化坐标转换为像素坐标
    x_pixel = int(x * w)
    y_pixel = int(y * h)
    width_pixel = int(width * w)
    height_pixel = int(height * h)

    # 边界检查，确保坐标在图像范围内
    x_pixel = max(0, min(x_pixel, w))
    y_pixel = max(0, min(y_pixel, h))
    width_pixel = min(width_pixel, w - x_pixel)
    height_pixel = min(height_pixel, h - y_pixel)

    # 裁剪指定区域
    cropped_region = image[y_pixel:y_pixel + height_pixel, x_pixel:x_pixel + width_pixel]

    # 如果裁剪区域太小，扩展到最小尺寸
    min_size = 64  # 最小64x64像素
    if cropped_region.shape[0] < min_size or cropped_region.shape[1] < min_size:
        # 计算需要扩展的尺寸
        pad_h = max(min_size - cropped_region.shape[0], 0)
        pad_w = max(min_size - cropped_region.shape[1], 0)

        # 上下左右均匀填充
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        cropped_region = cv2.copyMakeBorder(
            cropped_region, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

    # 使用高质量插值进行放大
    new_width = int(cropped_region.shape[1] * scale_factor)
    new_height = int(cropped_region.shape[0] * scale_factor)

    # 使用LANCZOS插值获得最佳质量
    upscaled_region = cv2.resize(
        cropped_region,
        (new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4
    )

    # 转换为PIL Image对象
    # OpenCV使用BGR，转换为RGB
    upscaled_region_rgb = cv2.cvtColor(upscaled_region, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(upscaled_region_rgb)

    # 如果指定了输出路径，保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_image.save(output_path)
        print(f"智能缩放结果已保存至: {output_path}")

    return result_image


def extract_zoom_coordinates(text: str) -> Optional[Tuple[float, float, float, float]]:
    """
    从文本中提取缩放坐标信息
    期望格式: [ZOOM: x, y, w, h] 其中坐标为0-1之间的归一化值

    Args:
        text: 包含坐标信息的文本

    Returns:
        Tuple[float, float, float, float]: (x, y, width, height) 或 None
    """
    import re

    # 正则表达式匹配 [ZOOM: x, y, w, h] 格式
    pattern = r'\[ZOOM:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    match = re.search(pattern, text)

    if match:
        try:
            x, y, w, h = map(float, match.groups())
            # 验证坐标在0-1范围内
            if all(0 <= val <= 1 for val in [x, y, w, h]):
                # 验证不会超出边界
                if x + w <= 1 and y + h <= 1:
                    return x, y, w, h
        except (ValueError, IndexError):
            pass

    return None


def visualize_zoom_area(
    image_path: str,
    x: float,
    y: float,
    width: float,
    height: float,
    save_path: Optional[str] = None
) -> Image.Image:
    """
    在原图上可视化缩放区域（画框）

    Args:
        image_path: 原始图像路径
        x, y, width, height: 归一化坐标
        save_path: 保存路径

    Returns:
        PIL Image: 带有标注框的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 转换为像素坐标
    x_pixel = int(x * w)
    y_pixel = int(y * h)
    width_pixel = int(width * w)
    height_pixel = int(height * h)

    # 画矩形框（绿色，线条粗细为2）
    color = (0, 255, 0)  # BGR格式的绿色
    thickness = 2
    cv2.rectangle(
        image,
        (x_pixel, y_pixel),
        (x_pixel + width_pixel, y_pixel + height_pixel),
        color, thickness
    )

    # 转换为PIL Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(image_rgb)

    if save_path:
        result.save(save_path)

    return result


class SmartZoomProcessor:
    """
    智能缩放处理器类
    封装智能缩放的所有功能
    """

    def __init__(self, default_scale_factor: float = 2.0, min_region_size: int = 64):
        """
        初始化智能缩放处理器

        Args:
            default_scale_factor: 默认缩放倍数
            min_region_size: 最小区域尺寸（像素）
        """
        self.default_scale_factor = default_scale_factor
        self.min_region_size = min_region_size

    def process_zoom_request(
        self,
        image_path: str,
        coordinates: Tuple[float, float, float, float],
        scale_factor: Optional[float] = None,
        output_dir: str = "./data/zoomed_regions/"
    ) -> Tuple[Image.Image, str]:
        """
        处理缩放请求

        Args:
            image_path: 原始图像路径
            coordinates: (x, y, width, height) 归一化坐标
            scale_factor: 缩放倍数，如果不指定使用默认值
            output_dir: 输出目录

        Returns:
            Tuple[PIL Image, str]: (缩放后的图像, 输出文件路径)
        """
        if scale_factor is None:
            scale_factor = self.default_scale_factor

        # 创建输出文件名
        base_name = Path(image_path).stem
        region_id = f"{coordinates[0]:.2f}_{coordinates[1]:.2f}_{coordinates[2]:.2f}_{coordinates[3]:.2f}"
        output_filename = f"{base_name}_zoom_{region_id}.png"
        output_path = os.path.join(output_dir, output_filename)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 执行智能缩放
        zoomed_image = smart_zoom(
            image_path=image_path,
            x=coordinates[0],
            y=coordinates[1],
            width=coordinates[2],
            height=coordinates[3],
            scale_factor=scale_factor,
            output_path=output_path
        )

        return zoomed_image, output_path

    def extract_and_process_zoom(
        self,
        text_response: str,
        image_path: str,
        output_dir: str = "./data/zoomed_regions/"
    ) -> Optional[Tuple[Image.Image, str]]:
        """
        从文本响应中提取缩放坐标并处理

        Args:
            text_response: 包含[ZOOM: ...]坐标的文本响应
            image_path: 原始图像路径
            output_dir: 输出目录

        Returns:
            Optional[Tuple[PIL Image, str]]: (缩放后的图像, 输出文件路径) 或 None
        """
        coordinates = extract_zoom_coordinates(text_response)
        if coordinates:
            return self.process_zoom_request(
                image_path=image_path,
                coordinates=coordinates,
                output_dir=output_dir
            )
        return None

    def batch_zoom_regions(
        self,
        image_path: str,
        coordinates_list,
        output_dir: str = "./data/zoomed_regions/"
    ):
        """
        批量缩放多个区域

        Args:
            image_path: 原始图像路径
            coordinates_list: 坐标列表
            output_dir: 输出目录

        Returns:
            List[Tuple[PIL Image, str]]: 缩放结果列表
        """
        from typing import List  # 导入List类型
        results: List[Tuple[Image.Image, str]] = []
        for i, coords in enumerate(coordinates_list):
            try:
                zoomed_img, output_path = self.process_zoom_request(
                    image_path=image_path,
                    coordinates=coords,
                    output_dir=output_dir
                )
                results.append((zoomed_img, output_path))
            except Exception as e:
                print(f"处理第{i+1}个区域时出错: {str(e)}")
                continue

        return results


# 使用示例和测试函数
def demo_smart_zoom():
    """
    智能缩放示例
    """
    print("=== 智能缩放示例 ===")

    # 示例图像路径（实际使用时替换为真实路径）
    sample_images = [
        img for img in os.listdir("./data/page_images/")
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ] if os.path.exists("./data/page_images/") else []

    if not sample_images:
        print("❌ 未找到示例图像，请先运行PDF渲染脚本")
        return

    image_path = os.path.join("./data/page_images/", sample_images[0])
    print(f"使用示例图像: {image_path}")

    # 示例坐标（左上角1/4区域）
    x, y, w, h = 0.0, 0.0, 0.5, 0.5

    try:
        # 执行智能缩放
        zoomed_image = smart_zoom(
            image_path=image_path,
            x=x, y=y, width=w, height=h,
            scale_factor=2.0
        )

        print(f"✓ 缩放完成，原始尺寸: {cv2.imread(image_path).shape[:2][::-1]}")
        print(f"✓ 缩放后尺寸: {zoomed_image.size}")

        # 保存结果
        output_path = "./data/zoomed_regions/demo_zoom.png"
        os.makedirs("./data/zoomed_regions/", exist_ok=True)
        zoomed_image.save(output_path)
        print(f"✓ 缩放结果已保存至: {output_path}")

        # 测试坐标提取功能
        sample_text = "图像中的细节不够清晰，需要放大查看 [ZOOM: 0.2, 0.3, 0.1, 0.1]"
        coords = extract_zoom_coordinates(sample_text)
        if coords:
            print(f"✓ 成功提取坐标: {coords}")

            # 使用处理器类
            processor = SmartZoomProcessor()
            zoom_result = processor.extract_and_process_zoom(
                text_response=sample_text,
                image_path=image_path
            )
            if zoom_result:
                print(f"✓ 处理器处理完成: {zoom_result[1]}")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")


if __name__ == "__main__":
    demo_smart_zoom()