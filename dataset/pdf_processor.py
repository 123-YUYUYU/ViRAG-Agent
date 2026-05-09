"""
PDF处理器模块
使用PyMuPDF将PDF文档渲染为高清图像并提取页面元数据
"""

import fitz  # PyMuPDF
from PIL import Image
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import hashlib
from tqdm import tqdm


class PDFProcessor:
    """
    PDF处理器类
    负责将PDF文档渲染为高清图像并返回页面元数据
    """

    def __init__(self, default_dpi: int = 300):
        """
        初始化PDF处理器

        Args:
            default_dpi: 默认渲染DPI，默认300
        """
        self.default_dpi = default_dpi

    def render_pdf_to_images(
        self,
        pdf_path: str,
        output_dir: str,
        dpi: Optional[int] = None
    ) -> List[Tuple[str, Dict]]:
        """
        将每个PDF页面渲染为高清图像

        Args:
            pdf_path: PDF文件路径
            output_dir: 图像输出目录
            dpi: 渲染DPI，如未指定则使用默认值

        Returns:
            List[Tuple[str, Dict]]: 包含(image_path, metadata)的列表
        """
        if dpi is None:
            dpi = self.default_dpi

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        print(f"渲染PDF: {pdf_path}")
        print(f"渲染参数: {dpi} DPI")

        # 打开PDF文档
        doc = fitz.open(pdf_path)
        results = []

        for page_num in tqdm(range(len(doc)), desc="渲染PDF页面"):
            page = doc.load_page(page_num)

            # 计算缩放因子以达到目标DPI
            # PyMuPDF默认DPI为72，所以缩放因子 = 目标DPI / 72
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            # 将页面渲染为图像（使用PNG格式以保留细节）
            pix = page.get_pixmap(matrix=mat)

            # 生成图像文件名（使用PDF_name_page_num.png）
            # 将PDF名称中的非ASCII字符转换为ASCII等效字符
            import unicodedata
            pdf_name = Path(pdf_path).stem
            pdf_name_ascii = unicodedata.normalize('NFKD', pdf_name).encode('ascii', 'ignore').decode('ascii')
            if not pdf_name_ascii:  # 如果所有字符都是非ASCII字符，则使用默认名称
                pdf_name_ascii = "document"
            img_filename = f"{pdf_name_ascii}_page_{page_num + 1}.png"
            img_path = os.path.join(output_dir, img_filename)

            # 保存图像
            pix.save(img_path)
            print(
                f"[PDF_DEBUG] page_index={page_num} dpi={dpi} "
                f"width={pix.width} height={pix.height} image_path={img_path}"
            )

            # 计算图像MD5哈希值用于唯一标识
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()

            # 提取页面文本内容
            text_content = page.get_text("text")

            # 创建详细元数据
            metadata = {
                "page_num": page_num + 1,  # 页面编号（从1开始）
                "source": pdf_name,    # 源PDF名称
                "original_pdf": pdf_path,  # 原始PDF路径
                "image_path": img_path,    # 生成的图像路径
                "image_width": pix.width,  # 图像宽度
                "image_height": pix.height,  # 图像高度
                "dpi": dpi,                # 渲染DPI
                "image_hash": img_hash,    # 图像哈希值
                "rotation": page.rotation,  # 页面旋转角度
                "page_width": page.rect.width,  # 原始页面宽度
                "page_height": page.rect.height,  # 原始页面高度
                "ocr_text": text_content,  # 页面的OCR文本内容，用于BM25检索
            }

            results.append((img_path, metadata))

            print(f"  渲染页面 {page_num + 1} -> {img_filename}")

        doc.close()
        print(f"✓ PDF渲染完成，生成了 {len(results)} 张高清图像")
        return results

    def get_pdf_info(self, pdf_path: str) -> Dict:
        """
        获取PDF文档的基本信息

        Args:
            pdf_path: PDF文件路径

        Returns:
            Dict: PDF文档信息
        """
        doc = fitz.open(pdf_path)

        info = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
            "pages_count": len(doc),
            "encrypted": bool(doc.is_encrypted),
        }

        doc.close()
        return info

    def validate_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """
        验证PDF文件是否可读

        Args:
            pdf_path: PDF文件路径

        Returns:
            Tuple[bool, str]: (是否有效, 错误消息)
        """
        try:
            if not os.path.exists(pdf_path):
                return False, f"PDF文件不存在: {pdf_path}"

            doc = fitz.open(pdf_path)

            if doc.is_encrypted:
                return False, "PDF文件已加密，无法处理"

            if len(doc) == 0:
                return False, "PDF文件为空，无页面内容"

            doc.close()
            return True, ""

        except Exception as e:
            return False, f"PDF文件验证失败: {str(e)}"


def batch_process_pdfs(
    pdf_dir: str,
    output_base_dir: str,
    dpi: int = 300
) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    批量处理目录中的所有PDF文件

    Args:
        pdf_dir: 包含PDF文件的目录
        output_base_dir: 输出目录的基础路径
        dpi: 渲染DPI

    Returns:
        Dict[str, List[Tuple[str, Dict]]]: 每个PDF的处理结果
    """
    processor = PDFProcessor(dpi=dpi)
    results = {}

    # 查找目录中的所有PDF文件
    pdf_files = [
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith('.pdf')
    ]

    if not pdf_files:
        print(f"在目录 {pdf_dir} 中未找到PDF文件")
        return results

    print(f"找到 {len(pdf_files)} 个PDF文件，开始批量处理...")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_name = Path(pdf_file).stem

        # 为每个PDF创建单独的输出子目录
        pdf_output_dir = os.path.join(output_base_dir, pdf_name)

        print(f"\n处理PDF: {pdf_file}")
        is_valid, error_msg = processor.validate_pdf(pdf_path)

        if not is_valid:
            print(f"  ❌ {error_msg}")
            continue

        try:
            pdf_results = processor.render_pdf_to_images(
                pdf_path=pdf_path,
                output_dir=pdf_output_dir,
                dpi=dpi
            )
            results[pdf_file] = pdf_results

        except Exception as e:
            print(f"  ❌ 处理 {pdf_file} 时出错: {str(e)}")
            continue

    print(f"\n✅ 批量处理完成，成功处理了 {len(results)} 个PDF文件")
    return results


# 示例用法
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdf_processor.py <pdf_path> [output_dir] [dpi]")
        print("示例: python pdf_processor.py ./data/manual/adas_manual.pdf ./data/page_images 300")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/page_images"
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    # 创建处理器实例
    processor = PDFProcessor(default_dpi=dpi)

    # 验证PDF
    is_valid, error_msg = processor.validate_pdf(pdf_path)
    if not is_valid:
        print(f"PDF验证失败: {error_msg}")
        sys.exit(1)

    # 将PDF渲染为图像
    results = processor.render_pdf_to_images(
        pdf_path=pdf_path,
        output_dir=output_dir,
        dpi=dpi
    )

    print(f"\n处理完成，生成了 {len(results)} 张图像:")
    for img_path, metadata in results:
        print(f"  - {img_path} ({metadata['image_width']}x{metadata['image_height']})")
