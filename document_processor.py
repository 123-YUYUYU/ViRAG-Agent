import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from data_types import DocBlock
from typing import List
import json
import uuid


def parse_document(file_path: str) -> List[DocBlock]:
    """
    解析文档并输出 List[DocBlock]

    Args:
        file_path: PDF文件路径

    Returns:
        List[DocBlock]: 包含文本、图像、表格的文档块列表
    """
    doc_blocks = []

    # 打开PDF文档
    pdf_doc = fitz.open(file_path)

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]

        # 获取页面中的文本块
        text_dict = page.get_text("dict")

        # 处理文本块
        for block in text_dict["blocks"]:
            if "lines" in block:  # 文本块
                text_content = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_content += span["text"] + " "

                if text_content.strip():  # 如果文本非空
                    doc_block = DocBlock(
                        id=str(uuid.uuid4()),
                        content=text_content.strip(),
                        block_type="text",
                        metadata={
                            "page_num": page_num + 1,
                            "source": os.path.basename(file_path),
                            "bbox": block["bbox"]  # 边界框信息
                        }
                    )
                    doc_blocks.append(doc_block)

            # 处理图像块
            if "images" in block:
                for img_idx, img in enumerate(block["images"]):
                    # 获取图像信息
                    xref = img["xref"]

                    # 尝试从PDF中提取图像
                    try:
                        pix = pdf_doc.extract_image(xref)
                        img_bytes = pix["image"]

                        # 保存图像到临时位置
                        img_filename = f"temp_img_page_{page_num + 1}_{img_idx}.png"
                        img_path = os.path.join("./data/temp_images/", img_filename)

                        # 确保目录存在
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)

                        with open(img_path, "wb") as img_file:
                            img_file.write(img_bytes)

                        # 进行OCR识别
                        pil_img = Image.open(img_path)
                        ocr_text = pytesseract.image_to_string(pil_img, lang='chi_sim+eng')

                        doc_block = DocBlock(
                            id=str(uuid.uuid4()),
                            content=img_path,
                            block_type="image",
                            metadata={
                                "page_num": page_num + 1,
                                "source": os.path.basename(file_path),
                                "ocr_text": ocr_text,
                                "bbox": img["bbox"] if "bbox" in img else None
                            }
                        )
                        doc_blocks.append(doc_block)

                    except Exception as e:
                        print(f"无法提取图像 {xref}，错误: {e}")
                        continue

        # 处理表格（作为特殊文本块）
        tables = page.find_tables()
        for table_idx, table in enumerate(tables):
            try:
                table_dict = table.extract()
                if table_dict:
                    # 将表格转换为JSON字符串
                    table_str = json.dumps(table_dict, ensure_ascii=False, indent=2)

                    doc_block = DocBlock(
                        id=str(uuid.uuid4()),
                        content=table_str,
                        block_type="table",
                        metadata={
                            "page_num": page_num + 1,
                            "source": os.path.basename(file_path),
                            "caption": f"Table on page {page_num + 1}",
                            "bbox": table.bbox
                        }
                    )
                    doc_blocks.append(doc_block)
            except Exception as e:
                print(f"无法提取表格 {table_idx}，错误: {e}")
                continue

    pdf_doc.close()
    return doc_blocks


def parse_document_with_images(file_path: str, output_dir: str = "./data/page_images/") -> List[DocBlock]:
    """
    将PDF渲染为图像后进行解析，符合原有项目的图像处理方式

    Args:
        file_path: PDF文件路径
        output_dir: 图像输出目录

    Returns:
        List[DocBlock]: 包含文档块的列表
    """
    doc_blocks = []

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开PDF文档
    pdf_doc = fitz.open(file_path)

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]

        # 渲染页面为图像
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
        pix = page.get_pixmap(matrix=mat)

        # 保存为图像文件
        img_filename = f"page_{page_num + 1:03d}.png"
        img_path = os.path.join(output_dir, img_filename)
        pix.save(img_path)

        # 对图像进行OCR识别
        pil_img = Image.open(img_path)
        ocr_text = pytesseract.image_to_string(pil_img, lang='chi_sim+eng')

        # 创建图像类型的DocBlock
        doc_block = DocBlock(
            id=str(uuid.uuid4()),
            content=img_path,
            block_type="image",
            metadata={
                "page_num": page_num + 1,
                "source": os.path.basename(file_path),
                "ocr_text": ocr_text
            }
        )
        doc_blocks.append(doc_block)

    pdf_doc.close()
    return doc_blocks