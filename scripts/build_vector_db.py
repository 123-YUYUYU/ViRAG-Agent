"""
ADAS手册视觉向量数据库构建主脚本
集成PDF处理、CLIP编码和ChromaDB存储功能
支持清空重建和增量添加模式
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.pdf_processor import PDFProcessor
from model.clip_encoder import CLIPEncoder
from retrieval.retriever import VisualRetriever, build_visual_index_from_images
from utils.logger import (
    log_step, log_info, log_success, log_error,
    log_failure, log_performance
)
import time


def main(reset_db=False):
    """
    主函数：构建ADAS手册视觉向量数据库
    Args:
        reset_db: 是否清空现有数据库
    """
    log_step("开始构建ADAS手册视觉向量数据库")

    start_time = time.time()

    try:
        # 1. 设置目录结构
        log_step("步骤 1: 设置目录结构")
        data_dir = Path("./data")
        manual_dir = data_dir / "manual"
        images_dir = data_dir / "page_images"
        db_dir = data_dir / "chroma_db"

        for dir_path in [data_dir, manual_dir, images_dir, db_dir]:
            dir_path.mkdir(exist_ok=True)

        log_success("目录结构设置完成")

        # 2. 验证输入PDF文件
        log_step("步骤 2: 验证输入PDF文件")
        pdf_files = [f for f in manual_dir.iterdir() if f.suffix.lower() == '.pdf']

        if not pdf_files:
            log_error(f"在 {manual_dir} 目录中未找到PDF文件")
            log_info("请将ADAS安装手册PDF文件放在 ./data/manual/ 目录中")
            return

        log_info(f"找到 {len(pdf_files)} 个PDF文件: {[f.name for f in pdf_files]}")

        # 3. 初始化PDF处理器
        log_step("步骤 3: 初始化PDF处理器")
        pdf_processor = PDFProcessor(default_dpi=300)

        # 4. 处理所有PDF文件
        log_step("步骤 4: 将PDF渲染为高清图像")
        all_image_data = []

        for pdf_file in pdf_files:
            log_info(f"处理PDF文件: {pdf_file.name}")

            # 验证PDF
            is_valid, error_msg = pdf_processor.validate_pdf(str(pdf_file))
            if not is_valid:
                log_error(f"PDF验证失败 {pdf_file.name}: {error_msg}")
                continue

            # 将PDF渲染为图像
            try:
                image_data = pdf_processor.render_pdf_to_images(
                    pdf_path=str(pdf_file),
                    output_dir=str(images_dir),
                    dpi=300
                )

                all_image_data.extend(image_data)
                log_success(f"完成处理 {pdf_file.name}, 生成 {len(image_data)} 张图像")

            except Exception as e:
                log_error(f"处理PDF {pdf_file.name} 时出错: {str(e)}")
                continue

        if not all_image_data:
            log_failure("没有成功处理任何PDF文件，退出程序")
            return

        log_success(f"PDF处理完成，总共生成 {len(all_image_data)} 张高清图像")

        # 5. 初始化CLIP编码器
        log_step("步骤 5: 初始化CLIP视觉编码器")
        clip_encoder = CLIPEncoder(model_name="openai/clip-vit-base-patch32")

        # 6. 批量编码所有图像
        log_step("步骤 6: 批量提取图像视觉特征")
        total_images = len(all_image_data)
        log_info(f"开始编码 {total_images} 张图像...")

        encode_start_time = time.time()

        # 分批处理以节省内存
        batch_size = 16
        all_embeddings = []

        for i in range(0, total_images, batch_size):
            batch_data = all_image_data[i:i + batch_size]
            batch_paths = [item[0] for item in batch_data]  # 提取图像路径

            log_info(f"处理批次 {i//batch_size + 1}/{(total_images-1)//batch_size + 1}")

            try:
                batch_embeddings = clip_encoder.encode_images(
                    images=batch_paths,
                    batch_size=4  # 使用较小的批量大小进行图像编码
                )

                all_embeddings.extend(batch_embeddings)
                log_info(f"  批次完成，编码 {len(batch_embeddings)} 张图像")

            except Exception as e:
                log_error(f"批次 {i//batch_size + 1} 编码失败: {str(e)}")
                continue

        # 转换为numpy数组
        import numpy as np
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            log_success(f"图像编码完成，特征矩阵形状: {all_embeddings.shape}")
        else:
            log_failure("没有成功编码任何图像，退出程序")
            return

        encode_end_time = time.time()
        log_performance("图像编码时间", encode_end_time - encode_start_time, "秒")

        # 7. 初始化向量数据库
        log_step("步骤 7: 初始化ChromaDB向量数据库")
        collection_name = "adas_manual_visual_index"

        # 如果reset_db为True，则清空整个数据库目录
        if reset_db:
            import shutil
            db_path = Path("./data/chroma_db")
            if db_path.exists():
                log_info(f"清空数据库目录: {db_path}")
                shutil.rmtree(db_path)
                # 重新创建目录
                db_path.mkdir(parents=True, exist_ok=True)

        retriever = VisualRetriever(
            collection_name=collection_name,
            db_path="./data/chroma_db",
            distance_metric="cosine"
        )

        # 8. 构建视觉索引
        log_step("步骤 8: 构建视觉向量索引")
        build_visual_index_from_images(
            image_data=all_image_data,
            clip_encoder=clip_encoder,
            retriever=retriever,
            batch_size=32
        )

        # 9. 验证数据库
        log_step("步骤 9: 验证向量数据库")
        stats = retriever.get_collection_stats()
        log_info(f"数据库统计: {stats}")

        # 10. 完成
        end_time = time.time()
        total_time = end_time - start_time

        log_success("=" * 60)
        log_success("🎉 ADAS手册视觉向量数据库构建完成!")
        log_success(f"📊 统计:")
        log_success(f"   - 处理的PDF文件: {len(pdf_files)}")
        log_success(f"   - 生成的图像: {len(all_image_data)}")
        log_success(f"   - 向量维度: {all_embeddings.shape[1]}")
        log_success(f"   - 数据库存储: {stats['total_vectors']} 个向量")
        log_success(f"   - 总耗时: {total_time:.2f} 秒")
        log_success(f"   - 数据库存储位置: {db_dir}")
        log_success(f"   - 操作模式: {'重置重建' if reset_db else '增量添加'}")
        log_success("=" * 60)

        # 11. 生成使用说明
        log_step("生成后续使用说明")
        usage_instructions = f"""
# ADAS手册视觉检索系统构建完成!

## 操作模式: {'重置重建' if reset_db else '增量添加'}

## 下一步骤：
1. 启动检索服务: python -m retrieval.retriever_test
2. 启动对话Agent: python -m agent.react_agent
3. 启动Web界面: python apps/web_demo.py

## 使用示例：
- 文本检索: '查找继电器接线图'
- 图像检索: 上传相似图像进行相似性搜索
- 智能问答: 直接询问有关ADAS安装的问题
        """

        readme_path = data_dir / "BUILD_COMPLETE_README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(usage_instructions)

        log_success(f"使用说明已生成: {readme_path}")

    except Exception as e:
        log_error(f"构建过程中发生错误: {str(e)}")
        log_failure("数据库构建失败")
        import traceback
        log_error(f"详细错误信息:\n{traceback.format_exc()}")

        # 记录错误到文件
        error_log_path = Path("./logs/error_build.log")
        error_log_path.parent.mkdir(exist_ok=True)
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 构建错误:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='构建ADAS手册视觉向量数据库')
    parser.add_argument('--reset', action='store_true', help='清空现有数据库并重新构建')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(reset_db=args.reset)