"""
ADAS多模态Agent系统模块完整性验证脚本
用于验证所有模块是否能够正确导入和初始化
"""

import sys
import os
from pathlib import Path
import importlib.util
import traceback


def check_module_import(module_path):
    """
    检查模块是否可以成功导入

    Args:
        module_path: 模块文件路径

    Returns:
        bool: 是否导入成功
    """
    try:
        spec = importlib.util.spec_from_file_location(
            module_path.stem,
            module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(f"错误: {module_path.name}: {str(e)}")
        return False


def validate_project_structure():
    """验证项目结构完整性"""
    print("检查项目结构完整性...")

    required_dirs = [
        "agent",
        "apps",
        "retrieval",
        "model",
        "tools",
        "dataset",
        "llm",
        "scripts",
        "data",
        "utils",
        "data/manual",
        "data/page_images",
        "data/chroma_db",
        "data/zoomed_regions",
        "data/memory"
    ]

    all_dirs_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"缺失: {dir_name}")
            all_dirs_exist = False
        else:
            print(f"正常: {dir_name}")

    return all_dirs_exist


def validate_python_modules():
    """验证Python模块完整性"""
    print("\n检查Python模块完整性...")

    # 需要验证的模块文件
    modules_to_check = [
        Path("config.py"),
        Path("agent/react_agent.py"),
        Path("agent/memory.py"),
        Path("apps/web_demo.py"),
        Path("retrieval/retriever.py"),
        Path("model/clip_encoder.py"),
        Path("tools/smart_zoom_tool.py"),
        Path("tools/schema.py"),
        Path("dataset/pdf_processor.py"),
        Path("llm/qwen2_vl_client.py"),
        Path("llm/prompts.py"),
        Path("scripts/build_vector_db.py"),
        Path("utils/logger.py"),
        Path("main.py")
    ]

    successful_imports = 0
    total_modules = len(modules_to_check)

    for module_path in modules_to_check:
        if module_path.exists():
            print(f"检查: {module_path}")
            if check_module_import(module_path):
                print(f"成功: {module_path.name} - 导入成功")
                successful_imports += 1
            else:
                print(f"失败: {module_path.name} - 导入失败")
        else:
            print(f"错误: 模块不存在: {module_path}")

    print(f"\n模块导入统计: {successful_imports}/{total_modules} 成功")
    return successful_imports == total_modules


def validate_dependencies():
    """验证依赖包"""
    print("\n检查依赖包...")

    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("gradio", "gradio"),
        ("chromadb", "chromadb"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("fitz", "PyMuPDF"),
        ("sentence_transformers", "sentence-transformers")
    ]

    successful_imports = 0

    for pkg_name, pkg_display in required_packages:
        try:
            if pkg_name == "fitz":
                import fitz
            elif pkg_name == "cv2":
                import cv2
            elif pkg_name == "PIL":
                from PIL import Image
            else:
                importlib.import_module(pkg_name)

            print(f"可用: {pkg_display}")
            successful_imports += 1
        except ImportError as e:
            print(f"缺失: {pkg_display} - 错误: {str(e)}")

    print(f"\n依赖包统计: {successful_imports}/{len(required_packages)} 可用")
    return successful_imports == len(required_packages)


def validate_data_files():
    """验证数据文件"""
    print("\n检查数据文件...")

    data_checks = [
        ("Data directory", Path("data")),
        ("PDF directory", Path("data/manual")),
        ("Images directory", Path("data/page_images")),
        ("Vector DB directory", Path("data/chroma_db")),
        ("Zoomed images directory", Path("data/zoomed_regions")),
        ("Memory directory", Path("data/memory"))
    ]

    all_good = True
    for desc, path in data_checks:
        if path.exists():
            print(f"存在: {desc}")
        else:
            print(f"缺失: {desc} (可通过运行构建脚本创建)")
            all_good = False

    return all_good


def run_comprehensive_test():
    """运行综合测试"""
    print("运行ADAS多模态Agent系统完整性测试...\n")
    print("="*60)

    # 测试项目结构
    structure_ok = validate_project_structure()

    # 测试模块导入
    modules_ok = validate_python_modules()

    # 测试依赖
    deps_ok = validate_dependencies()

    # 测试数据文件
    data_ok = validate_data_files()

    print("\n" + "="*60)
    print("完整性测试结果:")

    results = {
        "Project Structure": structure_ok,
        "Module Imports": modules_ok,
        "Dependencies": deps_ok,
        "Data Files": data_ok
    }

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过！系统已准备就绪。")
        print("\n推荐的启动命令:")
        print("   python main.py web          # 启动Web界面")
        print("   python main.py cli          # 启动CLI交互")
        print("   python main.py build        # 构建向量数据库")
    else:
        print("某些测试失败，请检查以上问题。")
        print("\n常见解决方案:")
        print("   pip install -r requirements.txt    # 安装依赖")
        print("   python main.py build               # 构建数据库")
        print("   python main.py info                # 查看系统信息")

    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)