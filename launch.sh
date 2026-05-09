#!/bin/bash

# ADAS多模态Agent系统启动脚本
# 用于Linux/macOS环境下的便捷启动

echo "🚛 ADAS工业车辆多模态Agent系统"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未找到，请确保已安装Python3并添加到PATH"
    exit 1
fi

echo "✅ Python3环境检查通过: $(python3 --version)"

# 检查依赖
echo "🔍 检查依赖包..."
if python3 -c "import torch, transformers, gradio, chromadb, fitz, PIL, cv2" &> /dev/null; then
    echo "✅ 依赖检查通过"
else
    echo "⚠️  部分依赖未安装，正在尝试安装..."
    if pip install -r requirements.txt; then
        echo "✅ 依赖安装完成"
    else
        echo "❌ 依赖安装失败，请检查网络连接或手动安装"
        exit 1
    fi
fi

# 显示菜单函数
show_menu() {
    echo ""
    echo "==================== 启动选项 ===================="
    echo "1. 启动Web界面 (推荐)"
    echo "2. 启动CLI交互模式"
    echo "3. 构建向量数据库"
    echo "4. 查看系统信息"
    echo "5. 检查数据文件"
    echo "6. 退出"
    echo "================================================"
}

# 主循环
while true; do
    show_menu
    read -p "请选择 (1-6): " choice

    case $choice in
        1)
            echo "🚀 启动Web界面..."
            python3 main.py web
            ;;
        2)
            echo "💻 启动CLI交互模式..."
            python3 main.py cli
            ;;
        3)
            echo "🏗️  构建向量数据库..."
            echo "请确保 ./data/manual/ 目录下有PDF文件"
            python3 main.py build
            ;;
        4)
            echo "📋 系统信息..."
            python3 main.py info
            ;;
        5)
            echo "🔍 检查数据文件..."
            if [ -d "data/manual/" ] && [ -n "$(ls -A data/manual/*.pdf 2>/dev/null)" ]; then
                echo "✅ 发现PDF文件:"
                ls data/manual/*.pdf
            else
                echo "❌ 未在 ./data/manual/ 目录找到PDF文件"
            fi

            if [ -d "data/page_images/" ] && [ -n "$(ls -A data/page_images/*.png 2>/dev/null)" ]; then
                image_count=$(ls data/page_images/*.png | wc -l)
                echo "✅ 发现页面图片: ${image_count} 个文件"
            else
                echo "⚠️  未生成页面图片 (运行构建命令可生成)"
            fi
            ;;
        6)
            echo "👋 再见！"
            exit 0
            ;;
        *)
            echo "⚠️  无效选择，请重新输入"
            ;;
    esac

    echo ""
    read -p "按Enter键继续..."
done