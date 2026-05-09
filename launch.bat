@echo off
REM ADAS多模态Agent系统批处理启动脚本
REM 用于Windows环境下的便捷启动

echo 🚛 ADAS工业车辆多模态Agent系统
echo ==========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未找到，请确保已安装Python并添加到PATH
    pause
    exit /b 1
)

echo ✅ Python环境检查通过

REM 检查依赖
echo 🔍 检查依赖包...
python -c "import torch, transformers, gradio, chromadb, fitz, PIL, cv2" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  部分依赖未安装，正在尝试安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请检查网络连接或手动安装
        pause
        exit /b 1
    )
    echo ✅ 依赖安装完成
) else (
    echo ✅ 依赖检查通过
)

REM 显示菜单
:menu
echo.
echo ==================== 启动选项 ====================
echo 1. 启动Web界面 (推荐)
echo 2. 启动CLI交互模式
echo 3. 构建向量数据库
echo 4. 查看系统信息
echo 5. 检查数据文件
echo 6. 退出
echo ================================================

set /p choice="请选择 (1-6): "

if "%choice%"=="1" goto web
if "%choice%"=="2" goto cli
if "%choice%"=="3" goto build
if "%choice%"=="4" goto info
if "%choice%"=="5" goto check_data
if "%choice%"=="6" goto exit

echo ⚠️  无效选择，请重新输入
goto menu

:web
echo 🚀 启动Web界面...
python main.py web
pause
goto menu

:cli
echo 💻 启动CLI交互模式...
python main.py cli
pause
goto menu

:build
echo 🏗️  构建向量数据库...
echo 请确保 ./data/manual/ 目录下有PDF文件
python main.py build
pause
goto menu

:info
echo 📋 系统信息...
python main.py info
pause
goto menu

:check_data
echo 🔍 检查数据文件...
dir /b data\manual\*.pdf >nul 2>&1
if errorlevel 1 (
    echo ❌ 未在 ./data/manual/ 目录找到PDF文件
) else (
    echo ✅ 发现PDF文件:
    dir /b data\manual\*.pdf
)
dir /b data\page_images\*.png >nul 2>&1
if errorlevel 1 (
    echo ⚠️  未生成页面图片 (运行构建命令可生成)
) else (
    echo ✅ 发现页面图片: (统计中...)
    for /f %%i in ('dir /b data\page_images\*.png ^| find /c /v ""') do echo %%i 个图片文件
)
goto menu

:exit
echo 👋 再见！
pause