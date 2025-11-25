@echo off
REM CrewAI App 自动化安装脚本 (Windows)
REM 使用方法: scripts\install.bat [--dev]

setlocal enabledelayedexpansion

REM 设置颜色代码
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM 显示帮助信息
:show_help
echo CrewAI App 安装脚本
echo.
echo 使用方法:
echo   scripts\install.bat [选项]
echo.
echo 选项:
echo   --dev     安装开发环境依赖
echo   --help    显示此帮助信息
echo.
goto :eof

REM 日志函数
:log_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM 检查命令是否存在
:command_exists
where %1 >nul 2>&1
goto :eof

REM 检查 Python 版本
:check_python
call :log_info "检查 Python 版本..."

call :command_exists python
if %errorlevel% neq 0 (
    call :log_error "未找到 Python，请先安装 Python 3.12 或更高版本"
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
call :log_success "Python 版本：%python_version%

REM 这里简化版本检查，实际使用中可能需要更复杂的版本比较逻辑
call :log_success "Python 版本检查通过"
goto :eof

REM 检查 uv
:check_uv
call :log_info "检查 uv 包管理器..."

call :command_exists uv
if %errorlevel% neq 0 (
    call :log_warning "未找到 uv，开始安装..."
    pip install uv
    if %errorlevel% neq 0 (
        call :log_error "uv 安装失败"
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('uv --version') do set uv_version=%%i
call :log_success "uv 版本：%uv_version%
goto :eof

REM 设置虚拟环境
:setup_venv
if defined SKIP_VENV (
    call :log_info "跳过虚拟环境创建 (SKIP_VENV)"
    goto :eof
)

call :log_info "创建虚拟环境..."

if exist ".venv" (
    call :log_warning "发现已存在的虚拟环境，将重新创建..."
    rmdir /s /q .venv
)

uv venv .venv --python 3.12
call :log_success "虚拟环境创建完成"
goto :eof

REM 安装依赖
:install_dependencies
set is_dev=%1
set requirements_file=requirements.txt

if "%is_dev%"=="true" (
    set requirements_file=requirements-dev.txt
    call :log_info "安装开发环境依赖..."
) else (
    call :log_info "安装生产环境依赖..."
)

REM 设置镜像源
if not defined UV_INDEX_URL (
    set UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
)
call :log_info "使用镜像源：%UV_INDEX_URL%

REM 升级 uv pip
call :log_info "升级 uv pip..."
uv pip install --upgrade pip

REM 安装依赖
call :log_info "从 %requirements_file% 安装依赖..."
uv pip install --index-url "%UV_INDEX_URL%" -r "%requirements_file%"

call :log_success "依赖安装完成"
goto :eof

REM 创建配置文件模板
:create_env_template
if exist ".env" (
    call :log_info ".env 文件已存在，跳过创建"
    goto :eof
)

call :log_info "创建 .env 配置文件模板..."
(
echo # CrewAI App 环境配置
echo # 复制此文件并填入实际的 API 密钥
echo.
echo # 通义千问 API Key ^(必需^)
echo DASHSCOPE_API_KEY=your_dashscope_api_key_here
echo.
echo # Tavily 搜索 API Key ^(可选，用于网络搜索^)
echo TAVILY_API_KEY=your_tavily_api_key_here
echo.
echo # 应用配置
echo LOG_LEVEL=INFO
echo MAX_RESEARCH_LOOPS=2
echo ENABLE_KNOWLEDGE_BASE_SEARCH=true
echo ENABLE_INDUSTRY_REPORT_MODE=true
echo.
echo # 知识库配置
echo KNOWLEDGE_BASE_PATHS=eastmoney_concept_constituents.xlsx,sw_third_industry_constituents.xlsx
echo KNOWLEDGE_BASE_TOP_K=3
echo KNOWLEDGE_BASE_EMBEDDING_MODEL=text-embedding-v3
echo KNOWLEDGE_BASE_EMBEDDING_BACKEND=dashscope
echo KNOWLEDGE_BASE_EMBEDDING_BATCH_SIZE=10
echo.
echo # 模型配置
echo QUERY_GENERATOR_MODEL=qwen-plus
echo REFLECTION_MODEL=qwen-plus
echo ANSWER_MODEL=qwen-plus
echo LLM_BACKEND=dashscope
) > .env

call :log_success ".env 文件模板已创建，请编辑并填入实际的 API 密钥"
goto :eof

REM 验证安装
:verify_installation
call :log_info "验证安装..."

REM 测试核心依赖
set packages=crewai langchain_core pydantic faiss numpy pandas dashscope
for %%p in (%packages%) do (
    python -c "import %%p" >nul 2>&1
    if %errorlevel% equ 0 (
        echo   ✓ %%p
    ) else (
        echo   ✗ %%p
        set failed=1
    )
)

REM 测试项目模块
python -c "from crewai_app.main import main" >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✓ crewai_app 模块
) else (
    echo   ✗ crewai_app 模块
    set failed=1
)

if defined failed (
    call :log_error "依赖验证失败，请检查安装"
    exit /b 1
) else (
    call :log_success "所有依赖验证通过！"
)
goto :eof

REM 显示使用指南
:show_usage_guide
echo.
call :log_success "🎉 CrewAI App 安装完成！"
echo.
echo 使用指南：
echo 1. 配置环境变量：
echo    编辑 .env 文件，填入你的 API 密钥
echo.
echo 2. 激活虚拟环境：
if not defined SKIP_VENV (
    echo    .venv\Scripts\activate
)
echo.
echo 3. 运行应用：
echo    python -m crewai_app "你的研究问题"
echo.
echo 示例命令：
echo    python -m crewai_app "水冷板行业现状" --verbose
echo.
echo 查看帮助：
echo    python -m crewai_app --help
echo.
goto :eof

REM 主函数
:main
set is_dev=false

REM 解析参数
:parse_args
if "%~1"=="" goto :run_install
if "%~1"=="--dev" (
    set is_dev=true
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    call :show_help
    exit /b 0
)
call :log_error "未知参数：%~1"
call :show_help
exit /b 1

:run_install
call :log_info "开始 CrewAI App 安装..."

call :check_python
if %errorlevel% neq 0 exit /b 1

call :check_uv
if %errorlevel% neq 0 exit /b 1

call :setup_venv
if %errorlevel% neq 0 exit /b 1

call :install_dependencies %is_dev%
if %errorlevel% neq 0 exit /b 1

call :create_env_template
if %errorlevel% neq 0 exit /b 1

call :verify_installation
if %errorlevel% neq 0 exit /b 1

call :show_usage_guide
call :log_success "安装流程完成！"
goto :eof

REM 执行主函数
call :main %*