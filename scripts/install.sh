#!/bin/bash
# CrewAI App 自动化安装脚本 (macOS/Linux)
# 使用方法: bash scripts/install.sh [--dev]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "CrewAI App 安装脚本"
    echo ""
    echo "使用方法:"
    echo "  bash scripts/install.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --dev     安装开发环境依赖"
    echo "  --help    显示此帮助信息"
    echo ""
    echo "环境变量:"
    echo "  UV_INDEX_URL    指定 PyPI 镜像源 (默认: 清华源)"
    echo "  SKIP_VENV       跳过虚拟环境创建 (设置为 '1' 时跳过)"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 版本比较
version_compare() {
    local version1=$1
    local version2=$2
    if [[ $(printf '%s\n' "$version1" "$version2" | sort -V | head -n1) == "$version2" ]]; then
        return 0  # version1 >= version2
    else
        return 1  # version1 < version2
    fi
}

# 检查 Python 版本
check_python() {
    log_info "检查 Python 版本..."

    if ! command_exists python3; then
        log_error "未找到 Python3，请先安装 Python 3.12 或更高版本"
        exit 1
    fi

    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local required_version="3.12"

    if ! version_compare "$python_version" "$required_version"; then
        log_error "Python 版本过低，需要 >=3.12，当前版本：$python_version"
        exit 1
    fi

    log_success "Python 版本检查通过：$python_version"
}

# 检查 uv
check_uv() {
    log_info "检查 uv 包管理器..."

    if ! command_exists uv; then
        log_warning "未找到 uv，开始安装..."
        if command_exists curl; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            log_error "需要 curl 来安装 uv，请先安装 curl 或手动安装 uv"
            exit 1
        fi

        # 重新检查
        if ! command_exists uv; then
            log_error "uv 安装失败"
            exit 1
        fi
    fi

    local uv_version=$(uv --version)
    log_success "uv 版本：$uv_version"
}

# 设置虚拟环境
setup_venv() {
    if [[ "${SKIP_VENV}" == "1" ]]; then
        log_info "跳过虚拟环境创建 (SKIP_VENV=1)"
        return
    fi

    log_info "创建虚拟环境..."

    if [[ -d ".venv" ]]; then
        log_warning "发现已存在的虚拟环境，将重新创建..."
        rm -rf .venv
    fi

    uv venv .venv --python 3.12
    log_success "虚拟环境创建完成"
}

# 激活虚拟环境
activate_venv() {
    if [[ "${SKIP_VENV}" != "1" ]]; then
        log_info "激活虚拟环境..."
        source .venv/bin/activate
        log_success "虚拟环境已激活"
    fi
}

# 安装依赖
install_dependencies() {
    local is_dev=$1
    local requirements_file="requirements.txt"

    if [[ "$is_dev" == "true" ]]; then
        requirements_file="requirements-dev.txt"
        log_info "安装开发环境依赖..."
    else
        log_info "安装生产环境依赖..."
    fi

    # 设置镜像源
    local index_url="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    log_info "使用镜像源：$index_url"

    # 升级 uv pip
    log_info "升级 uv pip..."
    uv pip install --upgrade pip

    # 安装依赖
    log_info "从 $requirements_file 安装依赖..."
    uv pip install --index-url "$index_url" -r "$requirements_file"

    log_success "依赖安装完成"
}

# 创建配置文件模板
create_env_template() {
    if [[ ! -f ".env" ]]; then
        log_info "创建 .env 配置文件模板..."
        cat > .env << 'EOF'
# CrewAI App 环境配置
# 复制此文件并填入实际的 API 密钥

# 通义千问 API Key (必需)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# Tavily 搜索 API Key (可选，用于网络搜索)
TAVILY_API_KEY=your_tavily_api_key_here

# 应用配置
LOG_LEVEL=INFO
MAX_RESEARCH_LOOPS=2
ENABLE_KNOWLEDGE_BASE_SEARCH=true
ENABLE_INDUSTRY_REPORT_MODE=true

# 知识库配置
KNOWLEDGE_BASE_PATHS=eastmoney_concept_constituents.xlsx,sw_third_industry_constituents.xlsx
KNOWLEDGE_BASE_TOP_K=3
KNOWLEDGE_BASE_EMBEDDING_MODEL=text-embedding-v3
KNOWLEDGE_BASE_EMBEDDING_BACKEND=dashscope
KNOWLEDGE_BASE_EMBEDDING_BATCH_SIZE=10

# 模型配置
QUERY_GENERATOR_MODEL=qwen-plus
REFLECTION_MODEL=qwen-plus
ANSWER_MODEL=qwen-plus
LLM_BACKEND=dashscope
EOF
        log_success ".env 文件模板已创建，请编辑并填入实际的 API 密钥"
    else
        log_info ".env 文件已存在，跳过创建"
    fi
}

# 验证安装
verify_installation() {
    log_info "验证安装..."

    local failed_imports=()

    # 测试核心依赖
    local packages=("crewai" "langchain_core" "pydantic" "faiss" "numpy" "pandas" "dashscope")

    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            echo "  ✓ $package"
        else
            echo "  ✗ $package"
            failed_imports+=("$package")
        fi
    done

    # 测试项目模块
    if python -c "from crewai_app.main import main" 2>/dev/null; then
        echo "  ✓ crewai_app 模块"
    else
        echo "  ✗ crewai_app 模块"
        failed_imports+=("crewai_app")
    fi

    if [[ ${#failed_imports[@]} -eq 0 ]]; then
        log_success "所有依赖验证通过！"
    else
        log_error "以下依赖验证失败：${failed_imports[*]}"
        exit 1
    fi
}

# 显示使用指南
show_usage_guide() {
    echo ""
    log_success "🎉 CrewAI App 安装完成！"
    echo ""
    echo "使用指南："
    echo "1. 配置环境变量："
    echo "   编辑 .env 文件，填入你的 API 密钥"
    echo ""
    echo "2. 激活虚拟环境："
    if [[ "${SKIP_VENV}" != "1" ]]; then
        echo "   source .venv/bin/activate"
    fi
    echo ""
    echo "3. 运行应用："
    echo "   python -m crewai_app \"你的研究问题\""
    echo ""
    echo "示例命令："
    echo "   python -m crewai_app \"水冷板行业现状\" --verbose"
    echo ""
    echo "查看帮助："
    echo "   python -m crewai_app --help"
    echo ""
}

# 主函数
main() {
    local is_dev=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                is_dev=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数：$1"
                show_help
                exit 1
                ;;
        esac
    done

    # 开始安装流程
    log_info "开始 CrewAI App 安装..."

    check_python
    check_uv
    setup_venv
    activate_venv
    install_dependencies "$is_dev"
    create_env_template
    verify_installation
    show_usage_guide

    log_success "安装流程完成！"
}

# 错误处理
trap 'log_error "安装过程中发生错误，请检查上方的错误信息"; exit 1' ERR

# 执行主函数
main "$@"