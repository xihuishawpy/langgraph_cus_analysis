#!/bin/bash
# Docker 环境设置脚本
# 使用方法: bash scripts/docker-setup.sh [prod|dev|test|jupyter]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    echo "Docker 环境设置脚本"
    echo ""
    echo "使用方法:"
    echo "  bash scripts/docker-setup.sh [环境] [选项]"
    echo ""
    echo "环境:"
    echo "  prod     生产环境"
    echo "  dev      开发环境"
    echo "  test     测试环境"
    echo "  jupyter  Jupyter Lab 开发环境"
    echo ""
    echo "选项:"
    echo "  --build    强制重新构建镜像"
    echo "  --clean    清理旧的容器和镜像"
    echo "  --help     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash scripts/docker-setup.sh prod --build"
    echo "  bash scripts/docker-setup.sh dev"
    echo "  bash scripts/docker-setup.sh jupyter"
}

# 检查 Docker 和 Docker Compose
check_docker() {
    log_info "检查 Docker 环境..."

    if ! command -v docker &> /dev/null; then
        log_error "未找到 Docker，请先安装 Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "未找到 Docker Compose，请先安装 Docker Compose"
        exit 1
    fi

    # 检查 Docker 服务是否运行
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行，请启动 Docker"
        exit 1
    fi

    log_success "Docker 环境检查通过"
}

# 清理旧的容器和镜像
clean_docker() {
    log_info "清理旧的 Docker 资源..."

    # 停止并删除相关容器
    docker-compose down --remove-orphans 2>/dev/null || true

    # 删除相关镜像
    docker images | grep crewai-app | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true

    log_success "清理完成"
}

# 检查配置文件
check_config() {
    log_info "检查配置文件..."

    if [[ ! -f ".env" ]]; then
        log_warning ".env 文件不存在，创建默认配置..."
        cat > .env << EOF
# 通义千问 API Key (必需)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# Tavily 搜索 API Key (可选)
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
        log_warning "请编辑 .env 文件并填入正确的 API 密钥"
    else
        log_success ".env 文件存在"
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."

    mkdir -p data
    mkdir -p logs
    mkdir -p .kb_cache

    log_success "目录创建完成"
}

# 设置生产环境
setup_production() {
    local build_flag=$1
    log_info "设置生产环境..."

    if [[ "$build_flag" == "true" ]]; then
        docker-compose build --target production crewai-app
    fi

    docker-compose up -d crewai-app

    log_success "生产环境启动完成"
    log_info "查看日志: docker-compose logs -f crewai-app"
    log_info "停止服务: docker-compose down crewai-app"
}

# 设置开发环境
setup_development() {
    local build_flag=$1
    log_info "设置开发环境..."

    if [[ "$build_flag" == "true" ]]; then
        docker-compose build --target development crewai-app-dev
    fi

    docker-compose --profile dev up -d crewai-app-dev

    log_success "开发环境启动完成"
    log_info "进入开发容器: docker-compose exec crewai-app-dev bash"
    log_info "查看日志: docker-compose logs -f crewai-app-dev"
    log_info "停止服务: docker-compose --profile dev down"
}

# 设置测试环境
setup_testing() {
    local build_flag=$1
    log_info "设置测试环境..."

    if [[ "$build_flag" == "true" ]]; then
        docker-compose build --target testing crewai-app-test
    fi

    log_info "运行测试..."
    docker-compose --profile test up --build crewai-app-test

    log_success "测试完成"
    log_info "查看测试结果: docker-compose logs crewai-app-test"
}

# 设置 Jupyter 环境
setup_jupyter() {
    local build_flag=$1
    log_info "设置 Jupyter Lab 开发环境..."

    if [[ "$build_flag" == "true" ]]; then
        docker-compose build --target development jupyter-dev
    fi

    docker-compose --profile jupyter up -d jupyter-dev

    log_success "Jupyter Lab 启动完成"
    log_info "访问地址: http://localhost:8888"
    log_info "进入容器: docker-compose exec jupyter-dev bash"
    log_info "查看日志: docker-compose logs -f jupyter-dev"
    log_info "停止服务: docker-compose --profile jupyter down"
}

# 显示运行状态
show_status() {
    log_info "Docker 容器状态:"
    docker-compose ps
}

# 主函数
main() {
    local environment=""
    local should_build=false
    local should_clean=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            prod|dev|test|jupyter)
                environment="$1"
                shift
                ;;
            --build)
                should_build=true
                shift
                ;;
            --clean)
                should_clean=true
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

    if [[ -z "$environment" ]]; then
        log_error "请指定环境 (prod|dev|test|jupyter)"
        show_help
        exit 1
    fi

    log_info "开始设置 $environment 环境..."

    check_docker

    if [[ "$should_clean" == "true" ]]; then
        clean_docker
    fi

    check_config
    create_directories

    case "$environment" in
        prod)
            setup_production "$should_build"
            ;;
        dev)
            setup_development "$should_build"
            ;;
        test)
            setup_testing "$should_build"
            ;;
        jupyter)
            setup_jupyter "$should_build"
            ;;
        *)
            log_error "未知环境：$environment"
            exit 1
            ;;
    esac

    show_status
    log_success "环境设置完成！"
}

# 错误处理
trap 'log_error "设置过程中发生错误"; exit 1' ERR

# 执行主函数
main "$@"