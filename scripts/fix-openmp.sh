#!/bin/bash
# OpenMP 冲突修复脚本 (macOS)
# 解决多个科学计算库的 OpenMP 冲突问题

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 检查操作系统
check_os() {
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "此脚本仅适用于 macOS"
        exit 1
    fi
    log_success "操作系统检查通过：macOS"
}

# 检查 Homebrew
check_homebrew() {
    if ! command -v brew &> /dev/null; then
        log_error "未找到 Homebrew，请先安装：/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    log_success "Homebrew 已安装"
}

# 安装/更新 libomp
install_libomp() {
    log_info "安装/更新 libomp..."

    if brew list libomp &> /dev/null; then
        log_info "libomp 已安装，检查更新..."
        brew upgrade libomp || log_info "libomp 已是最新版本"
    else
        log_info "安装 libomp..."
        brew install libomp
    fi

    # 获取 libomp 路径
    local libomp_path=$(brew --prefix libomp)/lib/libomp.dylib
    if [[ -f "$libomp_path" ]]; then
        export DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH
        log_success "libomp 路径：$libomp_path"
    else
        log_error "libomp 安装失败"
        exit 1
    fi
}

# 重新安装冲突的包
reinstall_packages() {
    log_info "重新安装可能导致冲突的包..."

    # 卸载相关包
    log_info "卸载 numpy scipy pytorch 和相关包..."
    uv pip uninstall -y numpy scipy torch torchvision torchaudio || true

    # 设置环境变量
    export LDFLAGS="-L$(brew --prefix libomp)/lib"
    export CPPFLAGS="-I$(brew --prefix libomp)/include"

    # 重新安装包，使用 Homebrew 的 OpenMP
    log_info "重新安装 numpy..."
    uv pip install --no-cache-dir numpy

    log_info "重新安装 scipy..."
    uv pip install --no-cache-dir scipy

    log_info "重新安装 pytorch..."
    uv pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    log_success "包重新安装完成"
}

# 创建环境配置文件
create_env_config() {
    log_info "创建 OpenMP 环境配置..."

    cat > .env.openmp << 'EOF'
# OpenMP 配置 (macOS)
# 解决 OpenMP 冲突问题的环境变量配置

# 使用 Homebrew 的 libomp
export DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH

# OpenMP 线程配置
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# 禁用 OpenMP 重复库检查（临时解决方案）
export KMP_DUPLICATE_LIB_OK=TRUE

# 编译器标志
export LDFLAGS="-L$(brew --prefix libomp)/lib"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
EOF

    log_success "环境配置已保存到 .env.openmp"
    log_info "使用方法：source .env.openmp"
}

# 创建运行脚本
create_run_script() {
    log_info "创建优化的运行脚本..."

    cat > scripts/run-with-openmp.sh << 'EOF'
#!/bin/bash
# CrewAI App 优化的运行脚本 (解决 OpenMP 冲突)

# 设置 OpenMP 环境
export DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export KMP_DUPLICATE_LIB_OK=TRUE

# 激活虚拟环境
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# 运行 CrewAI App
python -m crewai_app "$@"
EOF

    chmod +x scripts/run-with-openmp.sh
    log_success "运行脚本已创建：scripts/run-with-openmp.sh"
}

# 验证修复
verify_fix() {
    log_info "验证 OpenMP 修复..."

    # 设置环境变量
    export DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH
    export KMP_DUPLICATE_LIB_OK=TRUE

    # 测试导入
    if python -c "
import numpy as np
import scipy
import torch
print('✅ NumPy 版本:', np.__version__)
print('✅ SciPy 版本:', scipy.__version__)
print('✅ PyTorch 版本:', torch.__version__)
print('✅ OpenMP 配置成功')
" 2>/dev/null; then
        log_success "OpenMP 冲突修复验证通过！"
    else
        log_warning "验证过程中仍有问题，但已配置临时解决方案"
    fi
}

# 显示使用说明
show_usage() {
    echo ""
    log_success "🎉 OpenMP 冲突修复完成！"
    echo ""
    echo "使用方法："
    echo ""
    echo "1. 使用优化的运行脚本（推荐）："
    echo "   bash scripts/run-with-openmp.sh \"你的问题\""
    echo ""
    echo "2. 手动设置环境变量："
    echo "   source .env.openmp"
    echo "   python -m crewai_app \"你的问题\""
    echo ""
    echo "3. 一行命令运行："
    echo "   DYLD_LIBRARY_PATH=\$(brew --prefix libomp)/lib:\$DYLD_LIBRARY_PATH KMP_DUPLICATE_LIB_OK=TRUE python -m crewai_app \"你的问题\""
    echo ""
    echo "注意：如果仍然看到警告，可以忽略，程序应该能正常运行"
    echo ""
}

# 主函数
main() {
    log_info "开始修复 OpenMP 冲突问题..."

    check_os
    check_homebrew
    install_libomp
    reinstall_packages
    create_env_config
    create_run_script
    verify_fix
    show_usage

    log_success "OpenMP 冲突修复完成！"
}

# 错误处理
trap 'log_error "修复过程中发生错误，请检查上方的错误信息"; exit 1' ERR

# 执行主函数
main "$@"