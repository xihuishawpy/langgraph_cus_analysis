#!/bin/bash
# CrewAI App 安全运行脚本 (临时解决 OpenMP 冲突)

# 设置 OpenMP 环境变量（临时解决方案）
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# 激活虚拟环境
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

echo "🚀 运行 CrewAI App (OpenMP 安全模式)"
echo "⚠️  注意：这是一个临时解决方案，建议运行 scripts/fix-openmp.sh 获得永久修复"
echo ""

# 运行 CrewAI App
python -m crewai_app "$@"