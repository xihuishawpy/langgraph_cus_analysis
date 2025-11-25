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
