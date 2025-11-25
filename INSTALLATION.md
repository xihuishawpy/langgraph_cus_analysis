# CrewAI App 环境安装指南

## 项目概述
CrewAI App 是一个基于 CrewAI 的多智能体分析系统，用于企业研究和知识库分析。

## 环境要求
- Python 3.12+
- uv 包管理器
- Git

## 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/xihuishawpy/langgraph_cus_analysis.git
cd langgraph_cus_analysis
git checkout feature/crewai_app
```

### 2. 创建虚拟环境
```bash
# 创建 Python 虚拟环境
uv venv .venv

# 激活虚拟环境 (macOS/Linux)
source .venv/bin/activate

# 激活虚拟环境 (Windows)
.venv\Scripts\activate
```

### 3. 安装依赖包

#### 使用国内镜像源（推荐 - 速度快）
```bash
# 使用清华源安装
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  crewai \
  langchain-core \
  langchain-community \
  pydantic \
  python-dotenv \
  tavily-python \
  pyyaml \
  faiss-cpu \
  numpy \
  pandas \
  dashscope \
  sentence-transformers
```

#### 使用默认源（较慢）
```bash
uv pip install \
  crewai \
  langchain-core \
  langchain-community \
  pydantic \
  python-dotenv \
  tavily-python \
  pyyaml \
  faiss-cpu \
  numpy \
  pandas \
  dashscope \
  sentence-transformers
```

### 4. 验证安装
```bash
# 测试核心依赖
uv run python -c "
import crewai
import langchain_core
import pydantic
import faiss
import numpy as np
import pandas as pd
import dashscope
import sentence_transformers
print('✅ 所有依赖包安装成功！')
"

# 测试项目模块
uv run python -c "
from crewai_app.main import main
from crewai_app.crew_builder import ProSearchCrewBuilder
from crewai_app.configuration import Configuration
print('✅ CrewAI App 模块导入成功！')
"
```

## 核心依赖包列表

| 包名 | 版本 | 用途 |
|------|------|------|
| crewai | 1.6.0 | 多智能体框架 |
| langchain-core | 1.1.0 | LangChain 核心模块 |
| langchain-community | 0.4.1 | LangChain 社区扩展 |
| pydantic | 2.12.4 | 数据验证 |
| python-dotenv | 1.2.1 | 环境变量管理 |
| tavily-python | 0.7.13 | 网络搜索工具 |
| pyyaml | 6.0.3 | YAML 配置文件解析 |
| faiss-cpu | 1.13.0 | 向量检索 |
| numpy | 2.3.5 | 数值计算 |
| pandas | 2.3.3 | 数据处理 |
| dashscope | 1.25.2 | 通义千问 SDK |
| sentence-transformers | 5.1.2 | 文本嵌入 |

## 环境变量配置

创建 `.env` 文件：
```bash
# 通义千问 API Key
DASHSCOPE_API_KEY=your_dashscope_api_key

# Tavily 搜索 API Key (可选)
TAVILY_API_KEY=your_tavily_api_key
```

## 使用方式

```bash
# 运行 CrewAI App
uv run python -m crewai_app "研究主题或问题"

# 查看帮助
uv run python -m crewai_app --help

# 示例命令
uv run python -m crewai_app "水冷板行业现状" --verbose
```

## 安装时间参考
- 使用清华源：~3-5 分钟
- 使用默认源：~10-20 分钟（网络状况影响较大）

## 常见问题

### 1. uv 命令不存在
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或使用 pip
pip install uv
```

### 2. 网络连接问题
- 使用国内镜像源：`https://pypi.tuna.tsinghua.edu.cn/simple`
- 配置代理：`export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. 内存不足
- 如果内存小于 8GB，建议关闭其他应用程序
- 可以考虑使用云服务器进行安装

## 项目结构
```
crewai_app/
├── __init__.py
├── __main__.py
├── main.py              # 主入口
├── crew_builder.py      # CrewAI 构建器
├── configuration.py     # 配置管理
├── tools.py            # 工具函数
├── llm.py              # LLM 封装
├── knowledge_base.py   # 知识库
├── prompts.py          # 提示词
├── schemas.py          # 数据模型
├── query_patterns.py   # 查询模式
├── utils.py            # 工具函数
└── config/
    └── agents.yaml     # Agent 配置
```

## 更新依赖
```bash
# 更新单个包
uv pip install --upgrade package_name

# 更新所有包
uv pip list --outdated
uv pip install --upgrade package1 package2 ...
```

## 卸载环境
```bash
# 删除虚拟环境
rm -rf .venv

# 清理缓存
uv cache clean
```