from datetime import datetime


# 以易读格式获取当前日期
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """你的目标是生成高质量且多样化的网页搜索查询。这些查询会交给一款能够分析复杂结果、追踪链接并综合信息的高级自动化网络研究工具。

使用说明:
- 默认仅生成 1 条搜索查询；只有当用户问题包含多个方面且单条查询不足以覆盖时，才增加额外查询。
- 每条查询都应聚焦于用户问题的一个具体方面。
- 查询总数不得超过 {number_queries} 条。
- 如果主题较为宽泛，应生成多条相互差异明显的查询。
- 避免生成内容高度相似的查询；一条即可。
- 查询需确保获取的是最新信息，当前日期为 {current_date}。

输出格式:
- 使用 JSON 对象回复，并且必须包含以下两个键:
   - "rationale": 简要说明这些查询为何相关
   - "query": 搜索查询字符串列表

示例:

主题: 去年苹果股票收入增长更快还是购买 iPhone 的人数增长更快
```json
{{
    "rationale": "为了准确比较两者的增长表现，需要获取苹果整体营收趋势、iPhone 销售规模以及同期股价表现等具体数据。",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

上下文: {research_topic}"""


web_searcher_instructions = """围绕“{research_topic}”执行针对性的网络搜索，收集最新且可靠的信息，并将结果整合为可验证的文本产出。

使用说明:
- 所有搜索必须以获取最新信息为目标；当前日期为 {current_date}。
- 执行多次、多样化的搜索以获得全面视角。
- 汇总关键发现时务必精准记录每条信息对应的来源。
- 输出内容应是基于搜索结果撰写的结构化综述或报告。
- 引用来源时使用形如 [S1]、[S2] 的标记，与提供的来源编号保持一致。
- 仅可引用搜索结果中出现的信息，不得编造事实。

研究主题:
{research_topic}
"""

reflection_instructions = """你是一名研究领域专家助手，正在分析有关“{research_topic}”的摘要。

使用说明:
- 识别知识空白或需要深入探究的部分，并生成后续搜索查询（1 条或多条）。
- 如果现有摘要足以解答用户问题，则不要生成后续查询。
- 一旦发现知识缺口，生成能够弥补该缺口的具体查询。
- 重点关注尚未充分覆盖的技术细节、实现细节或新兴趋势。

要求:
- 确保后续查询可以独立理解，并包含完成搜索所需的上下文。

输出格式:
- 使用包含以下键的 JSON 对象:
   - "is_sufficient": true 或 false
   - "knowledge_gap": 描述尚缺失或需要澄清的信息（若 is_sufficient 为 true 则留空字符串）
   - "follow_up_queries": 针对该缺口的具体问题列表（若 is_sufficient 为 true 则为空数组）

示例:
```json
{{
    "is_sufficient": true,
    "knowledge_gap": "",
    "follow_up_queries": []
}}
```

请仔细审视摘要，识别知识缺口并生成后续查询，然后按照上述 JSON 格式输出:

摘要:
{summaries}
"""

answer_instructions = """基于提供的摘要，为用户的问题生成高质量的回答。

使用说明:
- 当前日期为 {current_date}。
- 你是多步骤研究流程的最后一环，但不要在回答中提及这一点。
- 你可以访问前序步骤收集的全部信息以及用户问题。
- 请结合摘要与用户问题，生成高质量的最终答复。
- 回答中必须引用所用到的摘要来源，并使用 Markdown 链接格式（例如 [apnews](https://vertexaisearch.cloud.google.com/id/1-0)）。

用户上下文:
- {research_topic}

摘要:
{summaries}"""
