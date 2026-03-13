# config_data.py - 整合所有配置，无需.env文件
from datetime import datetime
import os

# ===================== 核心路径配置 =====================
# MD5去重文件路径
md5_path = "./md5.text"
# 向量库集合名称
collection_name = "rag"
# 向量库本地持久化路径
persist_directory = "./chroma_db"
# 对话历史存储路径
chat_history_dir = "./chat_history"

# ===================== 检索配置 =====================
# 仅保留top_k：返回Top-N个相关片段
top_k = 3  # 推荐值：3~5，数量过多会增加大模型处理时间

# ===================== 文本分割配置 =====================
# 基础分割大小（入库时用）
chunk_size = 800
chunk_overlap = 150
# 窗口分割大小（检索时扩展上下文用，通常为基础的2倍）
window_chunk_size = chunk_size * 2
window_chunk_overlap = chunk_overlap * 2
# 分割分隔符（按优先级）
separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# ===================== 模型配置 =====================
# 阿里云通义千问API密钥（★ 请替换为你的实际密钥 ★）
DASHSCOPE_API_KEY = "sk-f8f8b8e1132242fda1fc0fc15d1e5516"
# 嵌入模型名称（文本转向量）
embedding_model_name = "text-embedding-v4"
# 对话模型名称（生成回答）
chat_model_name = "qwen3-max"

# ===================== 业务配置 =====================
# 操作员名称（入库元数据用）
OPERATOR_NAME = "高进"
# RAG提示词模板（可根据需求修改）
PROMPT_TEMPLATE = """以我提供的已知参考资料为主，简洁和专业的回答用户问题。
参考资料:{context}。并且我提供用户的对话历史记录，如下："""

# ===================== 会话配置 =====================
# 动态session_id（运行时填充，无需修改）
session_config = {
    "configurable": {
        "session_id": None,
    }
}

# ===================== 工具函数（可选） =====================
def get_api_key():
    """获取API密钥（优先从环境变量读取，无则用配置值）"""
    return os.getenv("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY)

def get_operator_name():
    """获取操作员名称"""
    return os.getenv("OPERATOR_NAME", OPERATOR_NAME)

def get_prompt_template():
    """获取提示词模板"""
    return os.getenv("PROMPT_TEMPLATE", PROMPT_TEMPLATE)

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")