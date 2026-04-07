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
DASHSCOPE_API_KEY = "请将此处替换为你的阿里云通义千问API密钥并添加进环境变量中"
# 嵌入模型名称（文本转向量）- 用于向量检索和入库
embedding_model_name = "text-embedding-v4"
# 对话模型名称（生成回答）- 用于答案生成
chat_model_name = "qwen3-max"

# ===================== MiniMax 模型配置（新增）====================
# MiniMax API密钥（请替换为你的实际密钥）
MINIMAX_API_KEY = "请替换为你的MiniMax API密钥"
# MiniMax 对话模型（仅用于生成回答，不影响向量检索）
MINIMAX_CHAT_MODEL = "MiniMax-M2.7"

# ===================== 模型切换配置 =====================
# 支持的模型列表
# 注意：切换模型仅改变"答案生成模型"，向量检索始终使用 embedding_model_name
AVAILABLE_MODELS = {
    "通义千问": {
        "chat_model_name": "qwen3-max",
    },
    "MiniMax M2.7": {
        "chat_model_name": "MiniMax-M2.7",
    }
}

# ===================== 业务配置 =====================
# 操作员名称（入库元数据用）
OPERATOR_NAME = "高进"
# RAG提示词模板（可根据需求修改）

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

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")