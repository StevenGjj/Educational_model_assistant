# model_factory.py - 模型工厂，支持切换不同的大语言模型
# 注意：向量嵌入始终使用通义千问的 text-embedding-v4
# 仅对话模型可以在通义千问和MiniMax之间切换
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
import os
import config_data as config


class MiniMaxChat:
    """MiniMax 大语言模型封装（兼容 LangChain 接口）"""
    
    def __init__(self, model_name="MiniMax-M2.7", api_key=None):
        self.model_name = model_name
        # 优先从环境变量读取API密钥
        env_key = os.getenv("MINIMAX_API_KEY")
        self.api_key = api_key or env_key
        
        if not self.api_key:
            # 提供更详细的错误信息
            raise ValueError(f"MiniMax API密钥未设置。请在PowerShell中执行: $env:MINIMAX_API_KEY='你的密钥'")
    
    def __call__(self, messages):
        """实现兼容 LangChain 的调用方式"""
        import requests
        
        # 将 LangChain 的 messages 格式转换为 MiniMax 格式
        minimax_messages = []
        for msg in messages:
            # 处理 LangChain 消息对象 - 支持多种格式
            try:
                # 情况1: LangChain Message 对象 (AIMessage, HumanMessage, SystemMessage)
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = msg.type
                    content = msg.content
                # 情况2: 字典格式
                elif isinstance(msg, dict):
                    role = msg.get('type', 'user')
                    content = msg.get('content', '')
                # 情况3: 元组格式 (type, content)
                elif isinstance(msg, (tuple, list)):
                    if len(msg) >= 2:
                        role = msg[0] if isinstance(msg[0], str) else 'user'
                        content = str(msg[1]) if len(msg) > 1 else ''
                    else:
                        continue
                # 情况4: 其他情况
                else:
                    role = 'user'
                    content = str(msg)
            except Exception as e:
                # 如果解析失败，使用默认值
                role = 'user'
                content = str(msg) if msg else ''
            
            # MiniMax 角色映射 (user/assistant/system)
            minimax_messages.append({
                "role": role,
                "content": content
            })
        
        url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
        
        # 确保 API 密钥正确格式化
        auth_key = self.api_key.strip()
        
        headers = {
            "Authorization": f"Bearer {auth_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": minimax_messages
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            return content
        else:
            raise RuntimeError(f"MiniMax API调用失败: {result}")


def get_chat_model(model_name: str):
    """
    获取对话模型实例
    :param model_name: 模型名称（如 qwen3-max, MiniMax-M2.7）
    :return: 兼容 LangChain 的聊天模型实例
    """
    if model_name == "MiniMax-M2.7":
        return MiniMaxChat(model_name=model_name)
    else:
        # 默认使用通义千问
        return ChatTongyi(model=model_name)


def get_embeddings_model(model_name: str = None):
    """
    获取嵌入模型实例
    注意：始终使用通义千问的嵌入模型，确保向量库兼容性
    :param model_name: 嵌入模型名称（忽略，仅使用配置中的默认模型）
    :return: 嵌入模型实例
    """
    # 始终使用通义千问嵌入模型，保证向量库共用
    return DashScopeEmbeddings(model=config.embedding_model_name)