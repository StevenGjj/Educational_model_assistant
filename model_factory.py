# model_factory.py - 模型工厂，支持切换不同的大语言模型
# 注意：向量嵌入始终使用通义千问的 text-embedding-v4
# 仅对话模型可以在通义千问和MiniMax之间切换
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
import os
import config_data as config
from logger import logger


class MiniMaxChat:
    """MiniMax 大语言模型封装（使用官方API）"""
    
    def __init__(self, model_name="MiniMax-M2.7", api_key=None):
        self.model_name = model_name
        # 优先从环境变量读取API密钥
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"MiniMax API密钥未设置。请在PowerShell中执行: $env:MINIMAX_API_KEY='你的密钥'")
        
        logger.info(f"MiniMax API Key loaded: {self.api_key[:10]}...")
        
        self.base_url = "https://api.minimax.chat/v1"
    
    def __call__(self, messages):
        """实现兼容 LangChain 的调用方式"""
        
        # 将 LangChain 的 messages 格式转换为 MiniMax 格式
        minimax_messages = []
        system_prompt = ""
        
        for msg in messages:
            # 提取 system 消息作为系统提示词
            if hasattr(msg, 'type') and msg.type == 'system':
                system_prompt = msg.content
            elif isinstance(msg, dict) and msg.get('type') == 'system':
                system_prompt = msg.get('content', '')
            else:
                # 其他消息格式
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    role = msg.type
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get('type', 'user')
                    content = msg.get('content', '')
                else:
                    role = 'user'
                    content = str(msg)
                
                # MiniMax 消息格式
                minimax_messages.append({
                    "role": role,
                    "content": content
                })
        
        # 使用 requests 调用 MiniMax API
        try:
            import requests
            
            url = f"{self.base_url}/text/chatcompletion_pro"
            
            # 尝试不带Bearer前缀
            headers = {
                "Authorization": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": system_prompt}] + minimax_messages if system_prompt else minimax_messages,
                "max_tokens": 4096,
                "temperature": 0.7
            }
            
            logger.info(f"Calling MiniMax API: {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            logger.info(f"Response status: {response.status_code}")
            
            result = response.json()
            logger.info(f"Response body: {result}")
            
            # 提取文本内容
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            elif "base_resp" in result and result["base_resp"]["status_code"] != 0:
                raise RuntimeError(f"MiniMax API error: {result['base_resp']['status_msg']}")
            else:
                raise RuntimeError(f"MiniMax API 返回格式异常: {result}")
        except Exception as e:
            logger.error(f"MiniMax API 调用失败: {str(e)}")
            raise


def get_chat_model(model_name: str):
    """
    获取对话模型实例
    :param model_name: 模型名称（如 qwen3-max, MiniMax-M2.7）
    :return: 兼容 LangChain 的聊天模型实例
    """
    if model_name == "MiniMax-M2.7" or model_name == "MiniMax M2.7":
        logger.info(f"创建MiniMax对话模型实例: {model_name}")
        return MiniMaxChat(model_name=model_name)
    else:
        logger.info(f"创建通义千问对话模型实例: {model_name}")
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