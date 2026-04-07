from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_message_histories import FileChatMessageHistory
import os
from prompts import *
from logger import logger
from model_factory import get_chat_model, get_embeddings_model


def print_prompt(prompt):
    """调试用：打印拼接后的提示词（可注释）"""
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)

    return prompt


class RagService(object):
    def __init__(self, model_name=None):
        """
        初始化RAG服务
        :param model_name: 模型名称，默认使用配置文件中的模型
        注意：向量嵌入始终使用通义千问 text-embedding-v4，确保向量库共用
        """
        # 获取对话模型配置
        if model_name and model_name in config.AVAILABLE_MODELS:
            model_config = config.AVAILABLE_MODELS[model_name]
            self.current_model_name = model_config["chat_model_name"]
        else:
            self.current_model_name = config.chat_model_name
        
        # 嵌入模型始终使用通义千问（确保向量库共用）
        self.embedding = get_embeddings_model()
        self.vector_service = VectorStoreService(embedding=self.embedding)
        self.retriever = self.vector_service.get_retriever()
        self.chat_model = get_chat_model(self.current_model_name)
        self.parser = StrOutputParser()  # 统一输出解析器
        # 初始化4条核心链
        self.qa_chain = self._get_qa_chain()  # 原有知识问答链
        self.doc_match_chain = self._get_doc_match_chain()  # 文档匹配度分析链
        self.gen_question_chain = self._get_gen_question_chain()  # 自动出题链
        self.gen_criteria_chain = self._get_gen_criteria_chain()  # 评分标准生成链
        # 新增：代码相关链
        self.code_question_chain = self._get_code_question_chain()  # 代码出题链
        self.code_analysis_chain = self._get_code_analysis_chain()  # 代码分析链
        logger.info(f"RAG服务初始化成功，当前使用模型: {self.current_model_name}")

    def _get_qa_chain(self):
        """原有知识问答链（保留历史对话+检索上下文）"""

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            return {
                "input": value["input"],
                "context": value["context"],
                "history": value.get("history", [])
            }

        base_chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnableLambda(format_for_retriever) | self.retriever | format_document
                }
                | RunnableLambda(format_for_prompt_template)
                | ChatPromptTemplate.from_messages([
            ("system", QA_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "请回答用户提问：{input}")
        ])
                | print_prompt
                | self.chat_model
                | self.parser
        )
        # 绑定对话历史
        conversation_chain = RunnableWithMessageHistory(
            base_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

    def _get_doc_match_chain(self):
        """文档匹配度分析链（课件↔教材）"""
        chain = (
                ChatPromptTemplate.from_template(DOC_MATCH_PROMPT)  # 用模板绑定参数
                | print_prompt
                | self.chat_model
                | self.parser
        )
        return chain

    def _get_gen_question_chain(self):
        """自动生成题目链（基于检索的知识库内容）"""

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关学习资料"
            return "\n".join([doc.page_content for doc in docs])

        def format_for_prompt_template(value):
            return {
                "input": value["input"],
                "context": value["context"],
            }

        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnablePassthrough() | self.retriever | format_document
                } | RunnableLambda(format_for_prompt_template)
                | ChatPromptTemplate.from_messages([
            ("system", GEN_QUESTION_PROMPT),
            ("user", "根据用户要求生成题目：{input}")
        ])
                | print_prompt
                | self.chat_model
                | self.parser
        )
        return chain

    def _get_gen_criteria_chain(self):
        """评分标准生成链（基于知识库+用户题目）"""

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            return "\n".join([doc.page_content for doc in docs])

        def format_for_prompt_template(value):
            return {
                "input": value["input"],
                "context": value["context"],
            }

        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnablePassthrough() | self.retriever | format_document
                } | RunnableLambda(format_for_prompt_template)
                | ChatPromptTemplate.from_messages([
            ("system", GEN_CRITERIA_PROMPT),
            ("user", "给用户上传的题目：'{input}'生成标准答案与评分依据")
        ])
                | print_prompt
                | self.chat_model
                | self.parser
        )
        return chain

    def clear_chat_history(self, session_config):
        """清空指定会话的对话历史（重构原有clear方法，修复路径硬编码）"""
        try:
            session_id = session_config['configurable']['session_id']
            history_file = os.path.join(config.chat_history_dir, f"{session_id}.json")
            history = FileChatMessageHistory(history_file)
            history.clear()
            logger.info(f"会话{session_id}的对话历史已清空")
            return True
        except Exception as e:
            logger.error(f"清空对话历史失败：{str(e)}")
            return False

    def _get_code_question_chain(self):
        """代码出题链（基于检索的代码资料）"""

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关代码资料"
            return "\n".join([doc.page_content for doc in docs])

        def format_for_prompt_template(value):
            return {
                "input": value["input"],
                "context": value["context"],
            }

        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnablePassthrough() | self.retriever | format_document
                } | RunnableLambda(format_for_prompt_template)
                | ChatPromptTemplate.from_messages([
            ("system", CODE_QUESTION_PROMPT),
            ("user", "根据代码资料生成编程题目：{input}")
        ])
                | print_prompt
                | self.chat_model
                | self.parser
        )
        return chain

    def _get_code_analysis_chain(self):
        """代码分析链（分析代码功能/输出/知识点）"""

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关代码资料"
            return "\n".join([doc.page_content for doc in docs])

        def format_for_prompt_template(value):
            return {
                "input": value["input"],
                "context": value["context"],
            }

        chain = (
                {
                    "input": RunnablePassthrough(),
                    "context": RunnablePassthrough() | self.retriever | format_document
                } | RunnableLambda(format_for_prompt_template)
                | ChatPromptTemplate.from_messages([
            ("system", CODE_ANALYSIS_PROMPT),
            ("user", "请分析以下代码：{input}")
        ])
                | print_prompt
                | self.chat_model
                | self.parser
        )
        return chain
