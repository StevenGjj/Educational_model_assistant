from langchain_chroma import Chroma
import config_data as config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger  # 导入日志模块

class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
        # 初始化时创建一次窗口分割器，复用
        self.window_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.window_chunk_size,
            chunk_overlap=config.window_chunk_overlap,
            separators=config.separators
        )


    def get_retriever(self):
        """
        核心：返回带引用窗口的检索器
        逻辑：先检索基础片段 → 再扩展为2倍大小的窗口 → 确保跨片段知识点完整
        """
        def custom_window_retriever(query):
            try:
                # 1. 基础检索：仅保留top_k参数，删除score_threshold（关键修复）
                logger.debug(f"开始检索问题：{query[:50]}...，返回Top-{config.top_k}结果")
                base_docs = self.vector_store.similarity_search(
                    query=query,
                    k=config.top_k  # 仅用top_k控制返回数量，无兼容性问题
                )

                if not base_docs:
                    logger.info(f"问题：{query[:50]}... 未检索到相关内容")
                    return []

                # 2. 拼接所有核心片段为完整文本（解决跨片段割裂）
                full_text = "\n".join([doc.page_content for doc in base_docs])
                logger.debug(f"拼接检索结果，总文本长度：{len(full_text)}字符")

                # 3. 按窗口大小重新分割（扩展上下文，复刻原压缩组件效果）
                window_docs = self.window_splitter.create_documents([full_text])
                logger.info(f"窗口分割完成，生成{len(window_docs)}个完整语义片段")

                # 4. 补充元数据（保持和原片段一致）
                for doc in window_docs:
                    doc.metadata = {
                        "source": base_docs[0].metadata.get("source", "unknown"),
                        "type": "window_extended",
                        "retrieval_time": config.get_current_time()  # 可选：添加检索时间
                    }

                return window_docs
            except Exception as e:
                logger.error(f"检索过程异常：{str(e)}", exc_info=True)
                return []

        # 封装为LangChain标准Retriever接口（兼容RAG链调用）
        from langchain_core.retrievers import BaseRetriever
        class WindowRetriever(BaseRetriever):
            def _get_relevant_documents(self, query, **kwargs):
                return custom_window_retriever(query)

        return WindowRetriever()

