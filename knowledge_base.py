import hashlib
import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config_data as config

# 导入日志（仅用于最终结果提示）
from logger import logger


# ===================== MD5 去重工具函数（移除中间日志） =====================
def get_string_md5(content: str):
    """计算字符串MD5值（用于内容去重）"""
    try:
        md5 = hashlib.md5()
        md5.update(content.encode("utf-8"))
        return md5.hexdigest()
    except Exception as e:
        raise


def check_md5(md5_hex: str):
    """检查MD5是否已存在（避免重复入库）"""
    try:
        if not os.path.exists(config.md5_path):
            with open(config.md5_path, "w", encoding="utf-8") as f:
                f.write("")
            return False

        with open(config.md5_path, "r", encoding="utf-8") as f:
            md5_list = f.read().splitlines()

        return md5_hex in md5_list
    except Exception as e:
        raise


def save_md5(md5_hex: str):
    """保存MD5到文件（记录已入库内容）"""
    try:
        with open(config.md5_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")
    except Exception as e:
        raise


# ===================== 知识库服务核心类 =====================
class KnowledgeBaseService(object):
    def __init__(self):
        """初始化知识库服务"""
        try:
            # 初始化嵌入模型
            self.embeddings = DashScopeEmbeddings(
                model=config.embedding_model_name,
            )
            # 初始化向量库
            self.chroma = Chroma(
                collection_name=config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=config.persist_directory
            )
            # 初始化语义分割器
            self.spliter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators
            )
        except Exception as e:
            logger.error(f"知识库服务初始化失败：{str(e)}")
            raise

    def upload_by_documents(self, documents, filename="未知文件"):
        """
        核心：传入LangChain Document列表入库（仅保留最终日志）
        :param documents: Document列表（含元数据）
        :param filename: 原始文件名
        :return: 处理结果字符串
        """
        try:
            # 1. 拼接内容做MD5去重
            full_text = "\n".join([doc.page_content.strip() for doc in documents])
            if not full_text.strip():
                return "[错误]文件内容为空，跳过处理"

            md5_hex = get_string_md5(full_text)

            # 2. 检查MD5去重
            if check_md5(md5_hex):
                return "[跳过]内容已存在于知识库中"

            # 3. 语义分割
            split_docs = self.spliter.split_documents(documents)
            if not split_docs:
                return "[错误]文档分割后无有效内容"

            # 4. 向量入库
            self.chroma.add_documents(split_docs)

            # 5. 保存MD5去重
            save_md5(md5_hex)

            # 仅保留最终成功日志
            logger.info(f"{filename} 载入向量库成功，共{len(split_docs)}个语义片段")
            return f"[成功]共分割{len(split_docs)}个语义片段，已载入向量库（含窗口检索）"

        except Exception as e:
            # 仅保留最终失败日志
            logger.error(f"{filename} 载入向量库失败：{str(e)}")
            return f"[错误]文件处理失败：{str(e)}"

    def upload_by_str(self, data: str, filename):
        """
        兼容：传入字符串入库（仅保留最终日志）
        :param data: 文本字符串
        :param filename: 文件名
        :return: 处理结果字符串
        """
        try:
            # 1. MD5去重检查
            md5_hex = get_string_md5(data)
            if check_md5(md5_hex):
                return "[跳过]内容已经存在知识库中"

            # 2. 文本分割
            if len(data) > config.max_split_char_number:
                knowledge_chunks = self.spliter.split_text(data)
            else:
                knowledge_chunks = [data]

            # 3. 补充元数据
            metadata = {
                "source": filename,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "operator": config.get_operator_name(),
            }

            # 4. 向量入库
            self.chroma.add_texts(
                knowledge_chunks,
                metadatas=[metadata for _ in knowledge_chunks],
            )

            # 5. 保存MD5
            save_md5(md5_hex)

            # 仅保留最终成功日志
            logger.info(f"{filename} 载入向量库成功，共{len(knowledge_chunks)}个片段")
            return "[成功]内容已经成功载入向量库"

        except Exception as e:
            # 仅保留最终失败日志
            logger.error(f"{filename} 载入向量库失败：{str(e)}")
            return f"[错误]内容入库失败：{str(e)}"

    def __del__(self):
        """析构函数：无日志输出"""
        pass