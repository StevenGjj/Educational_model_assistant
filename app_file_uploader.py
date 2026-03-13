import streamlit as st
import os
from datetime import datetime
from knowledge_base import KnowledgeBaseService
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from logger import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config_data as config

# 页面基础配置
st.set_page_config(
    page_title="知识库上传助手",
    page_icon="📤",
    layout="centered"
)

# 初始化会话状态（单例模式加载知识库服务）
def init_session_state():
    if "service" not in st.session_state:
        try:
            st.session_state["service"] = KnowledgeBaseService()
            logger.info("知识库服务初始化成功")  # 无logger则改为 print("知识库服务初始化成功")
        except Exception as e:
            st.error(f"知识库服务初始化失败：{str(e)}")
            logger.error(f"知识库服务初始化失败：{str(e)}", exc_info=True)  # 无logger则删除
            st.stop()


# 加载文件（LangChain Loader + 跨页合并）
def load_file_with_langchain(file_path, file_type):
    try:
        if file_type == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        elif file_type == "pdf":
            loader = PyPDFLoader(file_path)
            page_docs = loader.load()
            # 跨页合并：每2页合并为1个文档（解决跨页割裂）
            merged_docs = []
            for i in range(0, len(page_docs), 2):
                page1 = page_docs[i]
                page2 = page_docs[i + 1] if (i + 1) < len(page_docs) else Document(page_content="", metadata={"page": i})

                merged_text = page1.page_content.strip()
                if page2:
                    merged_text += "\n" + page2.page_content.strip()

                merged_metadata = {
                    "source": page1.metadata["source"],
                    "pages": f"{page1.metadata['page'] + 1}-{page2.metadata['page'] + 1 if page2 else page1.metadata['page'] + 1}"
                }
                merged_docs.append(Document(page_content=merged_text, metadata=merged_metadata))
            documents = merged_docs
        else:
            raise ValueError(f"不支持的文件类型：{file_type}")
        return documents
    except Exception as e:
        raise RuntimeError(f"文件加载失败：{str(e)}")


# 处理文档（语义分割）
def process_documents(documents, filename):
    # 拼接所有合并后的文本（无视页码）
    full_text = "\n".join([doc.page_content.strip() for doc in documents])

    # 语义分割器（核心：按语义分割，而非页码）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators
    )
    # 按语义分割为小片段
    semantic_docs = text_splitter.create_documents([full_text])

    # 补充元数据
    for i, doc in enumerate(semantic_docs):
        doc.metadata.update({
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": config.get_operator_name(),
            "chunk_id": i,
            "total_chunks": len(semantic_docs)
        })

    return full_text, semantic_docs

# 主页面逻辑
def main():
    init_session_state()
    st.title("📤 知识库文件上传（语义分割+窗口检索）")
    st.divider()

    # 文件上传
    upload_files = st.file_uploader(
        "请选择要上传的文件（支持TXT/PDF）",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if upload_files:
        temp_dir = "./temp_files"
        os.makedirs(temp_dir, exist_ok=True)

        for file in upload_files:
            temp_file_path = os.path.join(temp_dir, file.name)
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())

            file_ext = file.name.split(".")[-1].lower()
            if file_ext not in ["txt", "pdf"]:
                st.warning(f"跳过非支持文件：{file.name}")
                os.remove(temp_file_path)
                continue

            st.subheader(f"正在处理：{file.name}")
            try:
                # 加载文件（跨页合并）
                documents = load_file_with_langchain(temp_file_path, file_ext)
                # 处理文档（语义分割）
                full_text, processed_docs = process_documents(documents, file.name)
                # 入库
                with st.spinner("正在载入向量库..."):
                    result = st.session_state["service"].upload_by_documents(processed_docs, filename=file.name)
                    st.success(f"{file.name}：{result}")
            except Exception as e:
                st.error(f"{file.name} 处理失败：{str(e)}")
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    st.divider()
    st.info("✅ 支持格式：TXT（UTF-8编码）、PDF（可复制文本）\n📌 上传后自动去重、分割、向量化入库")

if __name__ == "__main__":
    main()