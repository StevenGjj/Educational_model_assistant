import streamlit as st
import os
from datetime import datetime
from knowledge_base import KnowledgeBaseService
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger
from langchain_core.documents import Document
import config_data as config
import re

# 页面基础配置
st.set_page_config(
    page_title="知识库上传助手",
    page_icon="📤",
    layout="centered"
)

# 支持的代码文件扩展名
CODE_EXTENSIONS = ["py", "java", "c", "cpp", "js", "h", "hpp", "go", "rs", "rb", "php", "cs", "S", "asm", "s"]

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


# 加载文件（LangChain Loader + 跨页合并 + 代码文件支持）
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
        elif file_type in ["py", "java", "c", "cpp", "js", "cpp", "h", "hpp", "go", "rs", "rb", "php", "cs"]:
            # 代码文件：直接读取为单个文档
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()
            file_name = os.path.basename(file_path)
            documents = [Document(page_content=code_content, metadata={"source": file_name, "type": "code"})]
        else:
            raise ValueError(f"不支持的文件类型：{file_type}")
        return documents
    except Exception as e:
        raise RuntimeError(f"文件加载失败：{str(e)}")


# 处理文档（语义分割 / 代码分割）
def process_documents(documents, filename, file_type=None):
    """
    处理文档，支持普通文本和代码文件
    :param documents: Document列表
    :param filename: 文件名
    :param file_type: 文件类型（用于判断是否为代码文件）
    """
    # 判断是否为代码文件
    is_code = file_type and file_type.lower() in ["py", "java", "c", "cpp", "js", "h", "hpp", "go", "rs", "rb", "php", "cs"]

    # 拼接所有文本
    full_text = "\n".join([doc.page_content.strip() for doc in documents])

    if is_code:
        # 代码文件：按函数/类/代码块分割
        # 使用更小的chunk size和代码特定的分隔符
        code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 代码块更小
            chunk_overlap=50,
            separators=[
                "\nclass ", "\ndef ", "\nfunction ", "\nfunc ", "\npublic ", "\nprivate ", 
                "\nprotected ", "\ndef ", "\n    def ", "\n# ---", "\n// ---", "\n/* ---",
                "\n\n", "\n", ";", "}", ")", "]", ""
            ]
        )
        semantic_docs = code_splitter.create_documents([full_text])
    else:
        # 普通文本：使用语义分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )
        semantic_docs = text_splitter.create_documents([full_text])

    # 补充元数据
    for i, doc in enumerate(semantic_docs):
        doc.metadata.update({
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": config.get_operator_name(),
            "chunk_id": i,
            "total_chunks": len(semantic_docs),
            "file_type": "code" if is_code else "text"
        })

    return full_text, semantic_docs

# 递归遍历文件夹获取所有代码文件
def get_code_files_from_folder(folder_path):
    """递归获取文件夹中所有支持的代码文件（自动跳过非代码文件）"""
    code_files = []
    code_extensions = set(CODE_EXTENSIONS)
    
    for root, dirs, files in os.walk(folder_path):
        # 跳过 .git, build 等目录
        dirs[:] = [d for d in dirs if d not in ['.git', 'build', '__pycache__', '.idea', 'node_modules', '.vscode']]
        
        for file in files:
            ext = file.split('.')[-1].lower() if '.' in file else ''
            # 只收集 CODE_EXTENSIONS 中定义的文件，其他自动跳过
            if ext in code_extensions:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, folder_path)
                code_files.append({
                    'path': full_path,
                    'name': relative_path
                })
            # 非代码文件会被自动跳过，不记录
    
    return code_files


# 加载文件夹中的所有代码文件
def load_folder_files(folder_path):
    """加载文件夹中所有代码文件为Document列表"""
    code_files = get_code_files_from_folder(folder_path)
    documents = []
    
    for file_info in code_files:
        try:
            with open(file_info['path'], 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
            
            # 添加文件头信息
            full_content = f"文件: {file_info['name']}\n\n{code_content}"
            
            documents.append(Document(
                page_content=full_content,
                metadata={
                    "source": file_info['name'],
                    "type": "code",
                    "file_path": file_info['path']
                }
            ))
        except Exception as e:
            logger.warning(f"读取文件失败 {file_info['path']}: {e}")
    
    return documents


# 主页面逻辑
def main():
    init_session_state()
    st.title("📤 知识库文件上传（支持文件夹批量上传）")
    st.divider()

    # 选项卡：单文件上传 / 文件夹批量上传
    upload_mode = st.radio(
        "选择上传方式：",
        ["单文件上传", "文件夹批量上传"],
        horizontal=True
    )

    if upload_mode == "单文件上传":
        # ===== 单文件上传模式 =====
        st.subheader("📄 单文件上传")
        # 文件上传
        upload_files = st.file_uploader(
            "请选择要上传的文件（支持TXT/PDF/代码文件）",
            type=["txt", "pdf", "py", "java", "c", "cpp", "js", "h", "hpp", "go", "rs", "rb", "php", "cs", "S", "asm", "s"],
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
                if file_ext not in ["txt", "pdf", "py", "java", "c", "cpp", "js", "h", "hpp", "go", "rs", "rb", "php", "cs", "S", "asm", "s"]:
                    st.warning(f"跳过非支持文件：{file.name}")
                    os.remove(temp_file_path)
                    continue

                st.subheader(f"正在处理：{file.name}")
                try:
                    # 加载文件（跨页合并/代码支持）
                    documents = load_file_with_langchain(temp_file_path, file_ext)
                    # 处理文档（语义分割/代码分割）
                    full_text, processed_docs = process_documents(documents, file.name, file_ext)
                    # 入库
                    with st.spinner("正在载入向量库..."):
                        result = st.session_state["service"].upload_by_documents(processed_docs, filename=file.name)
                        st.success(f"{file.name}：{result}")
                except Exception as e:
                    st.error(f"{file.name} 处理失败：{str(e)}")
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
    
    else:
        # ===== 文件夹批量上传模式 =====
        st.subheader("📁 文件夹批量上传")
        
        # 输入文件夹路径
        folder_path = st.text_input(
            "请输入代码文件夹路径：",
            placeholder="例如：d:/毕设/一些文件/SUSTechOS",
            help="请输入包含代码文件的文件夹绝对路径"
        )
        
        if folder_path:
            if not os.path.exists(folder_path):
                st.error("❌ 文件夹路径不存在，请检查路径是否正确")
            elif not os.path.isdir(folder_path):
                st.error("❌ 该路径不是文件夹，请输入有效的文件夹路径")
            else:
                with st.spinner("正在扫描文件夹中的代码文件..."):
                    code_files = get_code_files_from_folder(folder_path)
                
                if not code_files:
                    st.warning("⚠️ 文件夹中未找到支持的代码文件")
                else:
                    st.success(f"✅ 找到 {len(code_files)} 个代码文件")
                    
                    # 显示文件列表
                    with st.expander(f"查看文件列表 ({len(code_files)} 个文件)"):
                        for f in code_files:
                            st.text(f"• {f['name']}")
                    
                    # 确认上传按钮
                    if st.button("🚀 开始批量导入向量库", type="primary"):
                        with st.spinner(f"正在处理 {len(code_files)} 个文件..."):
                            try:
                                # 加载文件夹中所有文件
                                documents = load_folder_files(folder_path)
                                
                                if not documents:
                                    st.error("❌ 文件加载失败，未找到有效文件")
                                else:
                                    # 合并为一个文档进行分割
                                    full_text = "\n".join([doc.page_content for doc in documents])
                                    filename = os.path.basename(folder_path)
                                    
                                    # 使用代码分割器处理
                                    _, processed_docs = process_documents(
                                        [Document(page_content=full_text, metadata={"source": filename})],
                                        filename,
                                        "c"  # 标记为代码类型
                                    )
                                    
                                    # 入库
                                    result = st.session_state["service"].upload_by_documents(
                                        processed_docs, 
                                        filename=filename
                                    )
                                    
                                    st.success(f"✅ 文件夹 '{filename}' 导入完成：{result}")
                                    st.info(f"📊 共处理 {len(code_files)} 个代码文件，生成了 {len(processed_docs)} 个语义片段")
                            
                            except Exception as e:
                                st.error(f"❌ 批量导入失败：{str(e)}")
                                logger.error(f"文件夹批量导入异常：{e}", exc_info=True)

    st.divider()
    st.info("✅ 支持格式：TXT、PDF、代码文件（C/C++/Python/Java/Go/Rust等）\n📌 上传后自动去重、分割、向量化入库\n💡 文件夹模式支持递归扫描子目录，非代码文件会自动跳过")

if __name__ == "__main__":
    main()