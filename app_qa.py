import streamlit as st
from rag import RagService
import uuid
from logger import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from knowledge_base import KnowledgeBaseService
import pdfplumber

# 页面全局配置
st.set_page_config(
    page_title="教育智能助手",
    page_icon="📚",
    layout="wide"
)


# 初始化全局会话（单例RAG/知识库服务+唯一session_id）
def init_global_session():
    # 初始化RAG服务（固定使用通义千问）
    if "rag" not in st.session_state:
        try:
            # 暂时固定使用通义千问模型
            st.session_state["rag"] = RagService()
            logger.info("RAG服务全局初始化成功")
        except Exception as e:
            st.error(f"RAG服务初始化失败：{str(e)}")
            logger.error(f"RAG初始化失败：{e}", exc_info=True)
            st.stop()
    # 初始化知识库服务（文档匹配时加载文件用）
    if "kb_service" not in st.session_state:
        try:
            st.session_state["kb_service"] = KnowledgeBaseService()
            logger.info("知识库服务全局初始化成功")
        except Exception as e:
            st.error(f"知识库服务初始化失败：{str(e)}")
            logger.error(f"知识库初始化失败：{e}", exc_info=True)
            st.stop()
    # 初始化多用户隔离的session_id（唯一）
    if "session_config" not in st.session_state:
        st.session_state["session_config"] = {
            "configurable": {"session_id": str(uuid.uuid4())}
        }
    # 初始化对话消息（仅问答功能用）
    if "message" not in st.session_state:
        st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
    # 初始化临时文件存储（文档匹配用）
    if "temp_docs" not in st.session_state:
        st.session_state["temp_docs"] = {"doc_a": "", "doc_b": ""}


# 工具函数：加载本地文件为文本（文档匹配用，支持TXT/PDF）
def load_file_to_text(file):
    """将上传的文件转为纯文本，返回文本内容"""
    try:
        temp_dir = "./temp_func_files"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        full_text = ""

        if file.name.endswith(".txt"):
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            full_text = "\n".join([doc.page_content.strip() for doc in docs])

        elif file.name.endswith(".pdf"):
            # 使用 pdfplumber 解析 PDF，更稳定
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"

        else:
            st.error("仅支持 TXT / PDF 格式")
            os.remove(temp_path)
            return ""

        os.remove(temp_path)
        return full_text.strip()

    except Exception as e:
        logger.error(f"文件加载失败：{str(e)}")
        st.error(f"文件解析失败：{str(e)}")
        return ""


# 功能1：知识问答（原有功能，保留流式输出+历史对话）
def tab_qa():
    st.subheader("📝 知识问答（基于知识库）")
    # 显示历史对话
    for msg in st.session_state["message"]:
        st.chat_message(msg["role"]).write(msg["content"])
    # 清空历史按钮
    if st.button("🗑️ 清空对话历史", type="secondary", key="clear_qa"):
        st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
        st.session_state["rag"].clear_chat_history(st.session_state["session_config"])
        st.rerun()
    # 用户输入处理
    prompt = st.chat_input("请输入你的问题（如：进程调度的核心算法有哪些？）")
    if prompt and prompt.strip():
        st.chat_message("user").write(prompt)
        st.session_state["message"].append({"role": "user", "content": prompt})
        session_id = st.session_state["session_config"]["configurable"]["session_id"]
        with st.spinner("AI思考中..."):
            try:
                logger.info(f"用户{session_id}【知识问答】提问：{prompt[:50]}...")
                # 调用RAG的问答链（流式输出）
                res_stream = st.session_state["rag"].qa_chain.stream(
                    {"input": prompt},
                    st.session_state["session_config"]
                )
                # 流式写入回答
                full_answer = []

                def stream_answer():
                    for chunk in res_stream:
                        full_answer.append(chunk)
                        yield chunk

                st.chat_message("assistant").write_stream(stream_answer)
                # 记录完整回答
                st.session_state["message"].append({"role": "assistant", "content": "".join(full_answer)})
                logger.info(f"用户{session_id}【知识问答】回答生成完成")
            except Exception as e:
                error_msg = f"回答生成失败：{str(e)}"
                st.chat_message("assistant").write(error_msg)
                st.session_state["message"].append({"role": "assistant", "content": error_msg})
                logger.error(f"用户{session_id}【知识问答】异常：{e}", exc_info=True)


# 功能2：文档匹配度分析（从向量库选择文件对比，无需上传）
def tab_doc_match():
    st.subheader("📑 向量库文档匹配度分析")
    st.caption("从已上传知识库中选择两个文档，分析知识点匹配度")

    # 初始化向量库服务（如果未初始化）
    if "vector_service" not in st.session_state:
        from vector_stores import VectorStoreService
        from langchain_community.embeddings import DashScopeEmbeddings
        import config_data as config
        st.session_state["vector_service"] = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

    # 获取所有文档来源
    sources = st.session_state["vector_service"].list_all_sources()
    if not sources:
        st.warning("⚠️ 向量库中暂无文档，请先上传课件/教材到知识库")
        return

    # 选择两个待比较的文档
    col1, col2 = st.columns(2)
    with col1:
        doc_a_source = st.selectbox("选择【文档A】", options=sources, key="doc_a_source")
    with col2:
        doc_b_source = st.selectbox("选择【文档B】", options=sources, key="doc_b_source")

    # 分析按钮
    if st.button("🔍 开始分析", type="primary", key="match_analyse"):
        if doc_a_source == doc_b_source:
            st.warning("⚠️ 请选择两个不同的文档进行比较")
            return

        with st.spinner("正在从向量库加载文档内容..."):
            # 从向量库获取文档内容
            doc_a_content = st.session_state["vector_service"].get_document_content_by_source(doc_a_source)
            doc_b_content = st.session_state["vector_service"].get_document_content_by_source(doc_b_source)

        if not doc_a_content or not doc_b_content:
            st.error("❌ 选中的文档内容为空，请检查知识库")
            return

        # 处理超长文本（适配通义千问输入限制）
        MAX_INPUT_LENGTH = 250000
        doc_a_content = doc_a_content[:MAX_INPUT_LENGTH//2]
        doc_b_content = doc_b_content[:MAX_INPUT_LENGTH//2]
        if len(doc_a_content + doc_b_content) > MAX_INPUT_LENGTH:
            st.info(f"⚠️ 文档内容过长，已自动截断至 {MAX_INPUT_LENGTH} 字符进行分析")

        with st.spinner("正在分析知识点匹配度..."):
            try:
                logger.info(f"开始分析文档匹配：{doc_a_source} vs {doc_b_source}")
                # 调用匹配分析链
                result = st.session_state["rag"].doc_match_chain.invoke({
                    "doc_a_content": doc_a_content,
                    "doc_b_content": doc_b_content
                })
                st.divider()
                st.markdown(f"### 📊 匹配分析结果：{doc_a_source} ↔ {doc_b_source}")
                st.write(result)
                logger.info("文档匹配分析完成")
            except Exception as e:
                st.error(f"分析失败：{str(e)}")
                logger.error(f"文档匹配分析异常：{e}", exc_info=True)


# 功能3：自动生成题目（基于知识库，支持自定义题型/难度）
def tab_gen_question():
    st.subheader("📜 基于知识库自动生成题目")
    st.caption("根据知识库中的教育资料，生成单选/多选/判断/简答题目")
    # 自定义出题要求
    prompt = st.text_area(
        "输入出题要求（示例：生成5道操作系统中等难度的简答题，围绕进程管理）",
        placeholder="请说明题型、数量、难度、知识点范围...",
        height=100,
        key="gen_question_prompt"
    )
    # 生成按钮
    if st.button("✏️ 生成题目", type="primary", key="gen_question"):
        if not prompt or prompt.strip() == "":
            st.warning("⚠️ 请输入出题要求")
            return
        with st.spinner("正在根据知识库生成题目..."):
            try:
                session_id = st.session_state["session_config"]["configurable"]["session_id"]
                logger.info(f"用户{session_id}【自动出题】要求：{prompt[:50]}...")
                # 调用RAG的出题链
                result = st.session_state["rag"].gen_question_chain.invoke(prompt)
                # 展示结果
                st.divider()
                st.markdown("### 📝 生成的题目")
                st.write(result)
                logger.info(f"用户{session_id}【自动出题】完成")
            except Exception as e:
                st.error(f"题目生成失败：{str(e)}")
                logger.error(f"自动出题异常：{e}", exc_info=True)


# 功能4：自动生成评分标准（基于知识库+用户题目）
def tab_gen_criteria():
    st.subheader("⚖️ 题目评分标准&标准答案生成")
    st.caption("根据知识库内容，为题目生成标准答案、得分要点、评分细则")
    # 输入用户题目
    prompt = st.text_area(
        "输入需要生成评分标准的题目（可多道，分行输入）",
        placeholder="示例：1. 简述死锁的四个必要条件？\n2. 说明分页和分段的区别？",
        height=150,
        key="gen_criteria_prompt"
    )
    # 生成按钮
    if st.button("📋 生成评分标准", type="primary", key="gen_criteria"):
        if not prompt or prompt.strip() == "":
            st.warning("⚠️ 请输入题目内容")
            return
        with st.spinner("正在生成评分标准..."):
            try:
                session_id = st.session_state["session_config"]["configurable"]["session_id"]
                logger.info(f"用户{session_id}【生成评分标准】题目：{prompt[:50]}...")
                # 调用RAG的评分标准链
                result = st.session_state["rag"].gen_criteria_chain.invoke(prompt)
                # 展示结果
                st.divider()
                st.markdown("### 📊 评分标准&标准答案")
                st.write(result)
                logger.info(f"用户{session_id}【生成评分标准】完成")
            except Exception as e:
                st.error(f"评分标准生成失败：{str(e)}")
                logger.error(f"生成评分标准异常：{e}", exc_info=True)


# 功能5：代码出题（基于知识库中的代码文件）
def tab_code_question():
    st.subheader("💻 基于代码知识库自动生成题目")
    st.caption("根据知识库中的代码资料，生成编程题目（代码填空/改错/分析/实现）")
    # 自定义出题要求
    prompt = st.text_area(
        "输入出题要求（示例：生成3道Python中等难度的编程题，围绕列表操作）",
        placeholder="请说明题型、数量、难度、编程语言、知识点范围...",
        height=100,
        key="code_question_prompt"
    )
    # 生成按钮
    if st.button("✏️ 生成编程题目", type="primary", key="code_question"):
        if not prompt or prompt.strip() == "":
            st.warning("⚠️ 请输入出题要求")
            return
        with st.spinner("正在根据代码知识库生成编程题目..."):
            try:
                session_id = st.session_state["session_config"]["configurable"]["session_id"]
                logger.info(f"用户{session_id}【代码出题】要求：{prompt[:50]}...")
                # 调用RAG的代码出题链
                result = st.session_state["rag"].code_question_chain.invoke(prompt)
                # 展示结果
                st.divider()
                st.markdown("### 💻 生成的编程题目")
                st.write(result)
                logger.info(f"用户{session_id}【代码出题】完成")
            except Exception as e:
                st.error(f"编程题目生成失败：{str(e)}")
                logger.error(f"代码出题异常：{e}", exc_info=True)


# 功能6：代码分析（分析知识库中的代码文件）
def tab_code_analysis():
    st.subheader("🔍 代码分析（基于知识库）")
    st.caption("分析知识库中的代码文件，提取功能概述、算法思路、知识点等")
    # 输入代码相关问题或指定分析要求
    prompt = st.text_area(
        "输入分析要求（示例：分析代码中使用的算法/数据结构/输出结果）",
        placeholder="请说明想要分析的方向，如：分析代码的算法思路、找出关键语法点...",
        height=100,
        key="code_analysis_prompt"
    )
    # 分析按钮
    if st.button("🔬 分析代码", type="primary", key="code_analysis"):
        if not prompt or prompt.strip() == "":
            st.warning("⚠️ 请输入分析要求")
            return
        with st.spinner("正在分析代码..."):
            try:
                session_id = st.session_state["session_config"]["configurable"]["session_id"]
                logger.info(f"用户{session_id}【代码分析】要求：{prompt[:50]}...")
                # 调用RAG的代码分析链
                result = st.session_state["rag"].code_analysis_chain.invoke(prompt)
                # 展示结果
                st.divider()
                st.markdown("### 🔍 代码分析结果")
                st.write(result)
                logger.info(f"用户{session_id}【代码分析】完成")
            except Exception as e:
                st.error(f"代码分析失败：{str(e)}")
                logger.error(f"代码分析异常：{e}", exc_info=True)


# 主函数：标签页整合所有功能
def main():
    # ===== 模型切换功能暂时禁用 =====
    # import config_data as config_model
    # model_options = list(config_model.AVAILABLE_MODELS.keys())
    
    # 侧边栏：模型选择
    # st.sidebar.title("⚙️ 模型设置")
    # selected_model = st.sidebar.selectbox(
    #     "选择大语言模型：",
    #     options=model_options,
    #     index=0,
    #     key="model_selector"
    # )
    
    # 暂时固定使用通义千问
    selected_model = "通义千问"
    
    # 如果模型变化了，重新初始化
    # if "selected_model" not in st.session_state or st.session_state.get("selected_model") != selected_model:
    #     st.session_state["selected_model"] = selected_model
    #     # 清除现有的RAG服务，触发重新初始化
    #     if "rag" in st.session_state:
    #         del st.session_state["rag"]
    #     if "kb_service" in st.session_state:
    #         del st.session_state["kb_service"]
    #     if "vector_service" in st.session_state:
    #         del st.session_state["vector_service"]
    
    # 显示当前使用的模型
    # st.sidebar.markdown(f"当前使用：**{selected_model}**")
    # st.sidebar.divider()
    
    # 初始化全局会话
    init_global_session()
    st.title("📚 教育智能助手")
    st.divider()
    # 创建6个功能标签页
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "知识问答", "文档匹配分析", "自动生成题目", "评分标准生成", "代码出题", "代码分析"
    ])
    with tab1:
        tab_qa()
    with tab2:
        tab_doc_match()
    with tab3:
        tab_gen_question()
    with tab4:
        tab_gen_criteria()
    with tab5:
        tab_code_question()
    with tab6:
        tab_code_analysis()


if __name__ == "__main__":
    main()
