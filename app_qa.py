import streamlit as st
from rag import RagService
import uuid
from logger import logger  # 复用第一段的logger

# 页面配置（补全基础配置）
st.set_page_config(page_title="智能客服", page_icon="🤖", layout="wide")


# 初始化（保留极简，补充多用户隔离+单例RAG）
def init_session():
    if "message" not in st.session_state:
        st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
    if "rag" not in st.session_state:
        try:
            st.session_state["rag"] = RagService()
            logger.info("RAG服务初始化成功")
        except Exception as e:
            st.error(f"RAG初始化失败：{str(e)}")
            logger.error(f"RAG初始化失败：{e}", exc_info=True)
            st.stop()
    # 补充多用户隔离的session_id（替代固定config）
    if "session_config" not in st.session_state:
        st.session_state["session_config"] = {"configurable": {"session_id": str(uuid.uuid4())}}


# 核心逻辑
def main():
    init_session()
    st.title("智能客服")
    st.divider()

    # 显示历史对话
    for msg in st.session_state["message"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # 清空历史按钮（补充实用功能）
    if st.button("清空对话历史", type="secondary"):
        st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
        logger.info(f"会话 {st.session_state['session_config']['configurable']['session_id']} 历史已清空")

    # 用户输入处理
    prompt = st.chat_input("请输入你的问题...")
    if prompt and prompt.strip():
        # 记录用户提问
        st.chat_message("user").write(prompt)
        st.session_state["message"].append({"role": "user", "content": prompt})
        session_id = st.session_state["session_config"]["configurable"]["session_id"]

        # 流式回答（保留第二段的简洁，补充异常处理+日志）
        with st.spinner("AI思考中..."):
            try:
                logger.info(f"用户 {session_id} 提问：{prompt[:50]}...")
                res_stream = st.session_state["rag"].chain.stream(
                    {"input": prompt},
                    st.session_state["session_config"]
                )

                res = st.chat_message("assistant").write_stream(res_stream)

                # 拼接完整回答并记录
                full_answer = "".join(res)
                st.session_state["message"].append({"role": "assistant", "content": full_answer})
                logger.info(f"用户 {session_id} 回答生成完成")
            except Exception as e:
                error_msg = f"回答生成失败：{str(e)}"
                st.chat_message("assistant").write(error_msg)
                st.session_state["message"].append({"role": "assistant", "content": error_msg})
                logger.error(f"用户 {session_id} 问答异常：{e}", exc_info=True)


if __name__ == "__main__":
    main()