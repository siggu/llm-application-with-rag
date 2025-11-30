import streamlit as st

from llm import get_ai_message_stream

st.set_page_config(
    page_title="소득세 챗봇",
    page_icon="🤖",
)

st.title("🤖 소득세 챗봇")
st.caption("소득세 관련 질문에 답변해 드립니다.")

if "message_list" not in st.session_state:
    st.session_state["message_list"] = []

for message in st.session_state["message_list"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input(
    placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"
):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state["message_list"].append({"role": "user", "content": user_question})

    with st.chat_message("ai"):
        # ⭐ 상태 표시와 함께 스트리밍
        with st.status("🤔 답변을 생성하고 있습니다...", expanded=True) as status:
            message_placeholder = st.empty()
            full_response = ""

            # 스트리밍으로 답변 받기
            for chunk in get_ai_message_stream(
                user_question, st.session_state["message_list"][:-1]
            ):
                full_response += chunk
                # 커서 효과와 함께 실시간 표시
                message_placeholder.markdown(full_response + "▌")

            # 최종 응답 (커서 제거)
            message_placeholder.markdown(full_response)
            status.update(label="✅ 답변 완료!", state="complete")

    st.session_state["message_list"].append({"role": "ai", "content": full_response})
