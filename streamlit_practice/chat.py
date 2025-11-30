import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def get_ai_message(user_question: str) -> str:
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    index_name = "tax-markdown-index"

    database = PineconeVectorStore(
        embedding=embedding,
        index_name=index_name,
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    # 질문 변환 프롬프트
    question_transform_prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.
        
        [사전]
        : {dictionary}
        
        [사용자의 질문]
        : {{question}}
        """
    )

    # RAG 답변 생성 프롬프트
    rag_prompt = ChatPromptTemplate.from_template(
        """
        [Identity]
        - 당신은 최고의 한국 소득세 전문가입니다.
        - [Context]를 참고해서 사용자의 질문에 답변해주세요.

        [Context]
        {context}

        [Question]
        {question}
        """
    )

    dictionary_chain = question_transform_prompt | llm | StrOutputParser()

    # Retriever 생성
    retriever = database.as_retriever(search_kwargs={"k": 2})

    # 전체 체인: 질문 변환 -> 검색 -> RAG 답변 생성
    rag_chain = (
        {
            "context": dictionary_chain | retriever,
            "question": dictionary_chain,
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    ai_message = rag_chain.invoke({"question": user_question})

    return ai_message


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

    with st.spinner("AI가 답변을 생성하는 중입니다..."):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state["message_list"].append({"role": "ai", "content": ai_message})
