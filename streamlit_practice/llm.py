import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from config import answer_examples

load_dotenv()


@st.cache_resource
def get_retriever():
    # 벡터 데이터베이스 설정
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    index_name = "tax-markdown-index"

    database = PineconeVectorStore(
        embedding=embedding,
        index_name=index_name,
    )

    # retriever 설정
    retriever = database.as_retriever(search_kwargs={"k": 2})

    return retriever


@st.cache_resource
def get_llm(model: str = "gpt-4o-mini"):
    # LLM 설정
    llm = ChatOpenAI(model=model)

    return llm


def get_rag_prompt():
    # Few-shot examples를 문자열로 포맷팅
    examples_text = "\n\n".join(
        [f"질문: {ex['input']}\n답변: {ex['output']}" for ex in answer_examples]
    )

    # RAG 프롬프트 템플릿
    rag_prompt = ChatPromptTemplate.from_template(
        f"""
        [Identity]
        - 당신은 최고의 한국 소득세 전문가입니다.
        - [Context]를 참고해서 사용자의 질문에 답변해주세요.

        [Examples]
        다음은 좋은 답변들의 예시입니다:
        {examples_text}

        [Chat History]
        {{chat_history}}

        [Context]
        {{context}}

        [Question]
        {{question}}
        """
    )
    return rag_prompt


def history_str_formatter(chat_history):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])


def get_dictionary_chain(llm=None, chat_history=None):
    if llm is None:
        llm = get_llm()
    if chat_history is None:
        chat_history = []

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    # 질문 변환 프롬프트 (히스토리 포함)
    question_transform_prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 이전 대화 내역과 현재 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 명확하게 변경해주세요.
        만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.

        [이전 대화]
        : {history_str_formatter(chat_history)}

        [사전]
        : {dictionary}

        [사용자의 현재 질문]
        : {{question}}

        [변환된 질문]
        (이전 대화를 고려하여 완전한 문장으로 변환해주세요)
        """
    )

    dictionary_chain = question_transform_prompt | llm | StrOutputParser()

    return dictionary_chain


def get_ai_message_stream(user_question: str, chat_history: list):
    """스트리밍으로 AI 응답 생성"""
    if chat_history is None:
        chat_history = []

    retriever = get_retriever()
    rag_prompt = get_rag_prompt()
    dictionary_chain = get_dictionary_chain(chat_history=chat_history)
    llm = get_llm()

    rag_chain = (
        {
            "context": dictionary_chain | retriever,
            "question": dictionary_chain,
            "chat_history": lambda x: history_str_formatter(chat_history),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # ⭐ .invoke() 대신 .stream() 사용 (히스토리 포함)
    return rag_chain.stream(
        {"question": user_question, "chat_history": history_str_formatter(chat_history)}
    )
