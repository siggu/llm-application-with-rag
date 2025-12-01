"""
5. LLM 질의 + retriever + 답변
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from retriever import get_retriever


def get_llm(model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model)
    return llm


def get_rag_prompt():
    """RAG 프롬프트 템플릿"""
    rag_prompt = ChatPromptTemplate.from_template(
        """
        [Identity]
        - 당신은 최고의 한국 Prompt Engineering 전문가입니다.
        - [Context]를 참고해서 사용자의 질문에 답변해주세요.
        
        [Chat History]
        {chat_history}

        [Context]
        {context}

        [Question]
        {question}
        
        위 정보를 바탕으로 정답만 생성하세요.
        """
    )
    return rag_prompt


def format_docs(docs):
    """문서를 문자열로 포맷팅"""
    return "\n\n".join([doc.page_content for doc in docs])


def format_chat_history(chat_history):
    """채팅 히스토리를 문자열로 포맷팅"""
    if not chat_history:
        return "대화 기록 없음"

    formatted = []
    for msg in chat_history:
        # 딕셔너리 형태 처리
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"AI: {content}")
        # 튜플 형태 처리
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            formatted.append(f"User: {msg[0]}\nAI: {msg[1]}")

    return "\n".join(formatted) if formatted else "대화 기록 없음"


def get_ai_message_stream(user_question: str, chat_history: list):
    """스트리밍으로 AI 응답 생성"""
    if chat_history is None:
        chat_history = []

    retriever = get_retriever()
    rag_prompt = get_rag_prompt()
    llm = get_llm()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: format_chat_history(chat_history),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 스트리밍 반환
    return rag_chain.stream(user_question)
