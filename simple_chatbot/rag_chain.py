"""
5. LLM 질의 + retriever + 답변
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from retriever import get_retriever
from config import answer_examples


def get_llm(model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model)
    return llm


def get_rag_prompt():
    examples_text = "\n\n".join(
        [
            f"질문: {ex['input']}\n답변 예시 형식: {ex['output']}"
            for ex in answer_examples
        ]
    )

    rag_prompt = ChatPromptTemplate.from_template(
        f"""
        [Identity]
        당신은 프롬프트 엔지니어링 PDF 문서를 기반으로 답변하는 전문 어시스턴트입니다.

        [Instruction]
        - 답변은 반드시 [Context]의 내용만 사용해야 합니다.
        - Context에 없는 내용은 절대 생성하거나 추론하지 마세요.
        - Few-shot 예시는 '스타일 참고용'입니다. 정보 출처가 아닙니다.
        - Context가 부족하면: "제공된 문서(Context)에 해당 내용이 없습니다." 라고 답변하세요.

        [Answer Style Examples]
        (다음 예시는 답변의 형식과 구조만 참고하세요. 내용은 사용하지 마시오.)
        {examples_text}

        [Chat History]
        {{chat_history}}

        [Context]
        {{context}}

        [Question]
        {{question}}

        이제 위 조건을 STRICT하게 따라 답변하세요.
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


def get_query_transform_chain(chat_history):
    """채팅 히스토리를 고려하여 질문을 완전한 형태로 재구성하고 영어로 번역"""
    llm = get_llm()

    history_text = format_chat_history(chat_history)

    query_transform_prompt = ChatPromptTemplate.from_template(
        f"""
        You are a helpful assistant that reformulates user questions based on chat history.

        [Task]
        1. Read the chat history and current question
        2. If the current question is unclear or uses pronouns (e.g., "더 자세하게", "이것", "그거"), reformulate it into a complete standalone question
        3. Translate the reformulated question into English for document search
        4. If already clear and in English, return as is

        [Chat History]
        {history_text}

        [Current Question]
        {{question}}

        [Output]
        Return ONLY the reformulated English query, nothing else.
        """
    )

    return query_transform_prompt | llm | StrOutputParser()


def get_ai_message_stream(user_question: str, chat_history: list):
    """스트리밍으로 AI 응답 생성"""
    if chat_history is None:
        chat_history = []

    retriever = get_retriever()
    rag_prompt = get_rag_prompt()
    query_transform_chain = get_query_transform_chain(chat_history)
    llm = get_llm()

    rag_chain = (
        {
            "context": query_transform_chain
            | retriever
            | format_docs,  # 히스토리 기반 질문 재구성 -> 영어 번역 -> 검색
            "question": RunnablePassthrough(),  # 원본 질문은 그대로 전달 (한글)
            "chat_history": lambda x: format_chat_history(chat_history),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 스트리밍 반환
    return rag_chain.stream(user_question)
