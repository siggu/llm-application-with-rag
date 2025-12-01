"""
4. 유사도 검색
"""

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()


def get_retriever():
    # 벡터 데이터베이스 설정
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "prompt-engineering-index"

    database = PineconeVectorStore(
        embedding=embedding,
        index_name=index_name,
    )

    retriever = database.as_retriever(search_kwargs={"k": 2})

    return retriever
