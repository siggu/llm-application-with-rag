"""
1. 문서의 내용을 읽는다.
2. 문서를 쪼갠다.
3. 임베딩 생성 및 벡터 데이터베이스에 저장한다.
"""

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()

# 문서 로드 및 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = PyPDFLoader("./prompt_engineering.pdf")

documents = loader.load_and_split(text_splitter)

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone()

index_name = "prompt-engineering-index"

database = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embedding,
    index_name=index_name,
)
