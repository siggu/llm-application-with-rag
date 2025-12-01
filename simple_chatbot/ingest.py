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

# ===== 설정 (여기만 수정하면 됨) =====
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
PDF_PATH = "./prompt_engineering.pdf"
INDEX_NAME = "prompt-engineering-index"
FORCE_REBUILD = True  # True: 항상 재생성, False: 기존 것 사용
# =====================================

# 문서 로드 및 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

loader = PyPDFLoader(PDF_PATH)
documents = loader.load_and_split(text_splitter)
print(f"{len(documents)}개의 청크로 분할되었습니다.")

# 임베딩 및 Pinecone 초기화
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
pc = Pinecone()

# 인덱스 존재 여부 확인
existing_indexes = [index.name for index in pc.list_indexes()]

# 인덱스 재생성 또는 재사용
if INDEX_NAME in existing_indexes and FORCE_REBUILD:
    print("기존 인덱스 삭제 중...")
    pc.delete_index(INDEX_NAME)
    print("삭제 완료!")

if INDEX_NAME not in existing_indexes or FORCE_REBUILD:
    print("새 인덱스 생성 중...")
    database = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        index_name=INDEX_NAME,
    )
    print(f"{len(documents)}개 청크 저장 완료!")
else:
    print("기존 인덱스 사용")
    database = PineconeVectorStore(
        embedding=embedding,
        index_name=INDEX_NAME,
    )

print("\n벡터 데이터베이스 준비 완료!")
