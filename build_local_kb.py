import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma as LCChroma
from chromadb import PersistentClient
from chromadb.config import Settings
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from config import CHROMA_LOCAL_DIR, EMBEDDING_MODEL

# 로컬 원천 데이터가 저장될 폴더 (PDF, CSV 등)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "local_sources")

def build_kb():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 {DATA_DIR} 폴더가 생성되었습니다.")
        
    files = glob.glob(os.path.join(DATA_DIR, "*"))
    if not files:
        print(f"⚠️ {DATA_DIR} 폴더에 파일이 없습니다.")
        print("💡 PDF(규제/성분 가이드) 또는 CSV(수출통계) 파일을 폴더에 넣고 다시 실행해주세요.")
        return
        
    all_docs = []
    for file_path in files:
        print(f"🔄 로딩 중: {os.path.basename(file_path)}...")
        try:
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
            elif file_path.endswith(".csv"):
                loader = CSVLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                print(f"⏭️ 지원하지 않는 파일 형식: {file_path}")
                continue
                
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(file_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ {file_path} 로딩 오류: {e}")
            
    if not all_docs:
        print("⚠️ 추출된 문서가 없습니다.")
        return
        
    print(f"✅ 총 {len(all_docs)} 페이지/행의 문서를 추출했습니다. 청킹(Chunking)을 시작합니다...")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    
    print(f"✅ 총 {len(chunks)} 개의 청크가 생성되었습니다. 벡터 DB 인덱싱을 시작합니다...")
    
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    os.makedirs(CHROMA_LOCAL_DIR, exist_ok=True)
    client = PersistentClient(path=CHROMA_LOCAL_DIR, settings=Settings(anonymized_telemetry=False))
    
    # 기존 컬렉션 초기화 (중복 방지)
    try:
        client.delete_collection("local_kb")
    except:
        pass
        
    LCChroma.from_documents(
        documents=chunks,
        embedding=embedding,
        client=client,
        collection_name="local_kb",
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"🎉 Local DB 구축 완료! 저장 위치: {CHROMA_LOCAL_DIR}")

if __name__ == "__main__":
    build_kb()
