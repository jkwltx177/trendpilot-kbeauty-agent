import os
from langchain_chroma import Chroma as LCChroma
from langchain_openai import OpenAIEmbeddings
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_core.documents import Document
from config import CHROMA_LOCAL_DIR, EMBEDDING_MODEL

def get_local_context(queries: list, target_country: str) -> list:
    """Local RAG: 실제 규제/수출통계 DB(Chroma)에서 검색 수행"""
    if not queries:
        return []
        
    if not os.path.exists(CHROMA_LOCAL_DIR) or not os.listdir(CHROMA_LOCAL_DIR):
        print("[Local RAG] 로컬 DB를 찾을 수 없습니다. (build_local_kb.py 실행 필요)")
        return [Document(
            page_content="[Local KB 시스템 메시지] 현재 구축된 로컬 규제/통계 DB가 없습니다. 일반적인 지식으로 대안을 제시하세요.",
            metadata={"source_type": "system"}
        )]
        
    try:
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        client = PersistentClient(path=CHROMA_LOCAL_DIR, settings=Settings(anonymized_telemetry=False))
        db = LCChroma(client=client, collection_name="local_kb", embedding_function=embedding)
        
        # 국가 영문명 매핑 (단순 예시)
        country_en_map = {"일본": "Japan", "미국": "USA", "한국": "Korea", "유럽": "EU"}
        target_country_en = country_en_map.get(target_country, target_country)
        
        # MMR 검색기
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
        
        results = []
        for q in queries:
            intent = q.get("intent", "")
            # 쿼리에 타겟 국가 명시
            search_text = f"{target_country} {target_country_en} " + q.get("query", "")
            if search_text:
                docs = retriever.invoke(search_text)
                # intent 정보 임시 보관
                for d in docs:
                    d.metadata['intent'] = intent
                results.extend(docs)
                
        # 중복 제거 및 필터링
        seen = set()
        deduped_results = []
        
        # 규제(PDF)와 통계(CSV)를 골고루 담기 위한 버킷
        reg_docs = []
        stat_docs = []

        for doc in results:
            content = doc.page_content
            source_file = doc.metadata.get('source_file', '')
            
            # 타겟 국가 이름(한글/영문)이 문서 내용이나 파일명에 포함되어 있는지 확인
            if target_country in content or target_country_en in content or target_country_en in source_file or target_country in source_file:
                 if content not in seen:
                    seen.add(content)
                    doc.page_content = f"[Local KB - {source_file}] {content}"
                    
                    # 분리 저장
                    if ".csv" in source_file.lower():
                        stat_docs.append(doc)
                    else:
                        reg_docs.append(doc)
                        
        # 규제(최대 3개) + 통계(최대 3개) 조합하여 반환
        final_docs = reg_docs[:3] + stat_docs[:3]
        
        # 만약 필터링 후 결과가 너무 적으면 fallback
        if len(final_docs) < 2:
             for doc in results:
                content = doc.page_content
                if content not in seen:
                    seen.add(content)
                    doc.page_content = f"[Local KB - {doc.metadata.get('source_file', '알 수 없음')}] {content}"
                    final_docs.append(doc)
                    if len(final_docs) >= 4:
                        break
                        
        return final_docs
    except Exception as e:
        print(f"[Local RAG Error]: {e}")
        return []
