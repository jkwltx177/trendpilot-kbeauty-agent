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
            page_content="[Local KB 시스템 메시지] 현재 구축된 로컬 규제/통계 무역 DB가 없습니다. 일반적인 지식으로 대안을 제시하세요.",
            metadata={"source_type": "system"}
        )]
        
    try:
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        client = PersistentClient(path=CHROMA_LOCAL_DIR, settings=Settings(anonymized_telemetry=False))
        db = LCChroma(client=client, collection_name="local_kb", embedding_function=embedding)
        
        # 국가명 키워드 확장 (예: 미국 -> USA, US, United States)
        country_aliases = {"일본": ["Japan", "JP"], "미국": ["USA", "US", "United States"], "한국": ["Korea", "KR", "Rep. of Korea"], "유럽": ["EU", "Europe"]}
        aliases = country_aliases.get(target_country, [target_country])
        
        # MMR 검색기
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
        
        results = []
        for q in queries:
            intent = q.get("intent", "")
            query_str = q.get("query", "")
            search_text = f"{target_country} {aliases[0]} " + query_str
            
            # 통계/수출 데이터는 CSV에 영문(Trade Value, Partner 등)으로 기록되어 있으므로 
            # 한국어 "수출 통계" 쿼리와 매칭이 안 되는 현상 방지를 위해 키워드 보강
            if intent == "export_statistics" or "통계" in query_str or "수출" in query_str:
                search_text += " Trade Value Export Partner Reporter"

            if search_text:
                docs = retriever.invoke(search_text)
                for d in docs:
                    d.metadata['intent'] = intent
                results.extend(docs)
                
        # 중복 제거 및 필터링
        seen = set()
        reg_docs = []
        stat_docs = []

        for doc in results:
            content = doc.page_content
            source_file = doc.metadata.get('source_file', '')
            
            # 타겟 국가 키워드가 포함되어 있는지 확인
            is_match = target_country in content or target_country in source_file
            if not is_match:
                for alias in aliases:
                    if alias.lower() in content.lower() or alias.lower() in source_file.lower():
                        is_match = True
                        break
            
            # CSV(통계)는 데이터 형식이 파편화되어 있을 수 있어 쿼리 매칭만으로도 통과
            if is_match or ".csv" in source_file.lower():
                 if content not in seen:
                    seen.add(content)
                    doc.page_content = f"[Local KB - {source_file}] {content}"
                    
                    if ".csv" in source_file.lower():
                        stat_docs.append(doc)
                    else:
                        reg_docs.append(doc)
                        
        # 규제(PDF)와 통계(CSV)를 적절히 조합 (각 최대 3개)
        final_docs = reg_docs[:3] + stat_docs[:3]
        
        # 필터링 후 너무 적으면 안전장치로 추가 반환
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

