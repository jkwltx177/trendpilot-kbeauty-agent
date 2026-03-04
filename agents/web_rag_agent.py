import os
import time
import tempfile
import json
import requests
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma as LCChroma
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_core.documents import Document
from config import EMBEDDING_MODEL

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None

def get_naver_news_links(query: str, num_links: int = 5) -> list[str]:
    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={num_links}&sort=sim"
    headers = {
        'X-Naver-Client-Id': os.environ.get('NAVER_CLIENT_ID', ''),
        'X-Naver-Client-Secret': os.environ.get('NAVER_CLIENT_SECRET', '')
    }
    try:
        res = requests.get(url, headers=headers)
        items = res.json().get('items', [])
        return list(set([item['link'] for item in items if "n.news.naver.com/mnews/article/" in item['link']]))
    except Exception as e:
        print(f"[Naver API Error] {e}")
        return []

def get_google_trends_data(target_country: str, category: str) -> list[Document]:
    """Google Trends 데이터 수집"""
    docs = []
    if TrendReq is None:
        docs.append(Document(page_content="[Google Trends] 모듈이 설치되지 않았습니다.", metadata={"source_type": "google_trends"}))
        return docs
        
    try:
        pytrend = TrendReq(hl='en-US', tz=360)
        
        # 국가 코드 매핑 (현재는 일단 미국, 일본, 유럽 타겟)
        geo_map = {"미국": "US", "일본": "JP", "한국": "KR", "베트남": "VN", "영국": "GB", "프랑스": "FR", "유럽": "EU"}
        geo = geo_map.get(target_country, "")
        
        # 검색어 트렌드
        kw_list = [category]
        pytrend.build_payload(kw_list, cat=0, timeframe='today 3-m', geo=geo)
        related_queries = pytrend.related_queries()
        
        trend_info = f"[{target_country} Google Trends - {category} 관련 최근 3개월 트렌드]\n"
        
        if category in related_queries and related_queries[category] and related_queries[category]['top'] is not None:
            top_queries = related_queries[category]['top'].head(5)['query'].tolist()
            trend_info += f"Top 검색어: {', '.join(top_queries)}\n"
        else:
            trend_info += "Top 검색어 데이터 없음\n"
            
        if category in related_queries and related_queries[category] and related_queries[category]['rising'] is not None:
            rising_queries = related_queries[category]['rising'].head(5)['query'].tolist()
            trend_info += f"급상승 검색어: {', '.join(rising_queries)}\n"
        else:
            trend_info += "급상승 검색어 데이터 없음\n"
            
        docs.append(Document(page_content=trend_info, metadata={"source_type": "google_trends", "country": target_country}))
    except Exception as e:
        print(f"Google Trends Error: {e}")
        docs.append(Document(page_content=f"[{target_country} Google Trends] 트렌드 데이터를 가져오지 못했습니다. ({str(e)})", metadata={"source_type": "google_trends"}))
        
    return docs

def get_competitor_data(target_country: str, keywords: list) -> list[Document]:
    """네이버 뉴스를 활용한 최신 경쟁사 마케팅 사례 검색"""
    docs = []
    if not keywords:
        return docs
        
    try:
        # 주요 키워드 추출하여 검색
        search_kw = " ".join(keywords[:2]) if len(keywords) >= 2 else (keywords[0] if keywords else "")
        query = f"{target_country} 화장품 {search_kw} 진출 경쟁사"
        links = get_naver_news_links(query, num_links=3)
        
        if links:
            loader = WebBaseLoader(
                web_paths=links, 
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("newsct", "newsct-body"))),
                requests_per_second=1
            )
            raw_docs = loader.load()
            
            for doc in raw_docs:
                content = doc.page_content.replace('\t', ' ').replace('\n', ' ')
                content = ' '.join(content.split())
                docs.append(Document(
                    page_content=f"[경쟁사 사례 크롤링] {content[:1000]}...", # 1000자로 제한
                    metadata={"source_type": "competitor_crawl", "url": doc.metadata.get("source", "")}
                ))
    except Exception as e:
        print(f"Competitor Crawl Error: {e}")
        
    return docs

def get_web_context(queries: list, target_country: str, product_profile: dict) -> list:
    """웹 검색 (Google Trends + 경쟁사 크롤링) 통합 래퍼"""
    all_docs = []
    
    category = product_profile.get("category", "skincare")
    keywords = product_profile.get("keywords", [])
    
    # 1. Google Trends (카테고리 기반)
    trends_docs = get_google_trends_data(target_country, category)
    all_docs.extend(trends_docs)
    
    # 2. 경쟁사 크롤링 (키워드 기반)
    competitor_docs = get_competitor_data(target_country, keywords)
    all_docs.extend(competitor_docs)
    
    if not all_docs:
        return []
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    
    if not chunks:
        return []
        
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist_path = tempfile.mkdtemp(prefix="chroma_web_")
    client = PersistentClient(path=persist_path, settings=Settings(anonymized_telemetry=False))
    col_name = f"web_{int(time.time())}"
    
    db = LCChroma.from_documents(documents=chunks, embedding=embedding, client=client, collection_name=col_name)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # 쿼리들을 모아서 검색
    results = []
    search_text = " ".join([q.get('query', '') for q in queries if isinstance(q, dict)])
    if search_text:
        results.extend(retriever.invoke(search_text))
    else:
        results = chunks[:3] # fallback
        
    # Deduplicate
    seen = set()
    deduped_results = []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped_results.append(doc)
            
    return deduped_results
