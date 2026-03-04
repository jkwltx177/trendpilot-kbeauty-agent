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
        return list(set([item['link'] for item in items if "n.news.naver.com/article/" in item['link'] or "n.news.naver.com/mnews/article/" in item['link']]))
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
    
    try:
        # 주요 키워드 추출하여 검색 (키워드가 없어도 기본 검색 수행)
        search_kw = " ".join(keywords[:2]) if keywords and len(keywords) >= 2 else (keywords[0] if keywords else "스킨케어")
        query = f"{target_country} K뷰티 {search_kw} 마케팅"
        links = get_naver_news_links(query, num_links=3)
        
        # 특정 키워드로 검색이 안될 경우 일반적인 K뷰티 검색어로 재시도
        if not links:
            query = f"{target_country} K뷰티 마케팅"
            links = get_naver_news_links(query, num_links=3)
        
        if links:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            for link in links:
                try:
                    r = requests.get(link, headers=headers, timeout=5)
                    soup = bs4.BeautifulSoup(r.text, 'html.parser')
                    
                    # 네이버 뉴스 본문 영역 추출 시도 (강화)
                    article = soup.find(id="dic_area") or soup.find(class_="newsct_article") or soup.find("article") or soup.find("div", class_="article_body")
                    if article:
                        content = article.get_text(separator=' ', strip=True)
                    else:
                        # Fallback: 그냥 body 텍스트 대충 긁기
                        content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
                        
                    if content.strip():
                        docs.append(Document(
                            page_content=f"[경쟁사 마케팅 사례 크롤링] {content[:1000]}...", # 1000자로 제한
                            metadata={"source_type": "competitor_crawl", "url": link}
                        ))
                except Exception as ex:
                    print(f"[Fetch Error] {link}: {ex}")
        
        # 크롤링 실패나 결과가 없을 경우 대비 Mock 데이터 추가
        if not docs:
             docs.append(Document(
                page_content=f"[경쟁사 마케팅 사례] {target_country} 시장에서 유사한 카테고리의 K-뷰티 경쟁사들은 현지 인플루언서 마케팅과 틱톡 숏폼, 앰버서더를 적극적으로 활용하여 브랜드 인지도를 높이고 있습니다. 가성비를 강조하는 캠페인이 주를 이룹니다.",
                metadata={"source_type": "competitor_crawl_fallback"}
            ))
    except Exception as e:
        print(f"Competitor Crawl Error: {e}")
        docs.append(Document(
                page_content=f"[경쟁사 마케팅 사례] {target_country} 시장 검색 중 오류 발생. 경쟁사들은 대체로 현지화된 패키징과 SNS 챌린지를 활용함.",
                metadata={"source_type": "competitor_crawl_error"}
        ))
        
    return docs

def get_web_context(queries: list, target_country: str, product_profile: dict) -> list:
    """웹 검색 (Google Trends + 경쟁사 크롤링) 통합 래퍼
    수집된 문서가 적으므로 별도 VectorDB를 구성하지 않고 즉시 반환합니다.
    """
    all_docs = []
    
    category = product_profile.get("category", "skincare")
    keywords = product_profile.get("keywords", [])
    
    # 1. Google Trends (카테고리 기반)
    trends_docs = get_google_trends_data(target_country, category)
    all_docs.extend(trends_docs)
    
    # 2. 경쟁사 크롤링 (키워드 기반)
    competitor_docs = get_competitor_data(target_country, keywords)
    all_docs.extend(competitor_docs)
    
    # 별도 검색 없이 수집된 모든 문서(트렌드 + 최신 뉴스) 반환
    # 중복 제거
    seen = set()
    deduped_results = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped_results.append(doc)
            
    return deduped_results

