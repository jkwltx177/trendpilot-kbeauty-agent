import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import LLM_MODEL_ROUTER

def build_queries(product_profile: dict, target_country: str, platform: str) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL_ROUTER, max_tokens=512, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 K-뷰티 글로벌 마케팅 RAG 쿼리 빌더입니다.
제공된 제품 프로필과 목표 국가, 플랫폼 정보를 바탕으로 검색 의도에 맞는 로컬 DB 검색 쿼리와 웹 검색 쿼리를 JSON 형식으로 생성하세요.

[필수 검색 항목]
Local DB의 경우 다음 두 가지 주제에 대한 쿼리를 반드시 포함해야 합니다:
1. 규제 및 성분 제한 ("regulation")
2. 수출 통계 및 무역 동향 ("export_statistics")

Web 검색의 경우 다음 두 가지 주제를 반드시 포함해야 합니다:
1. Google Trends 키워드 ("trend")
2. 경쟁사 마케팅 사례 ("competitor")

[출력 형식]
{{
  "country": "KR|JP|US|EU|...",
  "platform": "instagram|tiktok|...",
  "local_queries": [
    {{"intent": "regulation", "query": "string"}},
    {{"intent": "export_statistics", "query": "string"}}
  ],
  "web_queries": [
    {{"intent": "trend", "query": "string"}},
    {{"intent": "competitor", "query": "string"}}
  ]
}}"""),
        ("user", """
[목표 국가] {country}
[목표 플랫폼] {platform}
[제품 프로필] {product_profile}
""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    try:
        return chain.invoke({
            "country": target_country,
            "platform": platform,
            "product_profile": json.dumps(product_profile, ensure_ascii=False)
        })
    except Exception as e:
        print(f"[Query Builder] Error: {e}")
        return {"local_queries": [], "web_queries": []}
