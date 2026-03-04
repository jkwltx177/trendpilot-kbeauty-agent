import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import LLM_MODEL_STRATEGY

def generate_strategy_report(product_profile: dict, context_pack: dict, target_country: str) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL_STRATEGY, temperature=0.5, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 K-뷰티 글로벌 진출 전략을 수립하는 최고 전략 책임자(CSO)입니다.
입력된 '제품 프로필'과 '컨텍스트 팩(Local/Web 근거)'을 바탕으로 진입 전략 보고서를 JSON 형식으로 생성하세요.

[출력 스키마]
{{
  "market_feasibility": {{
    "score": 85,
    "summary": "시장 진입 가능성 요약",
    "entry_priority": "High|Medium|Low"
  }},
  "risk_analysis": {{
    "regulatory": "Low|Moderate|High",
    "competitive": "Low|Moderate|High",
    "cultural": "Low|Moderate|High",
    "notes": ["리스크 상세 내용"]
  }},
  "differentiation_usp": [
    {{"usp": "핵심 소구점", "evidence_refs": ["근거 출처 요약"]}}
  ],
  "price_positioning": {{
    "strategy": "penetration|value|premium",
    "recommended_range": "가격대",
    "rationale": "이유"
  }},
  "messaging_direction": {{
    "tone": ["메시지 톤"],
    "do": ["해야할 것"],
    "dont": ["피해야 할 것"],
    "example_hooks": ["마케팅 훅 예시"]
  }}
}}"""),
        ("user", """
[목표 국가] {country}
[제품 프로필]
{profile}
[컨텍스트 팩]
{context}
""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    try:
        return chain.invoke({
            "country": target_country,
            "profile": json.dumps(product_profile, ensure_ascii=False),
            "context": json.dumps(context_pack, ensure_ascii=False)
        })
    except Exception as e:
        print(f"[Strategy Agent] Error: {e}")
        return {}
