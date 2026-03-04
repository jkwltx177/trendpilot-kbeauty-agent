import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from config import LLM_MODEL_CREATIVE

def generate_creative_pack(strategy_report: dict, target_country: str, platform: str) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL_CREATIVE, temperature=0.7, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 글로벌 뷰티 마케팅 크리에이티브 디렉터입니다.
전략 보고서를 기반으로 SNS용 카피와 이미지 생성 프롬프트를 JSON 형식으로 작성하세요.

[출력 스키마]
{{
  "copy": {{
    "slogans": ["슬로건 1", "슬로건 2"],
    "short_captions": ["본문 카피 1", "본문 카피 2"],
    "hashtags": ["#해시태그1", "#해시태그2"],
    "seo_description": "SEO 최적화 설명"
  }},
  "poster_prompts": ["DALL-E 3용 영어 프롬프트 (제품의 시각적 속성, 분위기, 타깃 반영. 텍스트 불포함 조건 명시)"]
}}"""),
        ("user", """
[목표 국가] {country}
[플랫폼] {platform}
[전략 보고서]
{strategy}
""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    try:
        pack = chain.invoke({
            "country": target_country,
            "platform": platform,
            "strategy": json.dumps(strategy_report, ensure_ascii=False)
        })
        
        # Generate Image for the first prompt
        pack["generated_images"] = []
        if pack.get("poster_prompts") and len(pack["poster_prompts"]) > 0:
            client = openai.OpenAI()
            try:
                res = client.images.generate(
                    model="dall-e-3",
                    prompt=pack["poster_prompts"][0],
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                pack["generated_images"].append(res.data[0].url)
            except Exception as e:
                print(f"[Creative Agent] Image Gen Error: {e}")
                
        return pack
    except Exception as e:
        print(f"[Creative Agent] Error: {e}")
        return {}
