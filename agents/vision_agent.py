import base64
import json
from io import BytesIO
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import LLM_MODEL_VISION

def encode_image_pil(image: Image.Image) -> str:
    buffered = BytesIO()
    image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_product_image(image: Image.Image) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL_VISION, max_tokens=1024, model_kwargs={"response_format": {"type": "json_object"}})
    image_encoded = encode_image_pil(image)
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', """당신은 K-뷰티 제품 전문가입니다. 업로드된 제품 사진을 분석하여 아래 JSON 스키마에 맞춰 속성을 추출하세요.
반드시 JSON 형식으로 응답하세요.

{{
  "product_name_guess": "유추되는 제품명 또는 카테고리",
  "category": "skincare|makeup|hair|device|etc 중 택 1",
  "visual_attributes": {{
    "color_palette": ["주요 색상 배열"],
    "material": "용기 재질 (예: 플라스틱, 유리)",
    "shape": "용기 형태",
    "mood": ["minimal", "premium", "cute", "clean", "sporty" 중 적절한 것들]
  }},
  "target_inference": {{
    "age_band": "예상 타깃 연령층",
    "gender": "예상 타깃 성별",
    "persona": "페르소나 (예: 민감성 피부를 가진 20대 대학생)"
  }},
  "keywords": ["핵심 특징 키워드 3~5개"],
  "constraints": {{
    "must_avoid_claims": ["과장 광고로 보일 수 있어 피해야 할 문구 (예: 100% 치료, 완치)"],
    "notes": ["기타 참고사항"]
  }}
}}"""),
        ('user', [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_encoded}"}}])
    ])
    
    chain = prompt | llm | JsonOutputParser()
    try:
        return chain.invoke({})
    except Exception as e:
        print(f"[Vision Agent] Parsing Error: {e}")
        return {}
