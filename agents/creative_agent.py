import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import base64
import re
from io import BytesIO
from PIL import Image
import tempfile
from config import LLM_MODEL_CREATIVE

def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()

def _save_b64_to_temp_png(b64_data: str) -> str:
    raw = base64.b64decode(b64_data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        f.write(raw)
        return f.name

def _resolve_language_instruction(copy_language_mode: str, target_country: str) -> str:
    target_lang = _target_language_from_country(target_country)
    if copy_language_mode == "한국어":
        return "Write all ad copy in Korean only."
    return (
        "Write all ad copy only in the target-country language. "
        "Do not output Korean. "
        f"The target country is {target_country}. "
        f"Use this language: {target_lang}."
    )

def _target_language_from_country(target_country: str) -> str:
    mapping = {
        "한국": "Korean",
        "대한민국": "Korean",
        "일본": "Japanese",
        "미국": "English",
        "영국": "English",
        "유럽": "English",
        "프랑스": "French",
        "독일": "German",
        "스페인": "Spanish",
        "이탈리아": "Italian",
        "베트남": "Vietnamese",
        "태국": "Thai",
        "인도네시아": "Indonesian",
        "중국": "Chinese (Simplified)",
        "대만": "Chinese (Traditional)",
    }
    return mapping.get(str(target_country).strip(), "English")

def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))

def _platform_copy_guideline(platform: str) -> str:
    p = (platform or "").lower()
    if p == "instagram":
        return (
            "Instagram optimized: aesthetic and story-driven tone, 1-3 emojis, "
            "natural CTA, and 5-8 relevant hashtags. Make captions richer than other platforms."
        )
    if p == "tiktok":
        return (
            "TikTok optimized: punchy hook in first line, trendy conversational tone, "
            "1-3 emojis, short lines, and 3-6 hashtags at the end."
        )
    if p == "amazon":
        return (
            "Amazon optimized: trust-focused and benefit-first copy, no slang, no emojis, "
            "no hashtags, clear purchase intent."
        )
    if p == "shopee":
        return (
            "Shopee optimized: deal-friendly and practical tone, highlight value and key benefits, "
            "minimal emoji use (0-1), no hashtag spam."
        )
    return "Use platform-native tone and conventions."

def _platform_caption_length_rule(platform: str) -> str:
    p = (platform or "").lower()
    if p == "instagram":
        return (
            "For Instagram, each short_captions item must be medium-long: "
            "at least 2-4 sentences (or about 120-220 characters in Korean)."
        )
    if p == "tiktok":
        return "For TikTok, keep each short_captions item short and punchy: about 1-2 short sentences."
    return "Use practical caption length for the platform."

def _platform_image_guideline(platform: str) -> tuple[str, str]:
    p = (platform or "").lower()
    if p in {"instagram", "tiktok"}:
        return (
            "Lifestyle social ad style: show a person naturally using the product in a real-life beauty routine. "
            "Candid, authentic, modern composition. Keep the exact product identity visible.",
            "1024x1536" if p == "tiktok" else "1024x1024",
        )
    return (
        "E-commerce product photo style: clean and natural product-focused shot, realistic lighting, "
        "clear packaging visibility, no person required, no cluttered props.",
        "1024x1024",
    )

def _cleanup_copy_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"^\s*\[[^\]]+\]\s*$", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*(hashtags?|해시태그|seo(\s*description)?|seo\s*설명)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def _normalize_hashtags(tags) -> list[str]:
    if not isinstance(tags, list):
        return []
    normalized = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        t = tag.strip()
        if not t:
            continue
        if not t.startswith("#"):
            t = f"#{t}"
        normalized.append(t.replace(" ", ""))
    return normalized[:10]

def _build_copy_text_from_fields(copy_fields: dict) -> str:
    slogans = copy_fields.get("slogans", [])
    short_captions = copy_fields.get("short_captions", [])
    hashtags = _normalize_hashtags(copy_fields.get("hashtags", []))
    seo_description = str(copy_fields.get("seo_description", "")).strip()

    s1 = slogans[0].strip() if isinstance(slogans, list) and len(slogans) > 0 and isinstance(slogans[0], str) else ""
    s2 = slogans[1].strip() if isinstance(slogans, list) and len(slogans) > 1 and isinstance(slogans[1], str) else ""
    c1 = short_captions[0].strip() if isinstance(short_captions, list) and len(short_captions) > 0 and isinstance(short_captions[0], str) else ""
    c2 = short_captions[1].strip() if isinstance(short_captions, list) and len(short_captions) > 1 and isinstance(short_captions[1], str) else ""
    tag_line = " ".join(hashtags)

    parts = [s1, s2, c1, c2]
    if tag_line:
        parts.append(tag_line)
    if seo_description:
        parts.append(seo_description)
    text = "\n\n".join([p for p in parts if p])
    return _cleanup_copy_text(text)

def _force_translate_copy_text_if_needed(
    text: str,
    copy_language_mode: str,
    target_country: str,
) -> str:
    if copy_language_mode == "한국어":
        return text
    if not text or not _contains_hangul(text):
        return text

    target_lang = _target_language_from_country(target_country)
    llm = ChatOpenAI(model=LLM_MODEL_CREATIVE, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Translate the text into the requested language only. "
            "Preserve meaning, line breaks, emojis, and hashtags. "
            "Do not add labels or explanations."
        ),
        (
            "user",
            "Target language: {target_lang}\n\nText:\n{text}"
        ),
    ])
    chain = prompt | llm
    try:
        translated = chain.invoke({"target_lang": target_lang, "text": text}).content
        return translated.strip() if isinstance(translated, str) else text
    except Exception:
        return text

def generate_creative_pack(
    strategy_report: dict,
    target_country: str,
    platform: str,
    input_image: Image.Image | None = None,
    copy_language_mode: str = "한국어",
) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL_CREATIVE, temperature=0.7, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 글로벌 뷰티 마케팅 크리에이티브 디렉터입니다.
전략 보고서를 기반으로 광고 카피와 이미지 생성 프롬프트를 JSON 형식으로 작성하세요.
모든 플랫폼에서 아래 카피 구조를 반드시 포함하세요.
플랫폼별 특성에 맞게 문체/길이를 조정하세요.

[출력 스키마]
{{
  "copy_fields": {{
    "slogans": ["슬로건 1", "슬로건 2"],
    "short_captions": ["본문 카피 1", "본문 카피 2"],
    "hashtags": ["#해시태그1", "#해시태그2"],
    "seo_description": "SEO 최적화 설명"
  }},
  "poster_prompts": ["영문 이미지 프롬프트 1개 이상"]
}}"""),
        ("user", """
[목표 국가] {country}
[타겟 언어] {target_language}
[플랫폼] {platform}
[카피 언어 지침] {language_instruction}
[플랫폼 카피 지침] {platform_copy_guideline}
[캡션 길이 규칙] {platform_caption_length_rule}
[플랫폼 이미지 지침] {platform_image_guideline}
[전략 보고서]
{strategy}
""")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    try:
        pack = chain.invoke({
            "country": target_country,
            "target_language": _target_language_from_country(target_country),
            "platform": platform,
            "language_instruction": _resolve_language_instruction(copy_language_mode, target_country),
            "platform_copy_guideline": _platform_copy_guideline(platform),
            "platform_caption_length_rule": _platform_caption_length_rule(platform),
            "platform_image_guideline": _platform_image_guideline(platform)[0],
            "strategy": json.dumps(strategy_report, ensure_ascii=False)
        })
        copy_fields = pack.get("copy_fields", {}) if isinstance(pack, dict) else {}
        if not isinstance(copy_fields, dict):
            copy_fields = {}
        pack["copy_text"] = _build_copy_text_from_fields(copy_fields)
        pack["copy_text"] = _force_translate_copy_text_if_needed(pack["copy_text"], copy_language_mode, target_country)
        if not pack["copy_text"]:
            pack["copy_text"] = "카피 생성 결과가 비어 있습니다. 다시 시도해주세요."
        
        # Generate image with input product image whenever available.
        pack["generated_images"] = []
        if pack.get("poster_prompts") and len(pack["poster_prompts"]) > 0:
            client = openai.OpenAI()
            platform_image_style, image_size = _platform_image_guideline(platform)
            try:
                if input_image is not None:
                    edit_prompt = (
                        "Use the provided product image as the exact hero product. "
                        "Preserve the same container shape, label identity, key colors, and material. "
                        "Do not replace the product with a different package. "
                        f"Create a premium ad composition for {platform} in {target_country}. "
                        f"{platform_image_style} "
                        "No readable text/logos overlaid on the image. "
                        f"Creative direction: {pack['poster_prompts'][0]}"
                    )
                    res = client.images.edit(
                        model="gpt-image-1",
                        image=[("product.png", _pil_to_png_bytes(input_image), "image/png")],
                        prompt=edit_prompt,
                        size=image_size,
                        quality="high",
                    )
                    if res.data and getattr(res.data[0], "b64_json", None):
                        pack["generated_images"].append(_save_b64_to_temp_png(res.data[0].b64_json))
                    elif res.data and getattr(res.data[0], "url", None):
                        pack["generated_images"].append(res.data[0].url)
                else:
                    res = client.images.generate(
                        model="dall-e-3",
                        prompt=f"{platform_image_style} {pack['poster_prompts'][0]}",
                        size=image_size,
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
