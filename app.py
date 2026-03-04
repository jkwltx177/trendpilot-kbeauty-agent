import gradio as gr
import json
import os
import sys

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from agents.vision_agent import analyze_product_image
from agents.rag_router import build_queries
from agents.local_rag_agent import get_local_context
from agents.web_rag_agent import get_web_context
from agents.strategy_agent import generate_strategy_report
from agents.creative_agent import generate_creative_pack

def format_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)

def run_pipeline(image, target_country, platform, copy_language_mode):
    if image is None:
        yield "이미지를 업로드해주세요.", None, None, None, None, None
        return
        
    yield "⏳ 1/5 - Vision Agent: 제품 사진 분석 중...", None, None, None, None, None
    product_profile = analyze_product_image(image)
    
    yield "⏳ 2/5 - RAG Router: 검색 쿼리 빌드 및 데이터 수집 중...", format_json(product_profile), None, None, None, None
    queries = build_queries(product_profile, target_country, platform)
    
    local_docs = get_local_context(queries.get("local_queries", []), target_country)
    web_docs = get_web_context(queries.get("web_queries", []), target_country, product_profile)
    
    context_pack = {
        "local_evidence": [d.page_content for d in local_docs],
        "web_evidence": [d.page_content for d in web_docs]
    }
    
    yield "⏳ 3/5 - Strategy Agent: 시장 진입 전략 보고서 생성 중...", format_json(product_profile), format_json(context_pack), None, None, None
    strategy_report = generate_strategy_report(product_profile, context_pack, target_country)
    
    yield "⏳ 4/5 - Creative Agent: 마케팅 에셋 및 포스터 생성 중...", format_json(product_profile), format_json(context_pack), format_json(strategy_report), None, None
    creative_pack = generate_creative_pack(
        strategy_report,
        target_country,
        platform,
        input_image=image,
        copy_language_mode=copy_language_mode
    )
    
    img_url = creative_pack.get("generated_images", [None])[0] if creative_pack.get("generated_images") else None
    copy_text = creative_pack.get("copy_text", "")
    
    yield "✅ 모든 파이프라인 완료!", format_json(product_profile), format_json(context_pack), format_json(strategy_report), copy_text, img_url

with gr.Blocks(title="TrendPilot: K-Beauty Global Agent") as demo:
    gr.Markdown("# 🚀 TrendPilot: K-Beauty 글로벌 진출 전략 & 크리에이티브 에이전트")
    gr.Markdown("제품 이미지 1장으로 **현지화 전략 보고서**부터 **SNS 크리에이티브 시안**까지 자동 생성합니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="제품 사진")
            country_input = gr.Textbox(label="목표 국가 (예: 미국, 일본, 유럽)", value="일본")
            platform_input = gr.Dropdown(choices=["Instagram", "TikTok", "Amazon", "Shopee"], value="Instagram", label="목표 플랫폼")
            copy_lang_input = gr.Dropdown(choices=["한국어", "타겟 국가 언어"], value="한국어", label="카피 출력 언어")
            submit_btn = gr.Button("🚀 패키지 자동 생성", variant="primary")
            status_text = gr.Markdown("대기 중...")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("1. 제품 프로필 (Vision)"):
                    profile_out = gr.Code(language="json")
                with gr.TabItem("2. 컨텍스트 팩 (RAG)"):
                    context_out = gr.Code(language="json")
                with gr.TabItem("3. 전략 보고서 (Strategy)"):
                    strategy_out = gr.Code(language="json")
                with gr.TabItem("4. 크리에이티브 (Execution)"):
                    creative_out = gr.Textbox(label="광고 카피 (복사/붙여넣기용)", lines=16)
                    img_out = gr.Image(label="광고 이미지 결과물")
                    
    submit_btn.click(
        fn=run_pipeline,
        inputs=[img_input, country_input, platform_input, copy_lang_input],
        outputs=[status_text, profile_out, context_out, strategy_out, creative_out, img_out]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
