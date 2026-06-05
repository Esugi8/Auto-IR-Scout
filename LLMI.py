import os
import json
import base64
import time
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
# 既存のインポートに加えてこれを追加
from langchain_openai import ChatOpenAI

# =========================================================
# 環境変数・設定
# =========================================================
load_dotenv("API.env")

# Mistral OCR 設定 (提示されたスクリプトを元に設定)
OCR_CFG = {
    "endpoint": os.environ.get("LLMI_BASE_URL", "https://api.llm-incubator.automotive.cloud/prod/v0/llm").rstrip('/') + "/ocr",
    "api_key": os.environ.get("LLMI_PROXY_TOKEN"),
    "username": os.environ.get("MY_USERNAME"),
    "model": os.environ.get("LLMI_OCR_MODEL", "mistral-document-ai-2505"),
}

# VIO:GPT-5 解析用設定
VIO_CFG = {
    "url": "https://vio.automotive-wan.com:446",
    "api_key": os.environ.get("API_TOKEN"), # または LLMI_PROXY_TOKEN
    "model": "VIO:GPT-5"
}

# --- データ構造 ---
class RegionalSales(BaseModel):
    japan: Optional[float] = None
    north_america: Optional[float] = None
    europe: Optional[float] = None
    asia_excl_japan: Optional[float] = None
    other: Optional[float] = None

class FinancialMetrics(BaseModel):
    revenue: float               
    operating_income: float      
    operating_margin_pct: float  
    volume: float                
    regional_sales: Optional[RegionalSales] = None

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: Optional[FinancialMetrics] = None

# OEM 設定 (前回同様)
OEM_CONFIG = {
    "Toyota": {"JP_name": "トヨタ自動車", "analysis_note": "標準的な連結数値を抽出してください。"},
    "Nissan": {"JP_name": "日産自動車", "analysis_note": "連結ベースの数値を優先してください。"},
    "Honda": {"JP_name": "本田技研工業", "analysis_note": ""},
    "Mazda": {"JP_name": "マツダ株式会社", "analysis_note": "グローバル販売台数と連結財務数値を抽出。"},
    "Mitsubishi": {"JP_name": "三菱自動車", "analysis_note": "連結財務数値を抽出。"},
    "Suzuki": {"JP_name": "スズキ株式会社", "analysis_note": "四輪事業を主軸に。"},
    "Isuzu": {"JP_name": "いすゞ自動車", "analysis_note": "CVとLCVの各セグメントの地域別売上を合算して算出してください。"},
    "Hino": {"JP_name": "日野自動車", "analysis_note": "連結数値を抽出。"},
    "Subaru": {"JP_name": "株式会社SUBARU", "analysis_note": "連結数値を抽出。"},
}

# =========================================================
# STEP 1: Mistral OCR で PDF を Markdown に変換
# =========================================================
def pdf_to_markdown_via_mistral(pdf_bytes):
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    
    payload = {
        "model": OCR_CFG["model"],
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64_pdf}",
        },
    }
    headers = {
        "Authorization": f"Bearer {OCR_CFG['api_key']}",
        "X-End-User-Id": OCR_CFG["username"],
        "X-Application-Name": "mistral-pdf-analyser",
        "Content-Type": "application/json",
    }

    response = requests.post(OCR_CFG["endpoint"], headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    
    ocr_result = response.json()
    pages = ocr_result.get("pages", [])
    
    md_parts = []
    for page in pages:
        md_parts.append(page.get("markdown", "").strip())
    
    return "\n\n---\n\n".join(md_parts)

# =========================================================
# STEP 2: VIO:GPT-5 で Markdown を解析
# =========================================================
def analyze_markdown_with_vio(markdown_text, oem_name):
    config = OEM_CONFIG[oem_name]
    
    llm = ChatOpenAI(
        model=VIO_CFG["model"],
        api_key=VIO_CFG["api_key"],
        base_url="https://vio.automotive-wan.com:446",
        temperature=0
    )

    # 地域データを確実に見つけさせるための指示を「RULES」に追加
    prompt = f"""
    Extract financial results for {oem_name} ({config['JP_name']}) from the following Markdown text.
    You MUST return the data in the following EXACT JSON structure.

    【JSON TEMPLATE】
    {{
      "company_name": "{config['JP_name']}",
      "prior_h1_actual": {{
        "revenue": 0.0, "operating_income": 0.0, "operating_margin_pct": 0.0, "volume": 0.0,
        "regional_sales": {{
          "japan": 0.0, "north_america": 0.0, "europe": 0.0, "asia_excl_japan": 0.0, "other": 0.0
        }}
      }},
      "h1_actual": {{
        "revenue": 0.0, "operating_income": 0.0, "operating_margin_pct": 0.0, "volume": 0.0,
        "regional_sales": {{ "japan": 0.0, "north_america": 0.0, "europe": 0.0, "asia_excl_japan": 0.0, "other": 0.0 }}
      }},
      "full_year_forecast": {{
        "revenue": 0.0, "operating_income": 0.0, "operating_margin_pct": 0.0, "volume": 0.0,
        "regional_sales": {{ "japan": 0.0, "north_america": 0.0, "europe": 0.0, "asia_excl_japan": 0.0, "other": 0.0 }}
      }}
    }}

    【RULES】
    1. REGIONAL SALES: Find the "Consolidated Sales Volume" (連結販売台数) section. 
       Match the numbers to region labels (Japan, North America, etc.) and fill in "regional_sales". 
       Do NOT leave them as 0.0 if data is available.
    2. UNIT: Billion JPY for financial values.
    3. ROUNDING: All numeric values EXCEPT "operating_margin_pct" MUST be rounded to ONE decimal place.
    4. If a value is missing, set it to 0.0.

    【TEXT CONTENT】
    {markdown_text:50000}
    """

    response = llm.invoke(prompt)
    raw_content = response.content
    
    try:
        json_str = raw_content[raw_content.find('{'):raw_content.rfind('}')+1]
        return ReportSchema(**json.loads(json_str))
    except Exception as e:
        st.error(f"VIO:GPT-5 解析エラー: {e}")
        st.code(raw_content) 
        raise e

# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.set_page_config(page_title="Auto OEM OCR Analyser", page_icon="📊", layout="wide")
    st.title("📊 Auto OEM IR Analyser (Mistral OCR + VIO:GPT-5)")
    st.markdown("PDFをMistralで高精度OCRし、その結果をVIO:GPT-5で財務解析します。")

    with st.sidebar:
        st.header("1. 設定確認")
        if not OCR_CFG["api_key"] or not VIO_CFG["api_key"]:
            st.error("APIキーが設定されていません。API.envを確認してください。")
        
        st.header("2. ファイルアップロード")
        uploaded_files = st.file_uploader("決算PDFを選択", type="pdf", accept_multiple_files=True)
        start_button = st.button("一括解析を開始", type="primary", width='stretch', disabled=not uploaded_files)

    if not uploaded_files:
        st.info("サイドバーからPDFをアップロードしてください。")
        return

    # メーカー紐付け
    mapping = {}
    cols = st.columns(2)
    for i, file in enumerate(uploaded_files):
        with cols[i % 2]:
            with st.expander(f"📄 {file.name}", expanded=True):
                default_index = 0
                for idx, oem in enumerate(OEM_CONFIG.keys()):
                    if oem.lower() in file.name.lower():
                        default_index = idx; break
                mapping[file.name] = st.selectbox(f"メーカー", list(OEM_CONFIG.keys()), index=default_index, key=f"sel_{i}")

    if start_button:
        all_results = []
        for file in uploaded_files:
            oem = mapping[file.name]
            with st.status(f"【{oem}】処理中...") as status:
                try:
                    # Step 1: OCR
                    status.write("OCR実行中（Mistral-Document-AI）...")
                    markdown_text = pdf_to_markdown_via_mistral(file.read())
                    
                    # Step 2: Analysis
                    status.write("財務データ抽出中（VIO:GPT-5）...")
                    res_data = analyze_markdown_with_vio(markdown_text, oem)
                    
                    # 保存
                    for m, label in [(res_data.prior_h1_actual, 'Prior Year (H1)'), 
                                     (res_data.h1_actual, 'Current Year (H1)'), 
                                     (res_data.full_year_forecast, 'Full Year Forecast')]:
                        if m and m.revenue > 0:
                            reg = m.regional_sales or RegionalSales()
                            all_results.append({
                                "Company": OEM_CONFIG[oem]["JP_name"], "Period": label,
                                "Revenue": m.revenue, "OpIncome": m.operating_income,
                                "Margin": m.operating_margin_pct, "Total Vol": m.volume,
                                "Japan": reg.japan, "NA": reg.north_america, "Europe": reg.europe,
                                "Asia(ex.JP)": reg.asia_excl_japan, "Other": reg.other
                            })
                    status.update(label=f"✅ {oem} 完了", state="complete")
                except Exception as e:
                    st.error(f"{oem} でエラー: {e}")

        if all_results:
            render_summary(all_results)

def render_summary(rows):
    df = pd.DataFrame(rows)
    st.divider()
    st.subheader("📋 解析結果")
    st.dataframe(df.style.format("{:,.1f}", subset=["Revenue", "OpIncome", "Total Vol"]), width='stretch')
    
    # 簡易グラフ
    fig = go.Figure()
    df_h1 = df[df["Period"] == "Current Year (H1)"]
    fig.add_trace(go.Bar(x=df_h1["Company"], y=df_h1["Revenue"], name="Revenue"))
    fig.update_layout(title="H1 Revenue Comparison", template="plotly_white")
    st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()