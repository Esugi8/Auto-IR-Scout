import os
import json
import pandas as pd
import plotly.graph_objects as go
import pymupdf4llm  # ローカルPDF解析
import tempfile
from typing import Optional
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

# 一番最初のコードと同じインポート
from langchain_openai import ChatOpenAI

# =========================================================
# 環境変数・設定
# =========================================================
load_dotenv("API.env")
API_TOKEN = os.getenv("API_TOKEN")

# --- データ構造定義 ---
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

# OEM 設定
OEM_CONFIG = {
    "Toyota": {"JP_name": "トヨタ自動車", "analysis_note": "標準的な連結数値を抽出。"},
    "Nissan": {"JP_name": "日産自動車", "analysis_note": "連結ベースの数値を優先。"},
    "Honda": {"JP_name": "本田技研工業", "analysis_note": ""},
    "Mazda": {"JP_name": "マツダ", "analysis_note": "グローバル販売台数と連結財務数値。"},
    "Mitsubishi": {"JP_name": "三菱自動車", "analysis_note": "連結財務数値。"},
    "Suzuki": {"JP_name": "スズキ", "analysis_note": "四輪事業を主軸。"},
    "Isuzu": {"JP_name": "いすゞ自動車", "analysis_note": "CVとLCVの各セグメントの地域別売上を合算して算出。"},
    "Hino": {"JP_name": "日野自動車", "analysis_note": "連結数値。"},
    "Subaru": {"JP_name": "株式会社SUBARU", "analysis_note": "連結数値。"},
}


### =========================================================
### ロジック1.5: Markdownの構造補正（★ここだけ追加）
### =========================================================
 
def normalize_regional_data(md_text: str):
    region_keys = [
        "japan",
        "north_america",
        "europe",
        "asia_excl_japan",
        "other"
    ]

    numbers = re.findall(r"\b\d{2,5}(?:,\d{3})*\b", md_text)
    numbers = [int(n.replace(",", "")) for n in numbers]

    if len(numbers) < 10:
        return None

    return {
        "prior": dict(zip(region_keys, numbers[:5])),
        "current": dict(zip(region_keys, numbers[5:10]))
    }

# =========================================================
# ロジック1: ローカルで PDF -> Markdown 変換
# =========================================================
def pdf_to_markdown_locally(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        # 表構造を維持してテキスト化
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return md_text

# =========================================================
# ロジック2: VIO:GPT-5 で解析 (最初のコードの呼び出し方を踏襲)
# =========================================================
def analyze_with_vio(markdown_text, oem_name):
    config = OEM_CONFIG[oem_name]
    
    llm = ChatOpenAI(
        model="VIO:GPT-5",
        api_key=API_TOKEN,
        base_url="https://vio.automotive-wan.com:446",
        temperature=0
    )

    
    # ★ここだけ追加（前置するだけ）
    normalized = normalize_regional_data(markdown_text)
    if normalized:
        markdown_text = (
            "[STRUCTURED_REGION_DATA]
            json.dumps(normalized, ensure_ascii=False) 
            "
            markdown_text
        )


    # プロンプトをより厳格に
    prompt = f"""
    Extract financial results for {oem_name} from the Markdown text.
    You MUST return the data in the EXACT JSON format shown below. 
    Do not change the key names. Do not use names like "volume_kunits" or "company".

    【STRICT JSON TEMPLATE】
    {{
      "company_name": "string",
      "prior_h1_actual": {{
        "revenue": 0.0,
        "operating_income": 0.0,
        "operating_margin_pct": 0.0,
        "volume": 0.0,
        "regional_sales": {{
          "japan": 0.0, "north_america": 0.0, "europe": 0.0, "asia_excl_japan": 0.0, "other": 0.0
        }}
      }},
      "h1_actual": {{
        "revenue": 0.0, "operating_income": 0.0, "operating_margin_pct": 0.0, "volume": 0.0,
        "regional_sales": {{ ... }}
      }},
      "full_year_forecast": {{
        "revenue": 0.0, "operating_income": 0.0, "operating_margin_pct": 0.0, "volume": 0.0,
        "regional_sales": {{ ... }}
      }}
    }}

    【RULES】
    
    1. UNIT: Convert all money values to "Billion JPY".
    - Millions (百万円) -> Value / 1,000
    - 100 Millions (億円) -> Value / 10
    - Trillions (兆円) -> Value * 1,000
    2. MARGIN: Operating margin as a percentage (e.g., 8.5).
    3. REGIONS: 
       - "asia_excl_japan": All Asia except Japan.
       - "other": Sum of Middle East, Africa, Oceania, and Latin America.
    4. Return result in valid JSON ONLY matching the defined schema.

    【MARKDOWN TEXT】
    {markdown_text[:40000]}
    """

    response = llm.invoke(prompt)
    raw_content = response.content
    
    # JSONの切り出し
    try:
        start_idx = raw_content.find('{')
        end_idx = raw_content.rfind('}') + 1
        json_str = raw_content[start_idx:end_idx]
        data_dict = json.loads(json_str)
        
        # Pydanticモデルへ変換（ここでエラーが出るのを防ぐために、不足キーの補完を検討しても良いですが、まずは厳格化で対応）
        return ReportSchema(**data_dict)
    except Exception as e:
        st.error(f"AIの応答形式が不正です。生データ: {raw_content[:500]}")
        raise e

# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.set_page_config(page_title="VIO IR Analyser", page_icon="🚗", layout="wide")
    st.title("🚗 Auto OEM IR Analyser (VIO Style)")
    st.markdown("最初の呼び出し仕様を維持しつつ、ローカルPDFからデータを抽出します。")

    with st.sidebar:
        st.header("1. PDFアップロード")
        uploaded_files = st.file_uploader("PDFを選択", type="pdf", accept_multiple_files=True)
        start_button = st.button("解析を開始", type="primary", width='stretch', disabled=not uploaded_files)

    if not uploaded_files:
        st.info("サイドバーからPDFをアップロードしてください。")
        return

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
        all_rows = []
        for file in uploaded_files:
            oem = mapping[file.name]
            with st.status(f"【{oem}】解析中...") as status:
                try:
                    status.write("PDFをMarkdownに変換中...")
                    md_text = pdf_to_markdown_locally(file.read())
                    
                    status.write("VIO:GPT-5 で財務データを抽出中...")
                    res_data = analyze_with_vio(md_text, oem)
                    
                    for m, label in [(res_data.prior_h1_actual, 'Prior Year (H1)'), 
                                     (res_data.h1_actual, 'Current Year (H1)'), 
                                     (res_data.full_year_forecast, 'Full Year Forecast')]:
                        if m and m.revenue > 0:
                            reg = m.regional_sales or RegionalSales()
                            all_rows.append({
                                "Company": OEM_CONFIG[oem]["JP_name"], "Period": label,
                                "Revenue": m.revenue, "OpIncome": m.operating_income,
                                "Margin": m.operating_margin_pct, "Total Vol": m.volume,
                                "Japan": reg.japan, "NA": reg.north_america, "Europe": reg.europe,
                                "Asia(ex.JP)": reg.asia_excl_japan, "Other": reg.other
                            })
                    status.update(label=f"✅ {oem} 完了", state="complete")
                except Exception as e:
                    st.error(f"{oem} エラー: {e}")

        if all_rows:
            st.divider()
            st.subheader("📋 解析結果")
            df = pd.DataFrame(all_rows)
            st.dataframe(df.style.format("{:,.1f}", subset=["Revenue", "OpIncome"]), width='stretch', hide_index=True)

if __name__ == "__main__":
    main()