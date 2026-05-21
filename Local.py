import os
import json
import io
import time
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

# Google GenAI (解析用)
from google import genai
from google.genai import types

# =========================================================
# 環境変数・設定
# =========================================================
load_dotenv("API.env")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# --- 解析用データ構造 ---
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
    fx_usd: float = 0.0
    regional_sales: Optional[RegionalSales] = None

class ReportSchema(BaseModel):
    company_name: str
    prior_h1_actual: FinancialMetrics
    h1_actual: FinancialMetrics
    full_year_forecast: Optional[FinancialMetrics] = None

# --- OEM 設定 ---
OEM_CONFIG = {
    "Toyota": {"JP_name": "トヨタ自動車", "analysis_note": "標準的な連結数値を抽出してください。"},
    "Nissan": {"JP_name": "日産自動車", "analysis_note": "連結ベースの数値を優先してください。"},
    "Honda": {"JP_name": "本田技研工業", "analysis_note": ""},
    "Mazda": {"JP_name": "マツダ株式会社", "analysis_note": "グローバル販売台数と連結財務数値を抽出してください。"},
    "Mitsubishi": {"JP_name": "三菱自動車", "analysis_note": "連結財務数値を抽出してください。"},
    "Suzuki": {"JP_name": "スズキ株式会社", "analysis_note": "四輪事業を主軸に数値を抽出してください。"},
    "Isuzu": {"JP_name": "いすゞ自動車", "analysis_note": "【必須】CVとLCVの各セグメントの地域別売上を合算して算出してください。"},
    "Hino": {"JP_name": "日野自動車株式会社", "analysis_note": "連結数値を抽出してください。"},
    "Subaru": {"JP_name": "株式会社SUBARU", "analysis_note": "連結数値を抽出してください。"},
}

# =========================================================
# AI 解析ロジック
# =========================================================
def process_pdf_bytes(pdf_bytes, oem_name):
    config = OEM_CONFIG[oem_name]
    client = genai.Client(api_key=GEMINI_KEY)

    # Gemini 1.5 Flash などのマルチモーダルモデルを使用
    # リトライループの追加
    for attempt in range(3):
        gemini_file = client.files.upload(file=io.BytesIO(pdf_bytes), config={'mime_type': 'application/pdf'})
        try:
            prompt = f"""
            Extract financial and regional sales results for {oem_name} ({config['JP_name']}) by following these strict logical rules.
            Specific Note for this OEM: {config['analysis_note']}

            【1. UNIT CONVERSION LOGIC】
            Target currency unit: "Billion JPY" (1,000,000,000 JPY).
            Identify the unit label (百万円, 億円, 兆円) and apply:
            - Millions (百万円): Value / 1,000
            - 100 Millions (億円): Value / 10
            - Trillions (兆円): Value * 1,000

            【2. PERCENTAGE FORMAT RULE】
            - For "operating_margin_pct", extract the value as a whole percentage number (e.g., 8.1).

            【3. ISUZU-STYLE SEGMENT INTEGRATION】
            - If regional data is split by segment (e.g., CV and LCV), you MUST SUM them.
            
            【4. REGIONAL MAPPING DEFINITION】
            - "asia_excl_japan": Sum of all Asia regions but EXCLUDING Japan.
            - "other": Sum of all remaining regions (Middle East, Africa, Oceania, Central/South America).

            【5. DATA HIERARCHY】
            - Financials (Revenue/Income): Use top-level "Consolidated" totals only.
            - Volume: Use only "Automobile" business. IGNORE Motorcycles.
            """
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite", # 最新モデルを使用
                contents=[gemini_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", 
                    response_schema=ReportSchema, 
                    temperature=0.0
                ),
            )
            client.files.delete(name=gemini_file.name)
            return response.parsed
        except Exception as e:
            if gemini_file:
                try: client.files.delete(name=gemini_file.name)
                except: pass
            if ("503" in str(e) or "overloaded" in str(e).lower()) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise e

# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.set_page_config(page_title="Auto OEM Local Analyser", page_icon="🚗", layout="wide")
    st.title("🚗 Auto OEM Local IR Analyser")
    st.markdown("ローカルにある決算説明資料（PDF）をアップロードして解析します。")

    with st.sidebar:
        st.header("1. ファイルアップロード")
        uploaded_files = st.file_uploader("決算PDFを選択（複数可）", type="pdf", accept_multiple_files=True)
        
        st.divider()
        st.header("2. 設定")
        start_button = st.button("解析を開始", type="primary", width='stretch', disabled=not uploaded_files)

    if not uploaded_files:
        st.info("左側のサイドバーからPDFファイルをアップロードしてください。")
        return

    # ファイルとメーカーの紐付け設定
    st.subheader("📁 アップロード済みファイルの確認")
    mapping = {}
    cols = st.columns(2)
    for i, file in enumerate(uploaded_files):
        with cols[i % 2]:
            with st.expander(f"📄 {file.name}", expanded=True):
                # ファイル名からメーカーを推測（簡易的）
                default_index = 0
                for idx, oem in enumerate(OEM_CONFIG.keys()):
                    if oem.lower() in file.name.lower():
                        default_index = idx
                        break
                
                selected_oem = st.selectbox(
                    f"メーカーを選択 ({file.name})", 
                    list(OEM_CONFIG.keys()), 
                    index=default_index,
                    key=f"select_{i}"
                )
                mapping[file.name] = selected_oem

    if start_button:
        if not GEMINI_KEY:
            st.error("GEMINI_API_KEY が設定されていません。")
            return

        all_results_rows = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            oem = mapping[file.name]
            company_jp = OEM_CONFIG[oem]["JP_name"]
            
            with st.status(f"[{i+1}/{len(uploaded_files)}] {company_jp} ({file.name}) を解析中...") as status:
                try:
                    # PDF読み込み
                    pdf_bytes = file.read()
                    
                    # AI解析実行
                    res_data = process_pdf_bytes(pdf_bytes, oem)
                    
                    # 結果の整理
                    periods_map = [
                        (res_data.prior_h1_actual, 'Prior Year (H1)'), 
                        (res_data.h1_actual, 'Current Year (H1)'), 
                        (res_data.full_year_forecast, 'Full Year Forecast')
                    ]
                    for m, label in periods_map:
                        if m and m.revenue > 0:
                            reg = m.regional_sales or RegionalSales()
                            all_results_rows.append({
                                "Company": company_jp,
                                "Period": label,
                                "Revenue": m.revenue,
                                "OpIncome": m.operating_income,
                                "Margin": m.operating_margin_pct,
                                "Total Vol": m.volume,
                                "Japan": reg.japan, "NA": reg.north_america, "Europe": reg.europe,
                                "Asia(ex.JP)": reg.asia_excl_japan, "Other": reg.other
                            })
                    status.update(label=f"✅ {company_jp} 解析完了", state="complete")
                except Exception as e:
                    st.error(f"{company_jp} 解析エラー: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            # API負荷軽減のための待機
            if i < len(uploaded_files) - 1:
                time.sleep(2)

        if all_results_rows:
            render_results(all_results_rows)

def render_results(all_results_rows):
    st.divider()
    st.header("📊 解析結果集計")
    df = pd.DataFrame(all_results_rows)

    # 1. データテーブル
    st.subheader("📋 財務データ一覧")
    st.dataframe(
        df.style.format({
            "Revenue": "{:,.1f}", "OpIncome": "{:,.1f}", "Margin": "{:.1f}%",
            "Total Vol": "{:,.0f}", "Japan": "{:,.0f}", "NA": "{:,.0f}", 
            "Europe": "{:,.0f}", "Asia(ex.JP)": "{:,.0f}", "Other": "{:,.0f}"
        }, na_rep="-")
        .background_gradient(subset=["Margin"], cmap="Greens", vmin=0, vmax=12)
        .map(lambda x: 'color: #E74C3C; font-weight: bold;' if isinstance(x, (int, float)) and x < 0 else '', subset=["OpIncome"]),
        width='stretch', hide_index=True
    )

    # 2. グラフ表示
    st.subheader("📈 メーカー間比較 (H1実績)")
    
    df_current = df[df["Period"] == "Current Year (H1)"].copy().sort_values(by="Revenue", ascending=False)
    df_prior_raw = df[df["Period"] == "Prior Year (H1)"].copy()
    ordered_companies = df_current["Company"].tolist()
    df_prior = df_prior_raw.set_index("Company").reindex(ordered_companies).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(name='Prior Year', x=df_prior["Company"], y=df_prior["Revenue"], marker_color='#FFB399'))
        fig_rev.add_trace(go.Bar(name='Current Year', x=df_current["Company"], y=df_current["Revenue"], marker_color='#FF4500'))
        fig_rev.update_layout(title="<b>Revenue</b> (Billion JPY)", barmode='group', template="plotly_white")
        st.plotly_chart(fig_rev, use_container_width=True)

    with c2:
        fig_inc = go.Figure()
        fig_inc.add_trace(go.Bar(name='Prior Year', x=df_prior["Company"], y=df_prior["OpIncome"], marker_color='#A992E2'))
        fig_inc.add_trace(go.Bar(name='Current Year', x=df_current["Company"], y=df_current["OpIncome"], marker_color='#483D8B'))
        fig_inc.update_layout(title="<b>Operating Income</b> (Billion JPY)", barmode='group', template="plotly_white")
        st.plotly_chart(fig_inc, use_container_width=True)

if __name__ == "__main__":
    main()