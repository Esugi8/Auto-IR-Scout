import asyncio
import sys
import os
import json
import re
import urllib.parse
import time
import io
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional
from datetime import datetime

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from playwright.async_api import async_playwright
from browser_use import ChatOpenAI
from browser_use.llm.messages import UserMessage

# Google GenAI (解析用)
from google import genai
from google.genai import types

# WindowsでのNotImplementedError対策
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nest_asyncio.apply()

# =========================================================
# 環境変数・設定
# =========================================================
load_dotenv("API.env")
VIO_TOKEN = os.getenv("API_TOKEN")
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
    "Toyota": {"JP_name": "トヨタ自動車", "filename": "決算報告プレゼンテーション資料", "analysis_note": "標準的な連結数値を抽出してください。"},
    "Nissan": {"JP_name": "日産自動車", "filename": "プレゼンテーション資料", "analysis_note": "連結ベースの数値を優先してください。"},
    "Honda": {"JP_name": "本田技研工業", "filename": "決算説明会資料", "analysis_note": ""},
    "Mazda": {"JP_name": "マツダ株式会社", "filename": "プレゼンテーション資料", "analysis_note": "グローバル販売台数と連結財務数値を抽出してください。"},
    "Mitsubishi": {"JP_name": "三菱自動車", "filename": "プレゼンテーション資料", "analysis_note": "連結財務数値を抽出してください。"},
    "Suzuki": {"JP_name": "スズキ株式会社", "filename": "決算説明会", "analysis_note": "四輪事業を主軸に数値を抽出してください。"},
    "Isuzu": {"JP_name": "いすゞ自動車", "filename": "決算説明会資料", "analysis_note": "【必須】CVとLCVの各セグメントの地域別売上を合算して算出してください。"},
    "Hino": {"JP_name": "日野自動車株式会社", "filename": "決算説明会資料", "analysis_note": "連結数値を抽出してください。"},
    "Subaru": {"JP_name": "株式会社SUBARU", "filename": "アナリスト向け説明会資料", "analysis_note": "連結数値を抽出してください。"},
}

TAB_CONTROL_OEMS = ["Nissan", "Mazda"]

# =========================================================
# 補助クラス・関数 (探索用)
# =========================================================
def classify_page(url: str) -> str:
    u = url.lower()
    if any(k in u for k in ["financialresult.html", "results.html", "financial_results", "archives", "financialaffairs", "library/result", "earning"]):
        return "GOAL_LIBRARY"
    if "bing.com" in u or "google.com" in u:
        return "SEARCH"
    return "INDEX"

class LLMDecision(BaseModel):
    action: str 
    href: Optional[str] = None
    pdf_url: Optional[str] = None

    @field_validator('action')
    def normalize_action(cls, v):
        v = v.lower()
        if v in ['open', 'select', 'goto']: return 'click'
        if v in ['finish', 'complete']: return 'done'
        return v

async def handle_cookie_banner(page, log_func):
    keywords = ["同意", "受け入れる", "Accept", "OK", "閉じる"]
    try:
        for btn in await page.query_selector_all("button, a"):
            text = await btn.inner_text()
            if any(k in text for k in keywords) and await btn.is_visible():
                log_func(f"🍪 [BROWSER] クッキー処理: {text}")
                await btn.click()
                await asyncio.sleep(1)
                return
    except: pass

async def extract_links(page) -> List[dict]:
    if page.url.lower().endswith(".pdf"): return []
    # 修正点：endswith -> endsWith (JavaScriptの正しい構文)
    return await page.evaluate("""
    () => Array.from(document.querySelectorAll("a"))
      .filter(a => {
        const isPdf = a.href && a.href.toLowerCase().endsWith('.pdf');
        if (isPdf) return true;
        const s = getComputedStyle(a);
        return s && s.display !== 'none' && s.visibility !== 'hidden' && a.offsetWidth > 0;
      })
      .map(a => {
        const row = a.closest('tr') || a.closest('div');
        const context = row ? (row.innerText || "").replace(/\\n/g, ' ').trim().slice(0, 80) : "";
        return {
          text: `[Context: ${context}] -> ${a.innerText.trim()}`,
          href: a.href || ''
        };
      })
      .filter(x => x.text && x.href && x.href.startsWith('http'));
    """)

async def llm_decide(llm, current_url, links, company_name, period, alt_period, filename, oem_choice, log_func) -> LLMDecision:
    is_interactive = oem_choice in TAB_CONTROL_OEMS
    tab_rule = "- 目的の年度が表示されていない場合は、年度タブの切り替え(action='click', href=null)を最優先してください。" if is_interactive else ""
    prompt = f"""
    あなたはWeb操作の判断エージェントです。必ずJSONで回答してください。
    目的：{company_name} の {period}（別名: {alt_period}）の「{filename}」PDFを見つけること。
    【重要ルール】
    - 「{period}」と「{alt_period}」は同じ対象を指しています。
    - リンク一覧で「(★既にこのページにいます)」と書かれたリンクを再度クリックしても無意味です。
    {tab_rule}
    返答形式： {{{{ "action": "click", "href": "..." }}}} または {{{{ "action": "done", "pdf_url": "..." }}}}

    リンク一覧：{json.dumps(links, ensure_ascii=False)}
    """
    log_func(f"🧠 [LLM] 判断中... (候補リンク数: {len(links)} 件)")
    response = await llm.ainvoke([UserMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    clean_content = re.sub(r'^.*?({.*}).*$', r'\1', content.replace('\n', ''), flags=re.DOTALL)
    return LLMDecision(**json.loads(clean_content))

async def llm_judge_pdf_match(llm, company_name, period, alt_period, filename, pdf_candidates, log_func):
    prompt = f"""あなたは資料判定アシスタントです。
    目的：{company_name} の 「{period}」（別名: {alt_period}）の 「{filename}」として正しいPDFを1つ選んでください。
    【最重要ルール】
    - 「{period}」と「{alt_period}」は全く同じ対象です。
    候補：{json.dumps([{"index": i, "text": p["text"], "url": p["href"]} for i, p in enumerate(pdf_candidates)], ensure_ascii=False)}
    回答形式：{{{{ "match": true, "index": <番号> }}}} または {{{{ "match": false }}}}"""
    log_func("🧠 [LLM] PDF精密一致判定中...")
    response = await llm.ainvoke([UserMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    try:
        data = json.loads(re.search(r'\{.*\}', content, re.DOTALL).group())
        return data["index"] if data.get("match") is True else None
    except: return None

# =========================================================
# メイン探索ロジック
# =========================================================
async def run_search(oem_choice, period, log_area, headless):
    current_logs = []
    def st_log(msg):
        current_logs.append(msg)
        log_area.code("\n".join(current_logs))

    config = OEM_CONFIG[oem_choice]
    company_name, filename = config["JP_name"], config["filename"]

    match_full = re.search(r'(\d{4})年3月期', period)
    alt_period = period
    if match_full:
        year = int(match_full.group(1))
        alt_period = period.replace(f"{year}年3月期", f"{year - 1}年度")

    match_y = re.search(r'(\d{4})年', period)
    target_simple_year = f"{match_y.group(1)}年" if match_y else ""

    llm = ChatOpenAI(model="VIO:GPT-5", api_key=VIO_TOKEN, base_url="https://vio.automotive-wan.com:446", timeout=120)
    visited = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless) 
        page = await browser.new_page()

        try:
            start_url = f"https://www.bing.com/search?q={urllib.parse.quote(f'{company_name} 決算報告')}"
            st_log(f"🌐 [START] Bing検索開始")
            await page.goto(start_url, wait_until="load", timeout=60000)

            for step in range(15):
                current_url = page.url
                page_type = classify_page(current_url)
                st_log(f"\n--- 🚀 STEP {step + 1} ---\n📍 URL: {current_url}")

                await handle_cookie_banner(page, st_log)
                links = await extract_links(page)
                
                formatted_links = []
                pdf_candidates = []
                for l in links:
                    display_text = l["text"]
                    if l["href"].split('#')[0].rstrip('/') == current_url.split('#')[0].rstrip('/'):
                        display_text += " (★既にこのページにいます)"
                    formatted_links.append({"text": display_text, "href": l["href"]})
                    if l["href"].lower().endswith(".pdf"):
                        pdf_candidates.append(l)

                if len(pdf_candidates) >= 3 or page_type == "GOAL_LIBRARY":
                    if pdf_candidates:
                        idx = await llm_judge_pdf_match(llm, company_name, period, alt_period, filename, pdf_candidates, st_log)
                        if idx is not None:
                            return pdf_candidates[idx]["href"]
                        st_log("⚠️ [MATCH] 合致なし。探索を継続します。")

                decision = await llm_decide(llm, current_url, formatted_links[:50], company_name, period, alt_period, filename, oem_choice, st_log)

                if decision.action == "done" and decision.pdf_url: return decision.pdf_url

                if decision.action == "click":
                    is_same_url = decision.href and decision.href.split('#')[0].rstrip('/') == current_url.split('#')[0].rstrip('/')
                    if decision.href and not is_same_url:
                        if oem_choice not in TAB_CONTROL_OEMS and decision.href in visited:
                            st_log("⛔ [LOOP] 遷移防止"); break
                        visited.add(decision.href)
                        st_log(f"🖱️ [BROWSER] 遷移中: {decision.href}")
                        await page.goto(decision.href, wait_until="domcontentloaded", timeout=60000)
                    elif oem_choice in TAB_CONTROL_OEMS:
                        target_fy, target_full = alt_period.split()[0], period.split()[0]
                        candidates = [target_fy, target_full, target_simple_year]
                        for label in candidates:
                            if not label: continue
                            try:
                                await page.select_option("select", label=label, timeout=3000)
                                st_log(f"✅ [BROWSER] 選択成功: {label}"); break
                            except: continue
                        try: await page.wait_for_selector(f"text={filename}", timeout=5000)
                        except: pass
                    else: break
                await asyncio.sleep(2)
                continue
            return "PDFは見つかりませんでした"
        finally:
            await browser.close()

# =========================================================
# AI 解析ロジック (503エラー対策リトライ実装)
# =========================================================
def process_pdf_bytes(pdf_bytes, oem_name):
    config = OEM_CONFIG[oem_name]
    client = genai.Client(api_key=GEMINI_KEY)

    # 修正点：リトライループの追加
    for attempt in range(3):
        gemini_file = client.files.upload(file=io.BytesIO(pdf_bytes), config={'mime_type': 'application/pdf'})
        try:
            prompt = f"""
            Extract financial and regional sales results by following these strict logical rules.
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
            - You MUST find the regional data for EACH segment and SUM them to calculate the total regional sales.
            
            【4. REGIONAL MAPPING DEFINITION】
            - "asia_excl_japan": Sum of all Asia regions but EXCLUDING Japan.
            - "other": Sum of all remaining regions (Middle East, Africa, Oceania, Central/South America).

            【5. DATA HIERARCHY】
            - Financials (Revenue/Income): Use top-level "Consolidated" totals only.
            - Volume: Use only "Automobile" business. IGNORE Motorcycles.
            """
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[gemini_file, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ReportSchema, temperature=0.0),
            )
            client.files.delete(name=gemini_file.name)
            return response.parsed
        except Exception as e:
            # 修正点：リトライ判断
            if gemini_file:
                try: client.files.delete(name=gemini_file.name)
                except: pass
            if ("503" in str(e) or "overloaded" in str(e).lower()) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise e

# =========================================================
# Streamlit UI & メインオーケストレーション
# =========================================================
def main():
    st.set_page_config(page_title="Auto OEM IR Analyser", page_icon="🚗", layout="wide")
    st.title("🚗 Auto OEM IR Collector & Analyser")

    with st.sidebar:
        st.header("設定")
        all_oems = list(OEM_CONFIG.keys())
        select_all = st.checkbox("全てのメーカーを選択")
        selected_oems = st.multiselect("メーカーを選択", all_oems, default=all_oems if select_all else ["Toyota"])
        
        current_year = datetime.now().year
        years = [f"{y}年3月期 ({y-1}年度)" for y in range(current_year, 2019, -1)]
        selected_year_text = st.selectbox("年度を選択", years)
        target_year = selected_year_text.split(" (")[0]
        
        quarters = ["第1四半期", "第2四半期", "第3四半期", "第4四半期"]
        selected_q = st.selectbox("四半期を選択", quarters, index=1)
        period_str = f"{target_year} {selected_q}"

        st.divider()
        headless_mode = st.toggle("Headlessモード", value=True)
        start_button = st.button("探索・解析を開始", type="primary", width='stretch')

    if start_button:
        if not VIO_TOKEN or not GEMINI_KEY:
            st.error("APIキーが不足しています。")
            return

        all_results_rows = []

        for i, oem in enumerate(selected_oems):
            company_jp = OEM_CONFIG[oem]["JP_name"]
            
            with st.status(f"[{i+1}/{len(selected_oems)}] {company_jp} を処理中...", expanded=True) as status:
                status.write("🔍 PDFを探索中...")
                log_placeholder = st.empty()
                pdf_url = asyncio.run(run_search(oem, period_str, log_placeholder, headless_mode))
                
                if not pdf_url or not pdf_url.startswith("http"):
                    st.error(f"PDFが見つかりませんでした: {company_jp}")
                    continue
                
                status.write("📥 ダウンロード中...")
                pdf_resp = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
                
                status.write("🧠 AI解析中...")
                try:
                    res_data = process_pdf_bytes(pdf_resp.content, oem)
                    
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
                    status.update(label=f"✅ {company_jp} 完了", state="complete")
                except Exception as e:
                    st.error(f"{company_jp} 解析エラー: {e}")

            if i < len(selected_oems) - 1:
                wait_box = st.empty()
                for r in range(10, 0, -1):
                    wait_box.info(f"⏳ 次のメーカーまであと {r} 秒待機中...")
                    time.sleep(1)
                wait_box.empty()

        if all_results_rows:
            st.divider()
            st.header("📊 全社集計結果")
            df = pd.DataFrame(all_results_rows)

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

            st.subheader("📈 メーカー間比較 (H1実績)")
            c1, c2 = st.columns(2)
            
            # 修正点：売上高順にソートするロジック
            df_current = df[df["Period"] == "Current Year (H1)"].copy().sort_values(by="Revenue", ascending=False)
            df_prior_raw = df[df["Period"] == "Prior Year (H1)"].copy()
            ordered_companies = df_current["Company"].tolist()
            df_prior = df_prior_raw.set_index("Company").reindex(ordered_companies).reset_index()

            with c1: # Revenue 比較
                fig_rev = go.Figure()
                fig_rev.add_trace(go.Bar(name='Prior Year', x=df_prior["Company"], y=df_prior["Revenue"], marker_color='#FFB399'))
                fig_rev.add_trace(go.Bar(name='Current Year', x=df_current["Company"], y=df_current["Revenue"], marker_color='#FF4500'))
                fig_rev.update_layout(title="<b>Revenue</b> (Billion JPY)", barmode='group', paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2')
                st.plotly_chart(fig_rev, width='stretch')

            with c2: # OpIncome 比較
                fig_inc = go.Figure()
                fig_inc.add_trace(go.Bar(name='Prior Year', x=df_prior["Company"], y=df_prior["OpIncome"], marker_color='#A992E2'))
                fig_inc.add_trace(go.Bar(name='Current Year', x=df_current["Company"], y=df_current["OpIncome"], marker_color='#483D8B'))
                fig_inc.update_layout(title="<b>Operating Income</b> (Billion JPY)", barmode='group', paper_bgcolor='#F2F2F2', plot_bgcolor='#F2F2F2')
                st.plotly_chart(fig_inc, width='stretch')

if __name__ == "__main__":
    main()