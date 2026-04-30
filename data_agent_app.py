"""
DataAgent — CSV/Excel Analysis UI with Google Gemini
====================================================
"""

import os
import io
import json
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="DataAgent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#f7f8fa; }
[data-testid="stSidebar"] { background:#ffffff; border-right: 1px solid #e8eaed; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid #e8eaed; border-radius:10px; padding:12px 16px; }
.stButton > button { background:#1D9E75 !important; color:white !important; border:none !important; border-radius:8px !important; font-weight:500 !important; }
.stButton > button:hover { background:#0F6E56 !important; }
.insight-card { background:#ffffff; border:1px solid #e8eaed; border-radius:10px; padding:14px 16px; margin-bottom:10px; line-height:1.6; font-size:0.9rem; }
.insight-card.trend   { border-left:3px solid #1D9E75; }
.insight-card.warning { border-left:3px solid #EF9F27; }
.insight-card.info    { border-left:3px solid #378ADD; }
.badge { display:inline-block; padding:2px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; margin-right:6px; }
.badge-green { background:#E1F5EE; color:#0F6E56; }
.badge-amber { background:#FAEEDA; color:#854F0B; }
.badge-blue  { background:#E6F1FB; color:#185FA5; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📊 DataAgent")
    st.markdown("*CSV & Excel analysis powered by Gemini*")
    st.divider()
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        placeholder="AIzaSy..."
    )
    model_name = st.selectbox(
    "Model",
    ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
)
    max_tokens = st.slider("Max tokens", 512, 8192, 2048, 256)
    st.divider()
    st.markdown("**Analysis options**")
    do_summary  = st.checkbox("Summary & statistics", value=True)
    do_insights = st.checkbox("AI insights", value=True)
    do_charts   = st.checkbox("Auto charts", value=True)
    do_export   = st.checkbox("Export results", value=True)
    st.divider()
    chart_type = st.selectbox(
        "Preferred chart",
        ["Auto (agent decides)", "Bar", "Line", "Scatter", "Histogram", "Box"]
    )

# ---------- Helpers ----------
def load_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)

def df_summary(df):
    buf = io.StringIO()
    df.info(buf=buf)
    return (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Columns & types:\n{buf.getvalue()}\n\n"
        f"Null counts:\n{df.isnull().sum().to_string()}\n\n"
        f"Numeric stats:\n{df.describe().to_string()}\n\n"
        f"First 5 rows:\n{df.head(5).to_string()}"
    )

def ask_gemini(system, user):
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system,
        generation_config={"max_output_tokens": max_tokens}
    )
    response = model.generate_content(user)
    return response.text

def build_chart(df, spec):
    kind = spec.get("type", "bar").lower()
    x_col = spec.get("x")
    y_col = spec.get("y")
    title = spec.get("title", "")
    color = spec.get("color")
    cols = df.columns.tolist()
    if x_col not in cols:
        x_col = cols[0] if cols else None
    if y_col and y_col not in cols:
        y_col = None
    try:
        if kind == "bar":
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, title=title, color=color)
            else:
                c = df[x_col].value_counts().reset_index()
                c.columns = [x_col, "count"]
                fig = px.bar(c, x=x_col, y="count", title=title)
        elif kind == "line":
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif kind == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=title, color=color)
        elif kind == "histogram":
            fig = px.histogram(df, x=x_col, title=title)
        elif kind == "box":
            fig = px.box(df, x=x_col, y=y_col, title=title)
        else:
            c = df[x_col].value_counts().reset_index()
            c.columns = [x_col, "count"]
            fig = px.bar(c, x=x_col, y="count", title=title)
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font_color="#1a1a2e", title_font_size=14,
            margin=dict(t=40, l=10, r=10, b=10)
        )
        if kind in ("bar", "histogram"):
            fig.update_traces(marker_color="#1D9E75")
        return fig
    except Exception:
        return None

# ---------- Session state ----------
for key in ("df", "summary_text", "insights", "chart_specs"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- Main UI ----------
st.markdown("# 📊 DataAgent")
st.markdown("Upload a CSV or Excel file, then let Gemini analyze it for you.")

st.markdown("### Step 1 — Upload your file")
uploaded = st.file_uploader(
    "Drop your file here",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed"
)

if uploaded:
    try:
        df = load_file(uploaded)
        st.session_state.df = df
        st.success(f"Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

df = st.session_state.df

if df is not None:
    st.markdown("### Step 2 — Preview")
    with st.expander("Show data preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 1) if df.size > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric cols", len(num_cols))
    c4.metric("Missing", f"{null_pct}%")

    st.markdown("### Step 3 — Configure analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        x_col = st.selectbox("X-axis / Category column", df.columns.tolist(), index=0)
    with col_b:
        y_options = ["(auto — count)"] + num_cols
        y_col = st.selectbox("Y-axis / Metric column", y_options, index=0)
    y_col = None if y_col == "(auto — count)" else y_col
    focus = st.text_input(
        "Focus question for the agent (optional)",
        placeholder="e.g. What drives peak load? Are there anomalies?"
    )

    st.markdown("### Step 4 — Run the agent")
    if st.button("🚀  Analyze with Gemini"):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"Could not configure Gemini: {e}")
            st.stop()

        summary = df_summary(df)
        tabs = st.tabs(["Summary", "Charts", "Insights", "Export"])

        # --- Summary ---
        with tabs[0]:
            if do_summary:
                with st.spinner("Generating summary…"):
                    try:
                        text = ask_gemini(
                            "You are a data analyst. Write a concise plain-English overview: what the dataset contains, key statistics, data quality notes, standout patterns. Use bullet points.",
                            f"Dataset summary:\n{summary}\n\n" + (f"Focus: {focus}" if focus else "")
                        )
                        st.session_state.summary_text = text
                        st.markdown(text)
                    except Exception as e:
                        st.error(f"Summary failed: {e}")
            else:
                st.info("Summary disabled.")

        # --- Charts ---
        with tabs[1]:
            if do_charts:
                with st.spinner("Planning charts…"):
                    try:
                        raw = ask_gemini(
                            "You are a data visualization expert. Return ONLY valid JSON, no explanation, no markdown fences.",
                            f"Columns: {df.columns.tolist()}\nNumeric: {num_cols}\nCategorical: {cat_cols}\n"
                            f"User x: {x_col}, y: {y_col}\nPreferred: {chart_type}\n\n"
                            f"Return ONLY a JSON array of 1-3 chart specs. Each: "
                            f'{{"type":"bar|line|scatter|histogram|box","x":"col","y":"col_or_null","title":"...","color":"col_or_null"}}'
                        )
                        clean = raw.strip().replace("```json", "").replace("```", "").strip()
                        specs = json.loads(clean)
                        st.session_state.chart_specs = specs
                    except Exception:
                        specs = [{"type": "bar", "x": x_col, "y": y_col, "title": f"{x_col} distribution"}]
                    for spec in specs:
                        fig = build_chart(df, spec)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Could not render: {spec.get('title','')}")
            else:
                st.info("Charts disabled.")

        # --- Insights ---
        with tabs[2]:
            if do_insights:
                with st.spinner("Generating insights…"):
                    raw_ins = ""
                    try:
                        raw_ins = ask_gemini(
                            "You are a senior data analyst. Return ONLY a JSON array of 4-6 insight objects, no markdown fences. Each object has: \"text\" (1-2 sentences), \"type\" (trend|warning|info), \"tag\" (short label).",
                            f"Dataset summary:\n{summary}\n\n" + (f"Focus: {focus}\n\n" if focus else "") + "Generate sharp, specific insights."
                        )
                        clean = raw_ins.strip().replace("```json", "").replace("```", "").strip()
                        insights = json.loads(clean)
                        st.session_state.insights = insights
                    except Exception:
                        insights = [{"text": raw_ins or "No insights generated.", "type": "info", "tag": "General"}]
                    for ins in insights:
                        kind = ins.get("type", "info")
                        tag = ins.get("tag", "insight")
                        badge_cls = {"trend": "badge-green", "warning": "badge-amber"}.get(kind, "badge-blue")
                        st.markdown(
                            f'<div class="insight-card {kind}"><span class="badge {badge_cls}">{tag}</span>{ins["text"]}</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.info("Insights disabled.")

        # --- Export ---
        with tabs[3]:
            if do_export:
                st.markdown("#### Export options")
                col1, col2, col3 = st.columns(3)
                col1.download_button(
                    "📥 Download data (CSV)",
                    df.to_csv(index=False).encode(),
                    "analyzed_data.csv", "text/csv",
                    use_container_width=True
                )
                if st.session_state.summary_text:
                    col2.download_button(
                        "📄 Download summary (TXT)",
                        st.session_state.summary_text.encode(),
                        "summary.txt", "text/plain",
                        use_container_width=True
                    )
                if st.session_state.insights:
                    ins_df = pd.DataFrame(st.session_state.insights)
                    col3.download_button(
                        "💡 Download insights (CSV)",
                        ins_df.to_csv(index=False).encode(),
                        "insights.csv", "text/csv",
                        use_container_width=True
                    )
                if num_cols:
                    st.markdown("#### Numeric statistics")
                    stats_df = df[num_cols].describe().T.reset_index()
                    stats_df.columns = ["column"] + list(stats_df.columns[1:])
                    st.dataframe(stats_df, use_container_width=True)
                    st.download_button(
                        "📊 Download statistics (CSV)",
                        stats_df.to_csv(index=False).encode(),
                        "statistics.csv", "text/csv"
                    )
            else:
                st.info("Export disabled.")
else:
    st.info("Upload a CSV or Excel file above to get started.")
    st.markdown("""
**What DataAgent will do:**
- **Summarize** your data — shape, types, nulls, key stats
- **Visualize** automatically — bars, lines, scatter based on your data
- **Surface insights** — trends, anomalies, correlations via Gemini
- **Export** everything — data, summary, insights, stats as downloadable files
""")
