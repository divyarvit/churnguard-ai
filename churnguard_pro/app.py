import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from pipeline.trainer import load_pipeline, train_pipeline, predict_single

st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f8f7ff;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label { color: #a5b4fc !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: .07em; }
section[data-testid="stSidebar"] .stRadio div[role="radio"] p { font-size: 0.82rem !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 0.8rem !important; color: #c7d2fe !important; }

/* ── LOGO ── */
.logo {
    font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    padding: 6px 0 2px;
}
.badge-live {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(52,211,153,0.18); color: #34d399 !important;
    border: 1px solid rgba(52,211,153,0.35);
    padding: 3px 10px; border-radius: 20px;
    font-size: .68rem; font-family: 'DM Mono', monospace;
    margin-bottom: 14px; letter-spacing: .05em;
}

/* ── PAGE BACKGROUND ── */
.main .block-container { background: transparent; }

/* ── SECTION HEADERS ── */
.sec {
    font-size: 1.1rem; font-weight: 700; color: #1e1b4b;
    margin: 22px 0 14px;
    padding: 10px 16px;
    background: linear-gradient(90deg, #ede9fe 0%, #f8f7ff 100%);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
}

/* ── KPI CARDS ── */
.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 22px 16px;
    border: 1px solid #e8e4ff;
    position: relative; overflow: hidden;
    box-shadow: 0 2px 12px rgba(99,102,241,0.07);
    margin-bottom: 4px;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.kpi-card.purple::before { background: linear-gradient(90deg,#6366f1,#8b5cf6); }
.kpi-card.red::before    { background: linear-gradient(90deg,#f43f5e,#fb7185); }
.kpi-card.green::before  { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.amber::before  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-lbl {
    font-size: .68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .09em; color: #7c3aed; margin-bottom: 6px;
}
.kpi-val {
    font-size: 2rem; font-weight: 800; color: #1e1b4b;
    line-height: 1.1; letter-spacing: -.02em;
}
.kpi-sub { font-size: .72rem; color: #6b7280; margin-top: 6px; }
.kpi-icon {
    position: absolute; right: 18px; top: 18px;
    font-size: 1.8rem; opacity: 0.13;
}

/* ── CHART SECTION CARD ── */
.chart-card {
    background: #ffffff; border-radius: 16px;
    padding: 20px 20px 8px;
    border: 1px solid #e8e4ff;
    box-shadow: 0 2px 12px rgba(99,102,241,0.06);
    margin-bottom: 16px;
}
.chart-title {
    font-size: .8rem; font-weight: 700; color: #4338ca;
    text-transform: uppercase; letter-spacing: .07em; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}

/* ── BANNER ── */
.hero-banner {
    background: linear-gradient(120deg, #4338ca 0%, #7c3aed 50%, #ec4899 100%);
    border-radius: 18px; padding: 24px 28px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 8px 30px rgba(99,102,241,0.22);
}
.hero-title { font-size: 1.3rem; font-weight: 800; color: #fff; margin-bottom: 4px; }
.hero-sub   { font-size: .82rem; color: rgba(255,255,255,0.75); }
.hero-stat  { text-align: right; }
.hero-stat-val { font-size: 2rem; font-weight: 800; color: #fff; }
.hero-stat-lbl { font-size: .72rem; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: .07em; }

/* ── RISK BADGES ── */
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:.7rem; font-weight:600; }
.badge-critical { background:#fce7f3; color:#be185d; }
.badge-high     { background:#fef3c7; color:#b45309; }
.badge-atrisk   { background:#ede9fe; color:#6d28d9; }
.badge-safe     { background:#d1fae5; color:#065f46; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    letter-spacing: .02em !important; padding: 0.5rem 1.4rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}

/* ── METRICS ── */
div[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important; color: #1e1b4b !important;
}
div[data-testid="stMetricLabel"] { font-weight: 600 !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #f5f3ff !important; border-radius: 10px !important;
    font-weight: 600 !important; color: #4338ca !important;
    border: 1px solid #e8e4ff !important;
}

/* ── DATAFRAME ── */
.stDataFrame { border-radius: 12px !important; border: 1px solid #e8e4ff !important; }

/* ── SELECTBOX / INPUTS ── */
.stSelectbox > div > div { border-radius: 10px !important; border-color: #c7d2fe !important; }

/* ── PLOTLY CHART CARDS ── */
.element-container:has(.stPlotlyChart) {
    background: #fff; border-radius: 16px;
    border: 1px solid #e8e4ff;
    padding: 12px;
    box-shadow: 0 2px 10px rgba(99,102,241,0.06);
}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_pkg(ds):
    p = load_pipeline(ds)
    if p is None:
        with st.spinner(f"Training {ds} model on real data… (~60s)"):
            p = train_pipeline(ds)
    return p

with st.sidebar:
    st.markdown('<div class="logo">⚡ ChurnGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge-live">● LIVE MODEL</div>', unsafe_allow_html=True)
    dataset = st.selectbox("Dataset", ["Telco (Telecom)", "Bank (BFSI)"])
    ds_key  = "telco" if "Telco" in dataset else "bank"
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview", "👥 Customer Table", "🎯 Predict Churn",
        "🤖 Model Performance", "📈 EDA & Insights",
        "🛡️ Retention Playbooks", "⚙️ Retrain"
    ])
    st.markdown("---")
    pkg = get_pkg(ds_key)
    st.markdown(f"**Records:** {pkg['n_records']:,}")
    st.markdown(f"**Best Model:** {pkg['best_model_name']}")
    st.markdown(f"**AUC-ROC:** {pkg['model_results'][pkg['best_model_name']]['roc_auc']:.4f}")
    st.markdown(f"**Churn Rate:** {pkg['churn_rate']:.2%}")
    st.markdown(f"**Status:** <span style='color:#10b981'>● Running</span>", unsafe_allow_html=True)

df     = pkg['scored_df']
res    = pkg['model_results']
bname  = pkg['best_model_name']
bres   = res[bname]
fi     = pkg['feature_importance']

import plotly.express as px
import plotly.graph_objects as go

DARK = dict(template='plotly_white', plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            font_color='#374151', margin=dict(l=0,r=0,t=30,b=0))
GRID = dict(gridcolor='#f0eeff', zerolinecolor='#e0d9ff')

def dkfig(fig, h=300, legend=False):
    fig.update_layout(**DARK, height=h, showlegend=legend,
                      xaxis=GRID, yaxis=GRID)
    return fig

# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="sec">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    total     = len(df)
    churned   = int(df['PredictedChurn'].sum())
    high_risk = int(df['RiskLevel'].isin(['High Risk','Critical']).sum())
    churn_pct = churned / total * 100

    # Hero banner
    st.markdown(f"""
    <div class="hero-banner">
      <div>
        <div class="hero-title">⚡ ChurnGuard AI — Live Intelligence</div>
        <div class="hero-sub">Real-time churn prediction powered by Random Forest · {total:,} customers scored</div>
        <div style="margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;">
          <span style="background:rgba(255,255,255,0.18);color:#fff;padding:4px 12px;border-radius:20px;font-size:.72rem;font-weight:600;">📊 Telco Dataset</span>
          <span style="background:rgba(255,255,255,0.18);color:#fff;padding:4px 12px;border-radius:20px;font-size:.72rem;font-weight:600;">🤖 AUC-ROC {bres['roc_auc']:.4f}</span>
          <span style="background:rgba(255,255,255,0.18);color:#fff;padding:4px 12px;border-radius:20px;font-size:.72rem;font-weight:600;">🔥 {churn_pct:.1f}% Churn Rate</span>
        </div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">{high_risk:,}</div>
        <div class="hero-stat-lbl">High-risk customers</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔥 Predicted Churn",   f"{churn_pct:.1f}%",   delta=f"{churned} customers")
    c2.metric("⚠️ High Risk",         f"{high_risk:,}",      delta="need intervention", delta_color="inverse")
    c3.metric("📋 Total Customers",   f"{total:,}")
    c4.metric("🏆 Best Model AUC",    f"{bres['roc_auc']:.4f}")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(df, x='ChurnProbability', nbins=40, color_discrete_sequence=['#6366f1'],
                           labels={'ChurnProbability':'Churn Probability (%)'})
        for xv,col,lbl in [(30,'#10b981','Safe/At-Risk'),(60,'#f59e0b','High Risk'),(80,'#ef4444','Critical')]:
            fig.add_vline(x=xv, line_dash="dash", line_color=col, annotation_text=lbl, annotation_font_color=col)
        st.plotly_chart(dkfig(fig,280), use_container_width=True)

    with col2:
        st.subheader("Risk Segments")
        seg = df['RiskLevel'].value_counts()
        fig2 = go.Figure(go.Pie(labels=seg.index, values=seg.values, hole=0.62,
                                marker_colors=['#10b981','#6366f1','#f59e0b','#f43f5e']))
        fig2.update_layout(**DARK, height=280, showlegend=True,
                           legend=dict(orientation='v', font=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Churn Drivers (Feature Importance)")
    top = fi.head(10)
    fig3 = px.bar(top, orientation='h', color=top.values,
                  color_continuous_scale='Purples',
                  labels={'value':'Importance','index':'Feature'})
    fig3.update_layout(**DARK, height=340, coloraxis_showscale=False)
    fig3.update_xaxes(**GRID); fig3.update_yaxes(**GRID)
    st.plotly_chart(fig3, use_container_width=True)

    if ds_key == 'telco' and 'Contract' in df.columns:
        col3,col4 = st.columns(2)
        with col3:
            st.subheader("Churn by Contract Type")
            ct = df.groupby('Contract')['Churn'].mean().sort_values()*100
            fig4 = px.bar(ct, color=ct.values, color_continuous_scale='RdYlGn_r',
                          labels={'value':'Churn Rate (%)','Contract':'Contract'},text=ct.round(1))
            fig4.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig4,280), use_container_width=True)
        with col4:
            st.subheader("Churn by Internet Service")
            inet = df.groupby('InternetService')['Churn'].mean().sort_values()*100
            fig5 = px.bar(inet, color=inet.values, color_continuous_scale='RdYlGn_r',
                          labels={'value':'Churn Rate (%)'},text=inet.round(1))
            fig5.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig5,280), use_container_width=True)

    if ds_key == 'bank' and 'Geography' in df.columns:
        col3,col4 = st.columns(2)
        with col3:
            st.subheader("Churn by Country")
            geo = df.groupby('Geography')['Churn'].mean().sort_values()*100
            fig4 = px.bar(geo, color=geo.values, color_continuous_scale='RdYlGn_r',
                          labels={'value':'Churn Rate (%)'},text=geo.round(1))
            fig4.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig4,280), use_container_width=True)
        with col4:
            st.subheader("Churn by Gender")
            gen = df.groupby('Gender')['Churn'].mean().sort_values()*100
            fig5 = px.bar(gen, color=gen.values, color_continuous_scale='RdYlGn_r',
                          labels={'value':'Churn Rate (%)'},text=gen.round(1))
            fig5.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig5,280), use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER TABLE
# ══════════════════════════════════════════════════════
elif page == "👥 Customer Table":
    st.markdown('<div class="sec">👥 Customer Risk Intelligence</div>', unsafe_allow_html=True)

    # ── Filters row ──
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    risk_f   = c1.selectbox("Risk Level", ["All","Critical","High Risk","At-Risk","Safe"])
    if ds_key == 'telco':
        seg_opts = ["All"] + sorted(df['Contract'].dropna().unique().tolist()) if 'Contract' in df.columns else ["All"]
        seg_f    = c2.selectbox("Contract", seg_opts)
        seg_col  = 'Contract'
    else:
        seg_opts = ["All"] + sorted(df['Geography'].dropna().unique().tolist()) if 'Geography' in df.columns else ["All"]
        seg_f    = c2.selectbox("Geography", seg_opts)
        seg_col  = 'Geography'
    sort_f   = c3.selectbox("Sort by", ["Risk % ↓","Risk % ↑"])
    search_q = c4.text_input("🔍 Search customer", placeholder="Type any value…")

    disp = df.copy()
    if risk_f  != "All": disp = disp[disp['RiskLevel']==risk_f]
    if seg_f   != "All": disp = disp[disp[seg_col]==seg_f]
    if search_q.strip():
        mask = disp.astype(str).apply(lambda col: col.str.contains(search_q.strip(), case=False)).any(axis=1)
        disp = disp[mask]
    disp = disp.sort_values('ChurnProbability', ascending=(sort_f=="Risk % ↑"))

    if ds_key == 'telco':
        show = ['ChurnProbability','RiskLevel','tenure','MonthlyCharges','Contract',
                'InternetService','PaymentMethod','Churn']
        show = [c for c in show if c in disp.columns]
    else:
        show = ['ChurnProbability','RiskLevel','CreditScore','Age','Balance',
                'NumOfProducts','IsActiveMember','Geography','Churn']
        show = [c for c in show if c in disp.columns]

    col_info, col_dl = st.columns([3,1])
    col_info.markdown(f"**Showing {len(disp):,} customers**")
    col_dl.download_button(
        label="⬇️ Export CSV",
        data=disp[show].to_csv(index=False).encode('utf-8'),
        file_name=f"churnguard_{ds_key}_risk_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    def color_risk(val):
        m = {'Critical':'background-color:#fce7f3;color:#be185d',
             'High Risk':'background-color:#fef3c7;color:#b45309',
             'At-Risk':'background-color:#ede9fe;color:#6d28d9',
             'Safe':'background-color:#d1fae5;color:#065f46'}
        return m.get(str(val),'')

    def color_prob(val):
        if val>=80: return 'color:#f43f5e;font-weight:700'
        if val>=60: return 'color:#f59e0b;font-weight:700'
        if val>=30: return 'color:#6366f1'
        return 'color:#10b981'

    styled = (disp[show].head(500)
        .style.applymap(color_risk, subset=['RiskLevel'])
        .applymap(color_prob, subset=['ChurnProbability'])
        .format({'ChurnProbability':'{:.1f}%','MonthlyCharges':'${:.2f}',
                 'Balance':'${:,.0f}','CreditScore':'{:.0f}'}))
    st.dataframe(styled, use_container_width=True, height=520)

# ══════════════════════════════════════════════════════
# PAGE 3 — PREDICT
# ══════════════════════════════════════════════════════
elif page == "🎯 Predict Churn":
    st.markdown('<div class="sec">🎯 Real-Time Churn Prediction</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1,1])

    with col_form:
        st.markdown("**Enter Customer Details**")
        inp = {}
        if ds_key == 'telco':
            inp['tenure']         = st.slider("Tenure (months)", 0, 72, 12)
            inp['MonthlyCharges'] = st.number_input("Monthly Charges ($)", 18.0, 119.0, 65.0, step=1.0)
            inp['TotalCharges']   = inp['tenure'] * inp['MonthlyCharges']
            inp['SeniorCitizen']  = int(st.selectbox("Senior Citizen", ["No","Yes"]) == "Yes")
            inp['Contract']       = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            inp['InternetService']= st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
            inp['PaymentMethod']  = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            inp['Partner']        = st.selectbox("Has Partner", ["Yes","No"])
            inp['Dependents']     = st.selectbox("Has Dependents", ["Yes","No"])
            inp['TechSupport']    = st.selectbox("Tech Support", ["Yes","No","No internet service"])
            inp['OnlineSecurity'] = st.selectbox("Online Security", ["Yes","No","No internet service"])
            # Engineered features
            inp['AvgMonthlyCharge'] = inp['TotalCharges'] / max(inp['tenure'],1)
            inp['ChargeRatio']      = inp['MonthlyCharges'] / (inp['AvgMonthlyCharge']+1)
            svc_map = {'Yes':1,'No':0,'No internet service':0,'No phone service':0}
            inp['ServiceCount']  = sum(svc_map.get(inp.get(k,'No'),0) for k in ['TechSupport','OnlineSecurity'])
            inp['ContractRisk']  = {'Month-to-month':2,'One year':1,'Two year':0}[inp['Contract']]
            inp['HighValue']     = int(inp['MonthlyCharges'] > 64.76)
            inp['AutoPay']       = int('automatic' in inp['PaymentMethod'])
            for col in ['PhoneService','MultipleLines','OnlineBackup','DeviceProtection','StreamingTV','StreamingMovies','PaperlessBilling']:
                inp[col] = 'No'
        else:
            inp['CreditScore']     = st.slider("Credit Score", 300, 900, 650)
            inp['Age']             = st.slider("Age", 18, 92, 38)
            inp['Tenure']          = st.slider("Tenure (years)", 0, 10, 5)
            inp['Balance']         = st.number_input("Account Balance ($)", 0.0, 260000.0, 75000.0, step=1000.0)
            inp['NumOfProducts']   = st.selectbox("Number of Products", [1,2,3,4])
            inp['HasCrCard']       = int(st.selectbox("Has Credit Card", ["Yes","No"]) == "Yes")
            inp['IsActiveMember']  = int(st.selectbox("Is Active Member", ["Yes","No"]) == "Yes")
            inp['EstimatedSalary'] = st.number_input("Estimated Salary ($)", 11.0, 200000.0, 100000.0, step=1000.0)
            inp['Geography']       = st.selectbox("Country", ["France","Germany","Spain"])
            inp['Gender']          = st.selectbox("Gender", ["Male","Female"])
            inp['BalancePerProduct']  = inp['Balance'] / (inp['NumOfProducts']+1)
            inp['SalaryBalanceRatio'] = inp['Balance'] / max(inp['EstimatedSalary'],1)
            inp['ZeroBalance']        = int(inp['Balance'] == 0)
            inp['EngagementScore']    = inp['IsActiveMember']*3 + inp['HasCrCard'] + inp['NumOfProducts']*2
            inp['GoodCredit']         = int(inp['CreditScore'] >= 700)
            inp['HighValue']          = int(inp['Balance'] > 75000)

        run = st.button("🚀 Run Prediction", use_container_width=True)

    with col_result:
        if run:
            prob, pred = predict_single(inp, ds_key)
            if prob is None:
                st.error("Model not loaded.")
            else:
                if   prob>=80: level,color,emoji,act = "Critical",  "#ef4444","🔴","Escalate immediately — personal call within 24h + custom retention offer (up to 30% discount)."
                elif prob>=60: level,color,emoji,act = "High Risk",  "#f59e0b","🟡","Proactive outreach this week — CSM check-in call + free training session."
                elif prob>=30: level,color,emoji,act = "At-Risk",   "#818cf8","🟣","Re-engagement campaign — personalized email + 30-day premium feature trial."
                else:          level,color,emoji,act = "Safe",       "#10b981","🟢","Healthy. Continue nurture. Look for upsell opportunity."

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    number={'suffix':'%','font':{'size':44,'color':color,'family':'Syne'}},
                    gauge={'axis':{'range':[0,100],'tickcolor':'#64748b'},
                           'bar':{'color':color,'thickness':0.25},
                           'bgcolor':'#f5f3ff',
                           'steps':[{'range':[0,30],'color':'rgba(16,185,129,.1)'},
                                    {'range':[30,60],'color':'rgba(99,102,241,.1)'},
                                    {'range':[60,80],'color':'rgba(245,158,11,.1)'},
                                    {'range':[80,100],'color':'rgba(244,63,94,.1)'}],
                           'threshold':{'line':{'color':color,'width':3},'thickness':0.8,'value':prob}},
                    title={'text':f'{emoji} {level}','font':{'size':16,'color':color,'family':'Syne'}}
                ))
                fig_g.update_layout(paper_bgcolor='#ffffff', font_color='#374151',
                                    margin=dict(l=10,r=10,t=50,b=10), height=300)
                st.plotly_chart(fig_g, use_container_width=True)
                st.markdown(f"**Churn Probability:** <span style='color:{color};font-size:1.3rem;font-weight:700'>{prob}%</span>", unsafe_allow_html=True)
                st.info(f"**Recommended Action:** {act}")

                # ── SHAP explanation ──
                st.markdown("#### 🔍 Why this score? (Top risk drivers)")
                try:
                    import shap
                    row_dict = {c: 0 for c in pkg['num_feats'] + pkg['cat_feats']}
                    for k, v in inp.items():
                        if k in row_dict: row_dict[k] = v
                    X_row = pd.DataFrame([row_dict])
                    for c in pkg['cat_feats']: X_row[c] = X_row[c].astype(str)
                    X_row_pp = np.nan_to_num(pkg['preprocessor'].transform(X_row), nan=0.0)
                    model = pkg['best_model']
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X_row_pp)
                    sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                    feat_names = pkg.get('feature_names', [f"f_{i}" for i in range(len(sv))])
                    shap_df = pd.DataFrame({'Feature': feat_names[:len(sv)], 'SHAP': sv})\
                                .reindex(pd.Series(sv).abs().sort_values(ascending=False).index)\
                                .head(8).reset_index(drop=True)
                    shap_df = shap_df.sort_values('SHAP')
                    colors_shap = ['#f43f5e' if v > 0 else '#10b981' for v in shap_df['SHAP']]
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df['SHAP'], y=shap_df['Feature'],
                        orientation='h', marker_color=colors_shap,
                        text=[f"+{v:.3f}" if v>0 else f"{v:.3f}" for v in shap_df['SHAP']],
                        textposition='outside'
                    ))
                    fig_shap.update_layout(**DARK, height=300,
                        xaxis=dict(title='Impact on churn score', **GRID),
                        title=dict(text='🔴 Red = increases churn risk  |  🟢 Green = reduces risk', font=dict(size=12)))
                    st.plotly_chart(fig_shap, use_container_width=True)
                except ImportError:
                    st.warning("Install shap for explainability: `pip install shap`")
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")
        else:
            st.markdown("""
            <div style='background:#f5f3ff;border:2px dashed #c7d2fe;border-radius:16px;padding:48px;text-align:center;color:#6d28d9;margin-top:20px'>
                <div style='font-size:2.5rem;margin-bottom:12px'>🎯</div>
                <div style='font-size:1rem;font-weight:600;margin-bottom:6px;color:#1e1b4b;'>Ready to predict</div>
                <div style='font-size:.82rem;color:#7c3aed;'>Fill the form and click<br><strong>Run Prediction</strong></div>
            </div>""", unsafe_allow_html=True)

    # ── What-If Simulator ──
    st.markdown("---")
    st.markdown('<div class="sec">🔬 What-If Simulator — See how changes reduce risk</div>', unsafe_allow_html=True)
    st.caption("Adjust the sliders below to see how changing a customer attribute affects their churn probability.")

    if ds_key == 'telco':
        wi_col1, wi_col2, wi_col3 = st.columns(3)
        wi_tenure   = wi_col1.slider("Tenure (months)", 0, 72, 12, key="wi_tenure")
        wi_charges  = wi_col2.slider("Monthly Charges ($)", 18, 119, 70, key="wi_charges")
        wi_contract = wi_col3.selectbox("Contract Type", ["Month-to-month","One year","Two year"], key="wi_contract")
        wi_inp = {
            'tenure': wi_tenure, 'MonthlyCharges': wi_charges,
            'TotalCharges': wi_tenure * wi_charges,
            'SeniorCitizen': 0, 'Contract': wi_contract,
            'InternetService': 'Fiber optic', 'PaymentMethod': 'Electronic check',
            'Partner': 'No', 'Dependents': 'No', 'TechSupport': 'No', 'OnlineSecurity': 'No',
            'AvgMonthlyCharge': (wi_tenure * wi_charges) / max(wi_tenure, 1),
            'ChargeRatio': wi_charges / (((wi_tenure * wi_charges) / max(wi_tenure, 1)) + 1),
            'ServiceCount': 0, 'ContractRisk': {'Month-to-month':2,'One year':1,'Two year':0}[wi_contract],
            'HighValue': int(wi_charges > 64.76), 'AutoPay': 0,
            'PhoneService':'No','MultipleLines':'No','OnlineBackup':'No',
            'DeviceProtection':'No','StreamingTV':'No','StreamingMovies':'No','PaperlessBilling':'No'
        }
    else:
        wi_col1, wi_col2, wi_col3 = st.columns(3)
        wi_credit  = wi_col1.slider("Credit Score", 300, 900, 600, key="wi_credit")
        wi_balance = wi_col2.slider("Balance ($)", 0, 250000, 80000, step=1000, key="wi_balance")
        wi_active  = wi_col3.selectbox("Active Member", ["Yes","No"], key="wi_active")
        wi_inp = {
            'CreditScore': wi_credit, 'Age': 38, 'Tenure': 5,
            'Balance': wi_balance, 'NumOfProducts': 2,
            'HasCrCard': 1, 'IsActiveMember': int(wi_active=="Yes"),
            'EstimatedSalary': 100000, 'Geography': 'France', 'Gender': 'Male',
            'BalancePerProduct': wi_balance / 3,
            'SalaryBalanceRatio': wi_balance / 100000,
            'ZeroBalance': int(wi_balance == 0),
            'EngagementScore': int(wi_active=="Yes")*3 + 1 + 4,
            'GoodCredit': int(wi_credit >= 700),
            'HighValue': int(wi_balance > 75000)
        }

    wi_prob, _ = predict_single(wi_inp, ds_key)
    if wi_prob is not None:
        if   wi_prob >= 80: wi_color, wi_label = "#f43f5e", "🔴 Critical"
        elif wi_prob >= 60: wi_color, wi_label = "#f59e0b", "🟡 High Risk"
        elif wi_prob >= 30: wi_color, wi_label = "#6366f1", "🟣 At-Risk"
        else:               wi_color, wi_label = "#10b981", "🟢 Safe"
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #e8e4ff;border-radius:14px;padding:18px 22px;display:flex;align-items:center;gap:20px;'>
          <div style='font-size:2rem;font-weight:800;color:{wi_color};'>{wi_prob}%</div>
          <div>
            <div style='font-size:1rem;font-weight:700;color:#1e1b4b;'>{wi_label}</div>
            <div style='font-size:.78rem;color:#6b7280;'>Predicted churn probability with these settings</div>
          </div>
          <div style='margin-left:auto;'>
            <div style='height:10px;width:200px;background:#f0eeff;border-radius:5px;overflow:hidden;'>
              <div style='height:100%;width:{wi_prob}%;background:{wi_color};border-radius:5px;transition:width .4s;'></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="sec">🤖 ML Model Performance Report</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("✅ Accuracy",   f"{bres['accuracy']*100:.1f}%")
    c2.metric("📈 AUC-ROC",   f"{bres['roc_auc']:.4f}")
    c3.metric("🎯 Precision",  f"{bres['precision']*100:.1f}%")
    c4.metric("🔍 Recall",     f"{bres['recall']*100:.1f}%")
    c5.metric("⚖️ F1 Score",  f"{bres['f1']:.4f}")

    st.markdown(f"**Best Model:** {bname} &nbsp;|&nbsp; **CV AUC:** {bres['cv_auc_mean']:.4f} ± {bres['cv_auc_std']:.4f} &nbsp;|&nbsp; **Train AUC:** {bres['train_auc_mean']:.4f}")

    # Model comparison table
    st.subheader("Model Comparison")
    comp = pd.DataFrame([{
        'Model': n,
        'Accuracy': f"{v['accuracy']*100:.1f}%",
        'AUC-ROC':  f"{v['roc_auc']:.4f}",
        'Precision':f"{v['precision']*100:.1f}%",
        'Recall':   f"{v['recall']*100:.1f}%",
        'F1':       f"{v['f1']:.4f}",
        'CV AUC':   f"{v['cv_auc_mean']:.4f}±{v['cv_auc_std']:.4f}",
    } for n,v in res.items()])
    st.dataframe(comp.set_index('Model'), use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves — All Models")
        fig_roc = go.Figure()
        colors  = ['#6366f1','#10b981','#f59e0b','#ef4444']
        for i,(name,r) in enumerate(res.items()):
            fig_roc.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines',
                line=dict(color=colors[i%len(colors)], width=2.5 if name==bname else 1.5,
                          dash='solid' if name==bname else 'dot'),
                name=f"{name} ({r['roc_auc']:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',
            line=dict(color='#475569',dash='dash'),name='Random',showlegend=True))
        fig_roc.update_layout(**DARK, height=340, showlegend=True,
            xaxis=dict(title='False Positive Rate',**GRID),
            yaxis=dict(title='True Positive Rate',**GRID),
            legend=dict(font=dict(size=11)))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader(f"Confusion Matrix — {bname}")
        cm = np.array(bres['confusion_matrix'])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=['Pred: No Churn','Pred: Churn'], y=['Actual: No Churn','Actual: Churn'],
            colorscale=[[0,'#f5f3ff'],[0.5,'#a5b4fc'],[1,'#4338ca']],
            text=cm, texttemplate='<b>%{text}</b>', showscale=False))
        fig_cm.update_layout(**DARK, height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.subheader("Precision-Recall Curve")
        fig_pr = go.Figure()
        for i,(name,r) in enumerate(res.items()):
            fig_pr.add_trace(go.Scatter(x=r['recall_curve'], y=r['precision_curve'], mode='lines',
                line=dict(color=colors[i%len(colors)], width=2 if name==bname else 1.5),
                name=f"{name} (AP={r['avg_precision']:.3f})"))
        fig_pr.update_layout(**DARK, height=300, showlegend=True,
            xaxis=dict(title='Recall',**GRID), yaxis=dict(title='Precision',**GRID),
            legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_pr, use_container_width=True)

    with col4:
        st.subheader("Cross-Validation AUC (5-Fold)")
        cv_df = pd.DataFrame({'Model':list(res.keys()),
                              'CV AUC Mean':[v['cv_auc_mean'] for v in res.values()],
                              'CV AUC Std': [v['cv_auc_std']  for v in res.values()]})
        fig_cv = go.Figure()
        for i,row in cv_df.iterrows():
            fig_cv.add_trace(go.Bar(name=row['Model'], x=[row['Model']],
                y=[row['CV AUC Mean']], error_y=dict(type='data',array=[row['CV AUC Std']]),
                marker_color=colors[i%len(colors)]))
        fig_cv.update_layout(**DARK, height=300, showlegend=False,
            yaxis=dict(range=[0.5,1.0],**GRID), xaxis=GRID)
        st.plotly_chart(fig_cv, use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 5 — EDA & INSIGHTS
# ══════════════════════════════════════════════════════
elif page == "📈 EDA & Insights":
    st.markdown('<div class="sec">📈 Exploratory Data Analysis & Insights</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    if ds_key == 'telco':
        with col1:
            st.subheader("Monthly Charges: Churned vs Retained")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['Churn']==0]['MonthlyCharges'], name='Retained', marker_color='#10b981', opacity=0.75, nbinsx=30))
            fig.add_trace(go.Histogram(x=df[df['Churn']==1]['MonthlyCharges'], name='Churned',  marker_color='#f43f5e', opacity=0.75, nbinsx=30))
            fig.update_layout(**DARK, height=300, barmode='overlay', showlegend=True,
                              xaxis=dict(title='Monthly Charges ($)',**GRID), yaxis=GRID)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Tenure: Churned vs Retained")
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=df[df['Churn']==0]['tenure'], name='Retained', marker_color='#10b981', opacity=0.75, nbinsx=30))
            fig2.add_trace(go.Histogram(x=df[df['Churn']==1]['tenure'], name='Churned',  marker_color='#f43f5e', opacity=0.75, nbinsx=30))
            fig2.update_layout(**DARK, height=300, barmode='overlay', showlegend=True,
                               xaxis=dict(title='Tenure (months)',**GRID), yaxis=GRID)
            st.plotly_chart(fig2, use_container_width=True)

        col3,col4 = st.columns(2)
        with col3:
            st.subheader("Churn by Payment Method")
            pm = df.groupby('PaymentMethod')['Churn'].mean().sort_values()*100
            fig3 = px.bar(pm, orientation='h', color=pm.values, color_continuous_scale='RdYlGn_r',
                          text=pm.round(1), labels={'value':'Churn Rate (%)','index':'Payment Method'})
            fig3.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig3,300), use_container_width=True)

        with col4:
            st.subheader("Tenure Bucket vs Churn Rate")
            tb = df.groupby('TenureBucket')['Churn'].mean().sort_values()*100
            fig4 = px.bar(tb, color=tb.values, color_continuous_scale='RdYlGn_r',
                          text=tb.round(1), labels={'value':'Churn Rate (%)','TenureBucket':'Tenure'})
            fig4.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig4,300), use_container_width=True)

    else:
        with col1:
            st.subheader("Age Distribution: Churned vs Retained")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['Churn']==0]['Age'], name='Retained', marker_color='#10b981', opacity=0.75, nbinsx=30))
            fig.add_trace(go.Histogram(x=df[df['Churn']==1]['Age'], name='Churned',  marker_color='#f43f5e', opacity=0.75, nbinsx=30))
            fig.update_layout(**DARK, height=300, barmode='overlay', showlegend=True,
                              xaxis=dict(title='Age',**GRID), yaxis=GRID)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Balance: Churned vs Retained")
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=df[df['Churn']==0]['Balance'], name='Retained', marker_color='#10b981', opacity=0.75, nbinsx=30))
            fig2.add_trace(go.Histogram(x=df[df['Churn']==1]['Balance'], name='Churned',  marker_color='#f43f5e', opacity=0.75, nbinsx=30))
            fig2.update_layout(**DARK, height=300, barmode='overlay', showlegend=True,
                               xaxis=dict(title='Balance ($)',**GRID), yaxis=GRID)
            st.plotly_chart(fig2, use_container_width=True)

        col3,col4 = st.columns(2)
        with col3:
            st.subheader("Churn by Number of Products")
            np_ = df.groupby('NumOfProducts')['Churn'].mean().sort_index()*100
            fig3 = px.bar(np_, color=np_.values, color_continuous_scale='RdYlGn_r',
                          text=np_.round(1), labels={'value':'Churn Rate (%)','NumOfProducts':'# Products'})
            fig3.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig3,300), use_container_width=True)

        with col4:
            st.subheader("Churn by Age Bucket")
            ab = df.groupby('AgeBucket')['Churn'].mean().sort_values()*100
            fig4 = px.bar(ab, color=ab.values, color_continuous_scale='RdYlGn_r',
                          text=ab.round(1), labels={'value':'Churn Rate (%)','AgeBucket':'Age Group'})
            fig4.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(dkfig(fig4,300), use_container_width=True)

    st.subheader("Correlation: Churn Probability vs Key Features")
    num_cols = [c for c in pkg['num_feats'] if c in df.columns][:8]
    corr_data = df[num_cols + ['ChurnProbability']].select_dtypes(include=[np.number]).copy()
    if 'ChurnProbability' in corr_data.columns and len(corr_data.columns) > 1:
        corr = corr_data.corr()[['ChurnProbability']].drop('ChurnProbability').sort_values('ChurnProbability')
        fig5 = px.bar(corr, orientation='h',
                      color=corr['ChurnProbability'],
                      color_continuous_scale='RdBu_r',
                      labels={'ChurnProbability':'Correlation with Churn','index':'Feature'})
        fig5.update_layout(**DARK, height=320, coloraxis_showscale=True)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Not enough numeric features to compute correlation.")

# ══════════════════════════════════════════════════════
# PAGE 6 — RETENTION PLAYBOOKS
# ══════════════════════════════════════════════════════
elif page == "🛡️ Retention Playbooks":
    st.markdown('<div class="sec">🛡️ Retention Strategy & Playbooks</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    critical_n = int((df['RiskLevel']=='Critical').sum())
    high_n     = int((df['RiskLevel']=='High Risk').sum())
    c1.metric("🔴 Critical",    f"{critical_n}",  "need immediate action")
    c2.metric("🟡 High Risk",   f"{high_n}",       "need outreach this week")
    c3.metric("📊 Model AUC",   f"{bres['roc_auc']:.4f}", "prediction confidence")
    c4.metric("✅ Recall",      f"{bres['recall']*100:.1f}%","churners caught")

    playbooks = [
        ("🔴 Critical — 80–100%","#ef4444","rgba(239,68,68,.08)",
         "Immediate human escalation required.",
         ["Account Manager personal call within 24 hours",
          "Custom retention offer: up to 30% discount or free upgrade",
          "Executive sponsor email for Enterprise customers",
          "Dedicated CSM assignment","Emergency product review session"],
         "34%","High","Escalate now"),
        ("🟡 High Risk — 60–80%","#f59e0b","rgba(245,158,11,.08)",
         "Proactive outreach before they decide to leave.",
         ["Automated 3-email re-engagement sequence",
          "Customer Success Manager check-in call this week",
          "Free feature training / onboarding session",
          "Plan right-sizing conversation","Personalised success story delivery"],
         "48%","Medium","This week"),
        ("🟣 At-Risk — 30–60%","#818cf8","rgba(129,140,248,.08)",
         "Early intervention while customer is still engaged.",
         ["Personalised re-engagement email campaign",
          "30-day premium feature free trial",
          "In-app tips, tooltips and usage nudges",
          "NPS survey + follow-up call if low score",
          "Relevant case study and ROI report"],
         "62%","Low","Within 2 weeks"),
        ("🟢 Safe — 0–30%","#10b981","rgba(16,185,129,.08)",
         "Healthy customers — focus on growth and advocacy.",
         ["Standard quarterly nurture email cadence",
          "Quarterly Business Review invitation",
          "Upsell / cross-sell opportunity identification",
          "Referral and loyalty program invitation",
          "Early access to new features"],
         "N/A","Minimal","Ongoing"),
    ]

    for title, color, bg, desc, actions, success, cost, timing in playbooks:
        with st.expander(title, expanded='Critical' in title or 'High' in title):
            ca,cb = st.columns([2,1])
            with ca:
                st.markdown(f"**Strategy:** {desc}")
                st.markdown("**Actions:**")
                for a in actions: st.markdown(f"  - {a}")
            with cb:
                st.metric("Success Rate", success)
                st.metric("Intervention Cost", cost)
                st.metric("Timing", timing)

    st.subheader("Risk Distribution — Action Priority")
    seg = df['RiskLevel'].value_counts().reindex(['Critical','High Risk','At-Risk','Safe']).fillna(0)
    fig = px.bar(seg, color=seg.index,
                 color_discrete_map={'Critical':'#ef4444','High Risk':'#f59e0b','At-Risk':'#818cf8','Safe':'#10b981'},
                 text=seg.values, labels={'value':'Customers','index':'Risk Level'})
    fig.update_traces(textposition='outside')
    fig.update_layout(**DARK, height=300, showlegend=False, xaxis=GRID, yaxis=GRID)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 7 — RETRAIN
# ══════════════════════════════════════════════════════
elif page == "⚙️ Retrain":
    st.markdown('<div class="sec">⚙️ Model Retraining</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        st.info(f"""
**Current Model — {ds_key.upper()}**
- Dataset: Real {ds_key} data
- Records: {pkg['n_records']:,}
- Best Model: {bname}
- AUC-ROC: {bres['roc_auc']:.4f}
- F1 Score: {bres['f1']:.4f}
- Churn Rate: {pkg['churn_rate']:.2%}
- Train size: {pkg['train_size']:,}
- Test size: {pkg['test_size']:,}
        """)
    with col2:
        st.info("""
**Pipeline Details**
- Preprocessing: StandardScaler + OneHotEncoder
- Imbalance: Upsampling (minority class)
- Validation: 5-Fold Stratified CV
- Models: RF, HistGBM, Logistic Reg
- Feature Eng: RFM, engagement, risk scores
        """)

    if st.button(f"🔄 Retrain on {ds_key.upper()} Data Now"):
        get_pkg.clear()
        with st.spinner("Retraining all models… (~60 seconds)"):
            new_pkg = train_pipeline(ds_key)
        st.success("✅ Models retrained successfully!")
        st.balloons()
        for name, r in new_pkg['model_results'].items():
            st.markdown(f"**{name}:** Acc={r['accuracy']:.3f} | AUC={r['roc_auc']:.3f} | F1={r['f1']:.3f} | CV={r['cv_auc_mean']:.3f}±{r['cv_auc_std']:.3f}")

    st.markdown("---")
    st.subheader("Upload Your Own CSV")
    st.markdown("Upload a CSV with the same columns as the dataset to score your own customers.")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        try:
            udf = pd.read_csv(uploaded)
            st.success(f"Loaded {len(udf):,} rows × {udf.shape[1]} columns")
            st.dataframe(udf.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
