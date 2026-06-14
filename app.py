import os, sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import replace_impossible_zeros, cap_outliers, feature_engineering

st.set_page_config(page_title="DiabetesScan", page_icon="🩺",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container {
    background: #0D1117 !important;
    padding: 0 !important; margin: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
iframe[title="keyboard"],
[data-testid="InputInstructions"] { display: none !important; }

* { font-family: 'Inter', sans-serif; }

label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .68rem !important; font-weight: 500 !important;
    color: #5A7A90 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
}
input[type="number"] {
    background: #161D2A !important;
    border: 1px solid #242E3E !important;
    border-radius: 8px !important;
    color: #C8D8E8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .88rem !important;
}
input[type="number"]:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    outline: none !important;
}
.stSelectbox > div > div {
    background: #161D2A !important;
    border: 1px solid #242E3E !important;
    border-radius: 8px !important;
    color: #C8D8E8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .82rem !important;
}
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    border-radius: 8px !important;
    border: none !important;
    letter-spacing: .03em !important;
    transition: all .18s !important;
}
.stButton > button:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stBaseButton-primary"] {
    background: #2563EB !important;
    color: #fff !important;
}
[data-testid="stBaseButton-secondary"] {
    background: #161D2A !important;
    color: #6A8BA0 !important;
    border: 1px solid #242E3E !important;
}
hr { border-color: #161D2A !important; margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    d = "models"
    if not os.path.exists(d): return None, None, None
    def _load(*ns):
        for n in ns:
            p = os.path.join(d, n)
            if os.path.exists(p): return joblib.load(p)
        raise FileNotFoundError(ns)
    return (
        joblib.load(os.path.join(d, "scaler.pkl")),
        {"XGBoost":             _load("xgboost.pkl"),
         "Random Forest":       _load("random_forest.pkl"),
         "Regressão Logística": _load("regressao_logistica.pkl","regressao_log\u00edstica.pkl")},
        ["Pregnancies","Glucose","BloodPressure","SkinThickness",
         "Insulin","BMI","DiabetesPedigreeFunction","Age",
         "BMI_category","Age_group","Glucose_category"]
    )

scaler, models, feature_names = load_artifacts()
if "page"   not in st.session_state: st.session_state.page   = "home"
if "result" not in st.session_state: st.session_state.result = None


# ══════════════════════════════════════════════════════
# TELA 1 — HOME
# ══════════════════════════════════════════════════════
if st.session_state.page == "home":
    # Visuals via pure HTML
    components.html("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* { box-sizing:border-box; margin:0; padding:0; }
html, body { background:#0D1117; }
.wrap {
    display: flex; flex-direction: column;
    align-items: center;
    padding: 7vh 2rem 2.5rem;
    text-align: center;
}
.eye {
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem; letter-spacing: .28em;
    color: #60A5FA; text-transform: uppercase;
    margin-bottom: 2rem;
}
.title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 5vw, 4.5rem);
    font-weight: 900; color: #F0F6FF;
    line-height: 1.08; margin: 0 0 1.4rem;
}
.title span { color: #60A5FA; }
.sub {
    font-family: 'Inter', sans-serif;
    font-size: 1rem; color: #7A9BB0;
    line-height: 1.85; margin: 0 auto 2.5rem;
    max-width: 420px;
}
.stats {
    display: flex; width: 100%; max-width: 720px;
    border: 1px solid #1A2535; border-radius: 12px;
    overflow: hidden; margin-top: 1rem;
}
.stat {
    flex: 1; padding: 1.5rem 1rem;
    border-right: 1px solid #1A2535;
    background: #111827;
}
.stat:last-child { border-right: none; }
.stat-v {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem; font-weight: 700;
    color: #60A5FA; margin-bottom: .3rem;
}
.stat-l {
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem; color: #3A5A70;
    letter-spacing: .08em; text-transform: uppercase;
    line-height: 1.55;
}
</style>
</head><body>
<div class="wrap">
    <div class="eye">IA Aplicada à Saúde &nbsp;·&nbsp; Projeto A3 &nbsp;·&nbsp; USJT 2026</div>
    <div class="title">Predição de risco de<br><span>Diabetes Tipo 2</span></div>
    <p class="sub">Sistema de triagem com Machine Learning.<br>
    Insira os dados clínicos e receba uma análise<br>com explicabilidade SHAP em segundos.</p>
    <div class="stats">
        <div class="stat"><div class="stat-v">0.913</div><div class="stat-l">ROC-AUC<br>XGBoost</div></div>
        <div class="stat"><div class="stat-v">84.7%</div><div class="stat-l">Acurácia<br>no teste</div></div>
        <div class="stat"><div class="stat-v">3</div><div class="stat-l">Modelos<br>comparados</div></div>
        <div class="stat"><div class="stat-v">768</div><div class="stat-l">Pacientes<br>no dataset</div></div>
    </div>
</div>
</body></html>""", height=580)

    # Botão nativo Streamlit — único jeito confiável de navegar
    _, col_btn, _ = st.columns([2.5, 1.5, 2.5])
    with col_btn:
        if st.button("Iniciar Análise →", type="primary", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()


# ══════════════════════════════════════════════════════
# TELA 2 — FORMULÁRIO
# ══════════════════════════════════════════════════════
elif st.session_state.page == "form":

    # Header
    st.markdown("<div style='padding:1.25rem 6vw 0;'>", unsafe_allow_html=True)
    hc1, hc2 = st.columns([1, 10])
    with hc1:
        if st.button("← Voltar", type="secondary"):
            st.session_state.page = "home"
            st.rerun()
    with hc2:
        st.markdown("""<p style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
            letter-spacing:.2em;color:#1E3040;text-transform:uppercase;padding-top:.6rem;margin:0;">
            Formulário · DiabetesScan</p>""", unsafe_allow_html=True)
    st.markdown("<hr style='margin:1rem 0 0 !important;border-color:#161D2A!important;'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Formulário centralizado
    _, col_form, _ = st.columns([1, 2.5, 1])
    with col_form:
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
           letter-spacing:.2em;color:#2A4A6A;text-transform:uppercase;
           text-align:center;margin-bottom:1.5rem;">
           Dados clínicos do paciente
        </p>""", unsafe_allow_html=True)

        fa, fb = st.columns(2, gap="large")
        with fa:
            glucose = st.number_input("Glicemia mg/dL",    0,   300, 120, 1)
            age     = st.number_input("Idade anos",         18,  100, 33,  1)
            bp      = st.number_input("Pressão diastólica", 0,   200, 70,  1)
            insulin = st.number_input("Insulina μU/mL",     0,   900, 80,  1)
        with fb:
            bmi     = st.number_input("IMC kg/m²",          0.0, 70.0, 25.0, 0.1)
            preg    = st.number_input("Gestações",           0,   20,   1,    1)
            dpf     = st.number_input("Histórico familiar",  0.0, 3.0,  0.47, 0.01)
            skin    = st.number_input("Espessura pele mm",   0,   100,  20,   1)

        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
        model_choice = st.selectbox("Modelo de predição", list(models.keys()) if models else [])
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        if st.button("Analisar Risco →", type="primary", use_container_width=True):
            raw = pd.DataFrame([{
                "Pregnancies":preg,"Glucose":glucose,"BloodPressure":bp,
                "SkinThickness":skin,"Insulin":insulin,"BMI":bmi,
                "DiabetesPedigreeFunction":dpf,"Age":age,"Outcome":0}])
            inp = feature_engineering(cap_outliers(replace_impossible_zeros(raw)))
            X   = scaler.transform(inp.drop("Outcome", axis=1)[feature_names])
            m   = models[model_choice]
            try:
                import shap as shap_lib
                if hasattr(m, "feature_importances_"):
                    sv = shap_lib.TreeExplainer(m).shap_values(X)
                    # Random Forest retorna array 3D (n_samples, n_features, n_classes)
                    # ou lista [class0, class1] — pega sempre classe positiva
                    if isinstance(sv, list):
                        sv = sv[1]
                    elif hasattr(sv, "ndim") and sv.ndim == 3:
                        sv = sv[:, :, 1]
                else:
                    sv = shap_lib.LinearExplainer(m, X).shap_values(X)
                arr = sv[0] if hasattr(sv[0], "__len__") else sv
                shap_s = pd.Series(arr, index=feature_names).sort_values(key=abs, ascending=False)
            except Exception as e:
                shap_s = None
            st.session_state.result = {
                "pred": int(m.predict(X)[0]),
                "proba": float(m.predict_proba(X)[0][1]),
                "model_name": model_choice,
                "shap": shap_s,
                "inputs": {
                    "Glicemia": glucose, "IMC": bmi, "Idade": age,
                    "Pressão": bp, "Insulina": insulin, "Gestações": preg,
                    "Histórico": dpf, "Pele": skin
                }
            }
            st.session_state.page = "result"
            st.rerun()


# ══════════════════════════════════════════════════════
# TELA 3 — RESULTADO
# ══════════════════════════════════════════════════════
elif st.session_state.page == "result":
    r = st.session_state.result

    # Header
    st.markdown("<div style='padding:1.25rem 6vw 0;'>", unsafe_allow_html=True)
    hc1, hc2, hc3 = st.columns([1, 6, 2])
    with hc1:
        if st.button("← Voltar", type="secondary"):
            st.session_state.page = "form"
            st.rerun()
    with hc2:
        st.markdown("""<p style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
            letter-spacing:.2em;color:#1E3040;text-transform:uppercase;padding-top:.6rem;margin:0;">
            Resultado da Análise · DiabetesScan</p>""", unsafe_allow_html=True)
    with hc3:
        if st.button("Nova Análise →", type="primary", use_container_width=True):
            st.session_state.result = None
            st.session_state.page = "form"
            st.rerun()
    st.markdown("<hr style='margin:1rem 0 1.5rem !important;border-color:#161D2A!important;'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if r is None:
        st.warning("Nenhum resultado encontrado.")
        st.stop()

    hi   = r["pred"] == 1
    pct  = r["proba"] * 100
    pct_str = f"{pct:.1f}"

    if hi:
        verdict     = "&#9888;&nbsp; Alto Risco"
        gravidade   = "Indicadores compatíveis com diabetes tipo 2."
        orientacao  = "Recomendamos consulta médica e exames confirmatórios (glicemia de jejum, HbA1c)."
        cor_titulo  = "#F87171"
        cor_fill    = "#DC2626"
        cor_borda   = "rgba(248,113,113,0.18)"
        cor_bg      = "linear-gradient(150deg,#1A0808 0%,#111827 60%)"
        cor_label   = "#7F1D1D"
        cor_grav    = "#A05050"
        cor_ori     = "#7A4040"
        badge_bg    = "rgba(220,38,38,0.1)"
        badge_color = "#F87171"
        badge_txt   = "RISCO ELEVADO"
    else:
        verdict     = "&#10003;&nbsp; Baixo Risco"
        gravidade   = "Sem indicadores significativos de risco no momento."
        orientacao  = "Mantenha hábitos saudáveis e realize check-ups periódicos."
        cor_titulo  = "#4ADE80"
        cor_fill    = "#16A34A"
        cor_borda   = "rgba(74,222,128,0.12)"
        cor_bg      = "linear-gradient(150deg,#040F08 0%,#111827 60%)"
        cor_label   = "#14532D"
        cor_grav    = "#3A8A50"
        cor_ori     = "#2A6A40"
        badge_bg    = "rgba(22,163,74,0.1)"
        badge_color = "#4ADE80"
        badge_txt   = "RISCO BAIXO"

    # SHAP rows
    shap_rows = ""
    if r["shap"] is not None:
        top = r["shap"].head(8)
        mx  = max(abs(top.values)) or 1
        for f, v in top.items():
            w   = abs(v) / mx * 42
            pos = v > 0
            bar = (
                f'<div style="height:100%;background:#DC2626;border-radius:1px;position:absolute;left:50%;width:{w:.1f}%"></div>'
                if pos else
                f'<div style="height:100%;background:#16A34A;border-radius:1px;position:absolute;right:50%;width:{w:.1f}%"></div>'
            )
            vc = "#F87171" if pos else "#4ADE80"
            inf = "&#8593; aumenta risco" if pos else "&#8595; reduz risco"
            shap_rows += (
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;color:#4A6A80;width:160px;flex-shrink:0;text-align:right;">{f}</span>'
                f'<div style="flex:1;height:2px;background:#1A2535;border-radius:1px;position:relative;">{bar}</div>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.62rem;color:{vc};width:50px;text-align:right;">{v:+.3f}</span>'
                f'</div>'
            )

    # Inputs resumo
    inputs_html = ""
    if r.get("inputs"):
        for k, v in r["inputs"].items():
            inputs_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:.6rem 0;border-bottom:1px solid #161D2A;">'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;color:#6A8FA8;">{k}</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.65rem;color:#8AA8C0;font-weight:500;">{v}</span>'
                f'</div>'
            )

    shap_section = ""
    if shap_rows:
        shap_section = f"""
        <div style="margin-top:2rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.25rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                    letter-spacing:.18em;color:#2A4A5A;text-transform:uppercase;">
                    Fatores que influenciaram a predição
                </span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#1E3040;">
                    SHAP Values
                </span>
            </div>
            <div style="background:#0D1117;border-radius:10px;padding:1.25rem 1.5rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:1rem;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#1E3040;">Feature</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#1E3040;">vermelho &#8593; risco &nbsp;&#183;&nbsp; verde &#8595; risco</span>
                </div>
                {shap_rows}
            </div>
        </div>"""

    inputs_section = f"""
        <div style="margin-top:2rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                letter-spacing:.18em;color:#2A4A5A;text-transform:uppercase;margin-bottom:1rem;">
                Dados inseridos
            </div>
            <div style="background:#0D1117;border-radius:10px;padding:.5rem 1.25rem;">
                {inputs_html}
            </div>
        </div>"""

    html_result = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:transparent; font-family:'Inter',sans-serif; padding: 0 6vw 3rem; }}

.hero {{
    background: {cor_bg};
    border: 1px solid {cor_borda};
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 3rem;
}}
.hero-left {{ flex: 1; }}
.hero-right {{
    text-align: center;
    flex-shrink: 0;
    width: 200px;
}}
.badge {{
    display: inline-block;
    background: {badge_bg};
    color: {badge_color};
    font-family: 'JetBrains Mono', monospace;
    font-size: .6rem; letter-spacing: .2em;
    text-transform: uppercase;
    padding: .35rem .9rem;
    border-radius: 20px;
    border: 1px solid {cor_borda};
    margin-bottom: 1rem;
}}
.label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: .62rem; letter-spacing: .22em;
    text-transform: uppercase; color: {cor_label};
    margin-bottom: .75rem;
}}
.verdict {{
    font-family: 'Playfair Display', serif;
    font-size: 3rem; font-weight: 900;
    color: {cor_titulo}; line-height: 1.05;
    margin-bottom: .6rem;
}}
.gravidade {{
    font-family: 'Inter', sans-serif;
    font-size: .95rem; color: {cor_grav};
    font-weight: 500; margin-bottom: .4rem;
}}
.orientacao {{
    font-family: 'Inter', sans-serif;
    font-size: .82rem; color: {cor_ori};
    line-height: 1.65;
}}
.prob-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: .6rem; color: #5A7A90;
    letter-spacing: .1em; text-transform: uppercase;
    margin-bottom: .5rem;
}}
.prob-val {{
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 900;
    color: {cor_titulo}; line-height: 1;
    margin-bottom: .75rem;
}}
.track {{
    height: 4px; background: #1A2535;
    border-radius: 2px; overflow: hidden;
    margin-bottom: .35rem;
}}
.fill {{ height:100%; background:{cor_fill}; border-radius:2px; width:{pct_str}%; }}
.ticks {{
    display:flex; justify-content:space-between;
    margin-bottom: .5rem;
}}
.tick {{
    font-family:'JetBrains Mono',monospace;
    font-size:.52rem; color:#1A2535;
}}
.model-tag {{
    font-family:'JetBrains Mono',monospace;
    font-size:.58rem; color:#1A2A38;
}}

.grid {{
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 1.5rem;
}}
.panel {{
    background: #111827;
    border: 1px solid #161D2A;
    border-radius: 12px;
    padding: 1.75rem;
}}
.panel-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: .62rem; letter-spacing: .18em;
    text-transform: uppercase; color: #2A4A5A;
    margin-bottom: 1.25rem;
}}

.disc {{
    background: rgba(161,130,18,.055);
    border: 1px solid rgba(161,130,18,.12);
    border-radius: 10px; padding: 1rem 1.25rem;
    display: flex; gap: .75rem; margin-top: 1.5rem;
    align-items: flex-start;
}}
.disc-t {{
    font-family: 'Inter', sans-serif;
    font-size: .75rem; color: #5A5020; line-height: 1.7;
}}
.disc-t strong {{ color: #7A6828; }}
</style>
</head><body>

<div class="hero">
    <div class="hero-left">
        <div class="label">Resultado da análise · {r["model_name"]}</div>
        <div class="verdict">{verdict}</div>
        <div class="gravidade">{gravidade}</div>
        <div class="orientacao">{orientacao}</div>
    </div>
    <div class="hero-right">
        <div class="badge">{badge_txt}</div>
        <div class="prob-label">Probabilidade</div>
        <div class="prob-val">{pct_str}%</div>
        <div class="track"><div class="fill"></div></div>
        <div class="ticks">
            <span class="tick">0%</span>
            <span class="tick">50%</span>
            <span class="tick">100%</span>
        </div>
        <div class="model-tag">Modelo · {r["model_name"]}</div>
    </div>
</div>

<div class="grid">
    <div class="panel">
        <div class="panel-title">Fatores que influenciaram a predição · SHAP Values</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:#1E3040;
            margin-bottom:1.1rem;">vermelho &#8593; aumenta risco &nbsp;&#183;&nbsp; verde &#8595; reduz risco</div>
        {shap_rows if shap_rows else '<p style="color:#1E3040;font-size:.75rem;">SHAP indisponível.</p>'}
    </div>
    <div class="panel">
        <div class="panel-title">Dados inseridos</div>
        {inputs_html}
    </div>
</div>

<div class="disc">
    <span style="flex-shrink:0;font-size:.8rem;color:#5A4A18;margin-top:.1rem;">&#9888;</span>
    <div class="disc-t">
        <strong>Uso responsável &mdash;</strong>
        Ferramenta de triagem cl&iacute;nica, n&atilde;o diagn&oacute;stico m&eacute;dico.
        Resultados devem ser confirmados com glicemia de jejum e HbA1c.
        Dados processados localmente &middot; <strong>Conformidade LGPD Art. 11</strong>
    </div>
</div>

</body></html>"""

    components.html(html_result, height=820, scrolling=True)
