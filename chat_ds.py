"""
Chat Conversacional - Ciencia de Datos
Maestría en Ciencia de Datos | EAFIT 2026-1
3 respuestas en paralelo con diferentes temperaturas
"""

import streamlit as st
import time
import json
import re
from groq import Groq

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataSci Chat | EAFIT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# ESTILOS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace !important; }

.stApp { background: #080c12; color: #dce3ee; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1a2233;
}

.block-container { padding-top: 1.8rem; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a2240 60%, #071a33 100%);
    border: 1px solid #1a3a5c;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "🔬";
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 4rem;
    opacity: 0.12;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    color: #38bdf8;
    margin: 0 0 0.3rem;
}
.hero-sub { color: #64748b; font-size: 0.88rem; margin: 0; }

/* Temp cards */
.temp-header {
    border-radius: 10px 10px 0 0;
    padding: 0.7rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}
.temp-body {
    border-radius: 0 0 10px 10px;
    padding: 1rem;
    font-size: 0.88rem;
    line-height: 1.65;
    min-height: 160px;
}

/* T=0.2 — azul frío */
.t-low .temp-header  { background: #0c2340; border: 1px solid #1e4976; color: #38bdf8; border-bottom: none; }
.t-low .temp-body    { background: #0a1e35; border: 1px solid #1e4976; border-top: none; color: #bae6fd; }

/* T=0.8 — verde */
.t-mid .temp-header  { background: #0c2e1e; border: 1px solid #1a5c38; color: #34d399; border-bottom: none; }
.t-mid .temp-body    { background: #091f15; border: 1px solid #1a5c38; border-top: none; color: #a7f3d0; }

/* T=1.5 — naranja */
.t-high .temp-header { background: #2e1a08; border: 1px solid #7c3a0a; color: #fb923c; border-bottom: none; }
.t-high .temp-body   { background: #1e1008; border: 1px solid #7c3a0a; border-top: none; color: #fed7aa; }

/* Metric pill */
.mpill {
    display: inline-block;
    background: #0d1a2a;
    border: 1px solid #1a3a5c;
    color: #38bdf8;
    border-radius: 20px;
    padding: 3px 11px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    margin: 2px;
}

/* Chat message */
.chat-user {
    background: #0d1a2a;
    border: 1px solid #1a3a5c;
    border-radius: 10px 10px 10px 2px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
.chat-bot {
    background: #091520;
    border: 1px solid #0e2a40;
    border-radius: 10px 10px 2px 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #bae6fd;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: #64748b;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: #0c2340 !important;
    color: #38bdf8 !important;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: #0d1117 !important;
    border: 1px solid #1a2233 !important;
    color: #dce3ee !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0c4a6e) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(3,105,161,0.4) !important;
}

/* Info box */
.info-box {
    background: rgba(3,105,161,0.1);
    border-left: 3px solid #0369a1;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #7dd3fc;
    margin: 0.6rem 0;
}

/* Metrics table */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}
.metric-card {
    background: #0d1a2a;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
}

/* Judge */
.judge-box {
    background: #0c1f0e;
    border: 1px solid #1a5c22;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SISTEMA
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """Eres DataSci GPT, un asistente especializado en Ciencia de Datos. 
Tienes dominio experto en: estadística, machine learning, deep learning, 
visualización de datos, Python (pandas, sklearn, matplotlib, seaborn), 
SQL, feature engineering, y despliegue de modelos.

Respondes siempre en español, con claridad pedagógica. Usas ejemplos concretos,
fragmentos de código cuando aplica, y explicas el razonamiento detrás de cada concepto.
Eres preciso, directo y académicamente riguroso."""

TEMPERATURES = [
    {"val": 0.2, "label": "T = 0.2 — Preciso",    "icon": "🧊", "cls": "t-low"},
    {"val": 0.8, "label": "T = 0.8 — Balanceado",  "icon": "⚖️", "cls": "t-mid"},
    {"val": 1.5, "label": "T = 1.5 — Creativo",    "icon": "🔥", "cls": "t-high"},
]

MODEL = "llama-3.3-70b-versatile"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔑 API Key")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros de Entrada")

    max_tokens = st.slider("Max tokens", 100, 1000, 400, 50)
    top_p      = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

    st.markdown("""
    <div style='margin-top:1rem;font-size:0.72rem;color:#334155;font-family:IBM Plex Mono,monospace;line-height:1.7'>
    🔬 DataSci Chat<br>
    Modelo: llama-3.3-70b<br>
    EAFIT 2026-1
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Parámetros activos")
    st.markdown(f"""
    <span class='mpill'>top_p = {top_p}</span>
    <span class='mpill'>max_tokens = {max_tokens}</span>
    <span class='mpill'>T ∈ {{0.2, 0.8, 1.5}}</span>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">DataSci Chat</div>
  <p class="hero-sub">Chat conversacional especializado en Ciencia de Datos · Llama 3.3 70B · Groq API</p>
  <div style="margin-top:0.7rem">
    <span class="mpill">🧊 T=0.2 Preciso</span>
    <span class="mpill">⚖️ T=0.8 Balanceado</span>
    <span class="mpill">🔥 T=1.5 Creativo</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []       # [{role, content}]
if "metrics" not in st.session_state:
    st.session_state.metrics = []       # [{pregunta, t02, t08, t15, ...}]
if "responses" not in st.session_state:
    st.session_state.responses = None   # última triple respuesta

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "💬 Chat Comparativo",
    "📊 Métricas de Desempeño",
    "🧪 Parámetros I/O",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <div class="info-box">
    Cada pregunta genera <b>3 respuestas en paralelo</b> con diferentes temperaturas.
    Observa cómo cambia la precisión vs creatividad del modelo.
    </div>
    """, unsafe_allow_html=True)

    # Sugerencias rápidas
    st.markdown("**💡 Preguntas sugeridas:**")
    sugerencias = [
        "¿Qué es el overfitting y cómo se corrige?",
        "Explica la diferencia entre Random Forest y XGBoost",
        "¿Cuándo usar PCA para reducción de dimensionalidad?",
        "¿Qué métricas usar para clasificación desbalanceada?",
    ]
    cols_sug = st.columns(4)
    selected_sug = None
    for i, sug in enumerate(sugerencias):
        with cols_sug[i]:
            if st.button(sug[:35] + "...", key=f"sug_{i}"):
                selected_sug = sug

    # Historial de chat (solo el último intercambio visible)
    if st.session_state.history:
        last_user = next((m for m in reversed(st.session_state.history) if m["role"] == "user"), None)
        if last_user:
            st.markdown(f"""
            <div class="chat-user">
            <b style="color:#38bdf8">🧑 Tú</b><br>{last_user['content']}
            </div>
            """, unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Pregunta sobre Ciencia de Datos...")
    question = selected_sug or user_input

    if question and api_key:
        client = Groq(api_key=api_key)

        # Construir mensajes con historial
        messages_base = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages_base += [m for m in st.session_state.history if m["role"] in ("user", "assistant")]
        messages_base.append({"role": "user", "content": question})

        st.session_state.history.append({"role": "user", "content": question})

        results = []

        with st.spinner("Generando 3 respuestas en paralelo..."):
            for t_cfg in TEMPERATURES:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages_base,
                    temperature=t_cfg["val"],
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                latency = round(time.time() - t0, 2)
                usage   = resp.usage
                tps     = round(usage.completion_tokens / latency, 1) if latency > 0 else 0
                answer  = resp.choices[0].message.content

                results.append({
                    "temp":     t_cfg["val"],
                    "label":    t_cfg["label"],
                    "icon":     t_cfg["icon"],
                    "cls":      t_cfg["cls"],
                    "answer":   answer,
                    "latency":  latency,
                    "tps":      tps,
                    "tokens_in":  usage.prompt_tokens,
                    "tokens_out": usage.completion_tokens,
                    "tokens_total": usage.total_tokens,
                })

            # LLM-as-a-Judge sobre T=0.8 (la balanceada)
            judge_prompt = (
                f"Evalúa esta respuesta sobre Ciencia de Datos del 1 al 10 "
                f"(veracidad, claridad, utilidad).\n\n"
                f"Pregunta: {question}\n\nRespuesta: {results[1]['answer']}\n\n"
                f"Devuelve SOLO JSON sin texto extra: "
                f'{{\"score\": <1-10>, \"feedback\": \"<una línea>\"}}'
            )
            try:
                judge_resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.1,
                    max_tokens=150,
                )
                jtext = judge_resp.choices[0].message.content.strip()
                jmatch = re.search(r'\{[^}]+\}', jtext, re.DOTALL)
                jdata = json.loads(jmatch.group()) if jmatch else {}
                judge_score    = jdata.get("score", "—")
                judge_feedback = jdata.get("feedback", "Sin evaluación")
            except Exception:
                judge_score, judge_feedback = "—", "Error en evaluación"

        st.session_state.responses = results
        st.session_state.history.append({
            "role": "assistant",
            "content": results[1]["answer"]   # guardamos T=0.8 como "canónica"
        })
        st.session_state.metrics.append({
            "Pregunta":         question[:45] + "...",
            "Lat T=0.2 (s)":   results[0]["latency"],
            "Lat T=0.8 (s)":   results[1]["latency"],
            "Lat T=1.5 (s)":   results[2]["latency"],
            "TPS T=0.2":        results[0]["tps"],
            "TPS T=0.8":        results[1]["tps"],
            "TPS T=1.5":        results[2]["tps"],
            "Tokens salida":    results[1]["tokens_out"],
            "Score Judge":      judge_score,
        })

        st.rerun()

    elif question and not api_key:
        st.warning("🔑 Ingresa tu Groq API Key en la barra lateral.")

    # Mostrar última triple respuesta
    if st.session_state.responses:
        results = st.session_state.responses
        col1, col2, col3 = st.columns(3)

        for col, r in zip([col1, col2, col3], results):
            with col:
                st.markdown(f"""
                <div class="{r['cls']}">
                  <div class="temp-header">{r['icon']} {r['label']}</div>
                  <div class="temp-body">{r['answer']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"<span class='mpill'>⏱ {r['latency']}s</span>"
                    f"<span class='mpill'>⚡ {r['tps']} TPS</span>"
                    f"<span class='mpill'>🔢 {r['tokens_out']} tokens</span>",
                    unsafe_allow_html=True,
                )

        # Judge score
        if st.session_state.metrics:
            last = st.session_state.metrics[-1]
            st.markdown(f"""
            <div class="judge-box">
            <b style="color:#4ade80">🧑‍⚖️ LLM-as-a-Judge</b> (evaluando T=0.8)<br>
            <span style="font-family:IBM Plex Mono,monospace;font-size:1.6rem;color:#4ade80">{last['Score Judge']}/10</span>
            &nbsp;&nbsp;<span style="color:#86efac;font-size:0.85rem">{judge_feedback if 'judge_feedback' in dir() else ''}</span>
            </div>
            """, unsafe_allow_html=True)

    if st.button("🗑️ Nueva conversación"):
        st.session_state.history   = []
        st.session_state.responses = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Métricas de Desempeño del LLM")

    if not st.session_state.metrics:
        st.info("Aún no hay datos. Realiza al menos una pregunta en el Chat Comparativo.")
    else:
        import pandas as pd
        import plotly.graph_objects as go

        df = pd.DataFrame(st.session_state.metrics)

        # KPIs del último turno
        last = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱ Lat promedio (s)",
                  round((last["Lat T=0.2 (s)"] + last["Lat T=0.8 (s)"] + last["Lat T=1.5 (s)"]) / 3, 2))
        c2.metric("⚡ TPS promedio",
                  round((last["TPS T=0.2"] + last["TPS T=0.8"] + last["TPS T=1.5"]) / 3, 1))
        c3.metric("🔢 Tokens salida (T=0.8)", int(last["Tokens salida"]))
        c4.metric("🧑‍⚖️ Score Judge", f"{last['Score Judge']}/10")

        st.markdown("---")

        # Tabla completa
        st.markdown("### 📋 Historial de métricas")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(df) > 1:
            st.markdown("---")
            # Latencia comparativa
            st.markdown("### ⏱ Latencia por temperatura")
            fig_lat = go.Figure()
            x = list(range(1, len(df)+1))
            for col, color, name in [
                ("Lat T=0.2 (s)", "#38bdf8", "T=0.2"),
                ("Lat T=0.8 (s)", "#34d399", "T=0.8"),
                ("Lat T=1.5 (s)", "#fb923c", "T=1.5"),
            ]:
                fig_lat.add_trace(go.Scatter(
                    x=x, y=df[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=7),
                ))
            fig_lat.update_layout(
                paper_bgcolor="#080c12", plot_bgcolor="#0d1117",
                font=dict(color="#dce3ee", family="Inter"),
                legend=dict(bgcolor="#0d1117"),
                xaxis_title="Turno", yaxis_title="Segundos",
                height=280, margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

            # TPS comparativo
            st.markdown("### ⚡ Tokens por segundo (TPS)")
            fig_tps = go.Figure()
            for col, color, name in [
                ("TPS T=0.2", "#38bdf8", "T=0.2"),
                ("TPS T=0.8", "#34d399", "T=0.8"),
                ("TPS T=1.5", "#fb923c", "T=1.5"),
            ]:
                fig_tps.add_trace(go.Bar(
                    x=x, y=df[col], name=name,
                    marker_color=color, opacity=0.8,
                ))
            fig_tps.update_layout(
                barmode="group",
                paper_bgcolor="#080c12", plot_bgcolor="#0d1117",
                font=dict(color="#dce3ee", family="Inter"),
                legend=dict(bgcolor="#0d1117"),
                xaxis_title="Turno", yaxis_title="TPS",
                height=280, margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig_tps, use_container_width=True)

            # Score Judge
            valid = [(i+1, s) for i, s in enumerate(df["Score Judge"]) if s != "—"]
            if len(valid) > 1:
                xi, yi = zip(*valid)
                fig_sc = go.Figure(go.Scatter(
                    x=list(xi), y=list(yi),
                    mode="lines+markers+text",
                    text=[str(v) for v in yi],
                    textposition="top center",
                    line=dict(color="#4ade80", width=2),
                    marker=dict(size=9, color="#4ade80"),
                    fill="tozeroy",
                    fillcolor="rgba(74,222,128,0.08)",
                ))
                fig_sc.update_layout(
                    title="🧑‍⚖️ Evolución del Score LLM-as-a-Judge",
                    yaxis=dict(range=[0, 11]),
                    paper_bgcolor="#080c12", plot_bgcolor="#0d1117",
                    font=dict(color="#dce3ee", family="Inter"),
                    title_font=dict(color="#4ade80"),
                    height=260, margin=dict(t=40, b=30),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PARÁMETROS I/O
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🧪 Parámetros de Entrada / Salida")

    st.markdown("""
    <div class="info-box">
    Aquí se explica cada parámetro que controla el comportamiento del LLM,
    su efecto en la respuesta y los valores recomendados.
    </div>
    """, unsafe_allow_html=True)

    import pandas as pd

    params_data = {
        "Parámetro": [
            "temperature",
            "top_p",
            "max_tokens",
            "model",
            "stream",
        ],
        "Tipo": ["float", "float", "int", "string", "bool"],
        "Rango": ["0.0 – 2.0", "0.0 – 1.0", "1 – 32768", "—", "True/False"],
        "Valor usado": [
            "0.2 / 0.8 / 1.5",
            str(top_p),
            str(max_tokens),
            MODEL,
            "False",
        ],
        "Efecto": [
            "Controla aleatoriedad. Bajo=preciso, Alto=creativo",
            "Nucleus sampling: limita el pool de tokens candidatos",
            "Máximo de tokens en la respuesta generada",
            "Arquitectura y capacidades del modelo a usar",
            "Si True, envía tokens conforme se generan (streaming)",
        ],
    }

    df_params = pd.DataFrame(params_data)
    st.markdown("### 📥 Parámetros de Entrada (Input)")
    st.dataframe(df_params, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📤 Parámetros de Salida (Output / Usage)")

    output_data = {
        "Campo": [
            "choices[0].message.content",
            "usage.prompt_tokens",
            "usage.completion_tokens",
            "usage.total_tokens",
            "choices[0].finish_reason",
        ],
        "Descripción": [
            "Texto generado por el modelo",
            "Tokens consumidos en el prompt (entrada)",
            "Tokens generados en la respuesta (salida)",
            "Total = prompt + completion tokens",
            "'stop' = terminó natural, 'length' = cortado por max_tokens",
        ],
    }
    st.dataframe(pd.DataFrame(output_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🌡️ Efecto visual de la Temperatura")

    import plotly.graph_objects as go
    import numpy as np

    tokens_demo = ["modelo", "algoritmo", "dato", "función", "variable", "resultado", "patrón", "árbol"]
    logits = np.array([3.5, 2.8, 2.1, 1.7, 1.3, 1.0, 0.6, 0.3])

    def softmax_t(logits, t):
        t = max(t, 1e-6)
        e = np.exp(logits / t - np.max(logits / t))
        return e / e.sum()

    fig_cmp = go.Figure()
    for t_val, color, name in [
        (0.2, "#38bdf8", "T=0.2 (preciso)"),
        (0.8, "#34d399", "T=0.8 (balance)"),
        (1.5, "#fb923c", "T=1.5 (creativo)"),
    ]:
        fig_cmp.add_trace(go.Scatter(
            x=tokens_demo,
            y=np.round(softmax_t(logits, t_val), 3),
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=7),
        ))

    fig_cmp.update_layout(
        xaxis_title="Token candidato",
        yaxis_title="Probabilidad",
        paper_bgcolor="#080c12",
        plot_bgcolor="#0d1117",
        font=dict(color="#dce3ee", family="Inter"),
        legend=dict(bgcolor="#0d1117"),
        height=320,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    Con <b>T=0.2</b> la distribución es muy concentrada → casi siempre elige el token más probable.<br>
    Con <b>T=1.5</b> la distribución se aplana → hay más variedad pero también más riesgo de incoherencia.
    </div>
    """, unsafe_allow_html=True)
