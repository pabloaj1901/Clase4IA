"""
Taller NLP & LLMs - EAFIT 2026-1
Maestría en Ciencia de Datos
Docente: Jorge Iván Padilla-Buriticá
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ─── Groq (opcional si no está instalado) ──────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─── NLTK ──────────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NLP & LLMs Taller | EAFIT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# ESTILOS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

section[data-testid="stSidebar"] {
    background: #13161f;
    border-right: 1px solid #1e2230;
}

.block-container { padding-top: 2rem; }

/* Cards */
.nlp-card {
    background: #161923;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.metric-pill {
    display: inline-block;
    background: #0a1628;
    border: 1px solid #1a3a5c;
    color: #4fc3f7;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    margin: 3px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13161f;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #8892a4;
    border-radius: 8px;
    padding: 6px 14px;
}

.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #4fc3f7 !important;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: #13161f !important;
    border: 1px solid #1e2a3a !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Sliders */
.stSlider > div > div > div { background: #1e3a5f !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(21,101,192,0.4) !important;
}

/* Dataframes */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d2137 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(79,195,247,0.12) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #4fc3f7;
    margin: 0 0 0.3rem;
}

.hero-sub {
    color: #8892a4;
    font-size: 0.9rem;
    margin: 0;
}

/* Tag / badge */
.tag {
    display: inline-block;
    background: rgba(79,195,247,0.1);
    border: 1px solid rgba(79,195,247,0.3);
    color: #4fc3f7;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin-right: 4px;
}

/* Judge score */
.judge-score {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    color: #4fc3f7;
    text-align: center;
}

/* Warning / info boxes */
.info-box {
    background: rgba(21,101,192,0.12);
    border-left: 3px solid #1565c0;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.87rem;
    color: #90caf9;
    margin: 0.6rem 0;
}

/* Token chips */
.token-chip {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔑 Configuración API")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Obtén tu key en console.groq.com",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Modelo LLM")
    model_choice = st.selectbox(
        "Modelo",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.72rem; color:#4a5568; line-height:1.6'>
    🎓 EAFIT 2026-1<br>
    Maestría en Ciencia de Datos<br>
    NLP & LLMs Workshop
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🧠 Taller NLP & LLMs</div>
  <p class="hero-sub">Maestría en Ciencia de Datos · EAFIT 2026-1 · Jorge Iván Padilla-Buriticá</p>
  <div style="margin-top:0.8rem">
    <span class="tag">BoW</span><span class="tag">TF-IDF</span>
    <span class="tag">N-grams</span><span class="tag">Tokenización</span>
    <span class="tag">Llama 3.3</span><span class="tag">LLM-as-a-Judge</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔤 Tokenización & Encoding",
    "📊 Vectorización Clásica",
    "🔗 N-grams & Secuencias",
    "🌡️ Laboratorio LLM",
    "🤖 Agente Conversacional",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOKENIZACIÓN & ENCODING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Tokenización y Encoding")
    st.markdown("""
    <div class="info-box">
    Compara cómo el mismo texto es fragmentado por diferentes estrategias:
    <b>Word-level</b> (separación por espacios/puntuación) vs <b>BPE / Subword</b> (byte-pair encoding como usa Llama).
    </div>
    """, unsafe_allow_html=True)

    default_text = "El procesamiento del lenguaje natural permite que las máquinas entiendan el texto humano."
    user_text = st.text_area("✏️ Ingresa tu texto:", value=default_text, height=90)

    if st.button("🔍 Analizar Tokenización"):
        col1, col2 = st.columns(2)

        # Word-level
        with col1:
            st.markdown("### Word-Level Tokenizer")
            word_tokens = re.findall(r"\w+|[^\w\s]", user_text)
            chips = " ".join([
                f'<span class="token-chip" style="background:rgba(21,101,192,0.25);border:1px solid #1565c0;color:#90caf9">{t}</span>'
                for t in word_tokens
            ])
            st.markdown(chips, unsafe_allow_html=True)
            st.markdown(f"<br><span class='metric-pill'>🔢 {len(word_tokens)} tokens</span>", unsafe_allow_html=True)

            # One-hot mini demo
            st.markdown("#### One-Hot Encoding (muestra 5 tokens)")
            vocab = sorted(set(word_tokens))
            demo_tokens = word_tokens[:5]
            oh_data = {}
            for t in demo_tokens:
                row = [1 if v == t else 0 for v in vocab[:10]]
                oh_data[t] = row
            oh_df = pd.DataFrame(oh_data, index=vocab[:10]).T
            st.dataframe(oh_df.style.highlight_max(axis=1, color="#1e3a5f"), use_container_width=True)

        # BPE / Subword simulation
        with col2:
            st.markdown("### BPE / Subword Tokenizer (Llama-style)")
            words = user_text.split()
            bpe_tokens = []
            colors = ["rgba(0,150,136,0.25)", "rgba(156,39,176,0.25)", "rgba(230,81,0,0.25)"]
            border_colors = ["#00897b", "#8e24aa", "#e65100"]
            for i, word in enumerate(words):
                ci = i % 3
                if len(word) > 6:
                    mid = len(word) // 2
                    parts = [word[:mid] + "▁", word[mid:]]
                    for p in parts:
                        bpe_tokens.append((p, ci))
                else:
                    bpe_tokens.append((word, ci))

            chips2 = " ".join([
                f'<span class="token-chip" style="background:{colors[c]};border:1px solid {border_colors[c]};color:#e8eaf0">{t}</span>'
                for t, c in bpe_tokens
            ])
            st.markdown(chips2, unsafe_allow_html=True)
            st.markdown(f"<br><span class='metric-pill'>🔢 {len(bpe_tokens)} tokens</span>", unsafe_allow_html=True)

            st.markdown("#### Comparativa")
            comp = pd.DataFrame({
                "Método": ["Word-Level", "BPE / Subword"],
                "# Tokens": [len(word_tokens), len(bpe_tokens)],
                "Vocabulario cerrado": ["Sí", "No"],
                "Maneja OOV": ["No", "Sí"],
                "Semántica densa": ["Baja", "Alta"],
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)

        # Encoding vs Embeddings explanation
        st.markdown("---")
        st.markdown("### One-Hot vs Embeddings Densos")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="nlp-card">
            <b style="color:#4fc3f7">🔲 One-Hot Encoding</b><br><br>
            • Dimensionalidad = |Vocabulario| (puede ser >50k)<br>
            • Vectores <b>dispersos</b>: un 1, el resto 0s<br>
            • <b>No captura semántica</b>: "rey" y "reina" son ortogonales<br>
            • Sin relación entre palabras similares<br>
            • Eficiente para vocabularios pequeños
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="nlp-card">
            <b style="color:#4fc3f7">🌐 Embeddings Densos</b><br><br>
            • Dimensionalidad fija (128, 256, 768...)<br>
            • Vectores <b>densos</b>: todos los valores son reales<br>
            • <b>Captura semántica</b>: "rey" − "hombre" + "mujer" ≈ "reina"<br>
            • Similitud coseno mide cercanía conceptual<br>
            • Aprendidos durante entrenamiento (Word2Vec, GloVe, BERT)
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VECTORIZACIÓN CLÁSICA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Vectorización Clásica: BoW y TF-IDF")
    st.markdown("""
    <div class="info-box">
    Ingresa un <b>corpus</b> (un documento por línea). Se generará la Bolsa de Palabras (BoW)
    y los valores TF-IDF para comparar ambas representaciones.
    </div>
    """, unsafe_allow_html=True)

    default_corpus = """El aprendizaje automático transforma los datos en conocimiento.
Los modelos de lenguaje procesan texto de forma eficiente.
El procesamiento del lenguaje natural es un campo fascinante.
Los transformers revolucionaron el aprendizaje profundo en NLP.
BERT y GPT son modelos basados en la arquitectura transformer."""

    corpus_input = st.text_area("📄 Corpus (un documento por línea):", value=default_corpus, height=150)

    max_feat = st.slider("Máximo de características (tokens):", 5, 30, 15)

    if st.button("📈 Generar Matrices"):
        docs = [d.strip() for d in corpus_input.strip().split("\n") if d.strip()]

        if len(docs) < 2:
            st.warning("Ingresa al menos 2 documentos.")
        else:
            # BoW
            bow_vec = CountVectorizer(max_features=max_feat)
            bow_matrix = bow_vec.fit_transform(docs).toarray()
            bow_df = pd.DataFrame(bow_matrix, columns=bow_vec.get_feature_names_out(),
                                  index=[f"Doc {i+1}" for i in range(len(docs))])

            # TF-IDF
            tfidf_vec = TfidfVectorizer(max_features=max_feat)
            tfidf_matrix = tfidf_vec.fit_transform(docs).toarray()
            tfidf_df = pd.DataFrame(
                np.round(tfidf_matrix, 3),
                columns=tfidf_vec.get_feature_names_out(),
                index=[f"Doc {i+1}" for i in range(len(docs))],
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🧮 Bolsa de Palabras (BoW)")
                st.markdown("*Frecuencia cruda de cada término*")
                fig_bow = px.imshow(
                    bow_df,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    title="Matriz BoW",
                )
                fig_bow.update_layout(
                    paper_bgcolor="#13161f",
                    plot_bgcolor="#13161f",
                    font=dict(color="#e8eaf0", family="DM Sans"),
                    title_font=dict(color="#4fc3f7"),
                )
                st.plotly_chart(fig_bow, use_container_width=True)
                st.dataframe(bow_df, use_container_width=True)

            with col2:
                st.markdown("### 📐 TF-IDF")
                st.markdown("*Frecuencia ponderada por rareza del término*")
                fig_tf = px.imshow(
                    tfidf_df,
                    color_continuous_scale="Teal",
                    aspect="auto",
                    title="Matriz TF-IDF",
                )
                fig_tf.update_layout(
                    paper_bgcolor="#13161f",
                    plot_bgcolor="#13161f",
                    font=dict(color="#e8eaf0", family="DM Sans"),
                    title_font=dict(color="#4fc3f7"),
                )
                st.plotly_chart(fig_tf, use_container_width=True)
                st.dataframe(tfidf_df, use_container_width=True)

            # Top palabras TF-IDF
            st.markdown("---")
            st.markdown("### 🏆 Top palabras por TF-IDF (promedio entre documentos)")
            mean_tfidf = tfidf_df.mean().sort_values(ascending=False)
            fig_bar = go.Figure(go.Bar(
                x=mean_tfidf.index,
                y=mean_tfidf.values,
                marker=dict(
                    color=mean_tfidf.values,
                    colorscale="Teal",
                    showscale=True,
                ),
            ))
            fig_bar.update_layout(
                xaxis_title="Término",
                yaxis_title="TF-IDF Promedio",
                paper_bgcolor="#13161f",
                plot_bgcolor="#161923",
                font=dict(color="#e8eaf0", family="DM Sans"),
                xaxis=dict(tickangle=-35),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — N-GRAMS & SECUENCIAS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## N-grams y Modelado de Secuencias")

    # N-grams
    st.markdown("### 🔗 Análisis de N-grams")
    ng_text = st.text_area(
        "Texto para analizar:",
        value="El modelo de lenguaje aprende las probabilidades de secuencias de palabras para generar texto coherente.",
        height=80,
    )
    top_n = st.slider("Top N n-gramas a mostrar:", 5, 20, 10)

    if st.button("🔗 Generar N-grams"):
        tokens = re.findall(r"\w+", ng_text.lower())

        col1, col2 = st.columns(2)

        for col, n, label in [(col1, 2, "Bigramas"), (col2, 3, "Trigramas")]:
            with col:
                grams = list(zip(*[tokens[i:] for i in range(n)]))
                counts = Counter(grams).most_common(top_n)
                if counts:
                    labels = [" ".join(g) for g, _ in counts]
                    values = [c for _, c in counts]
                    fig = go.Figure(go.Bar(
                        x=values[::-1],
                        y=labels[::-1],
                        orientation="h",
                        marker=dict(
                            color=values[::-1],
                            colorscale="Blues" if n == 2 else "Teal",
                            showscale=False,
                        ),
                    ))
                    fig.update_layout(
                        title=f"Top {label}",
                        paper_bgcolor="#13161f",
                        plot_bgcolor="#161923",
                        font=dict(color="#e8eaf0", family="DM Sans"),
                        title_font=dict(color="#4fc3f7"),
                        height=380,
                        margin=dict(l=0, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No hay suficientes tokens para {label}.")

    # Cuadro comparativo RNN / LSTM / GRU
    st.markdown("---")
    st.markdown("### 📋 Comparativo: RNN · LSTM · GRU")

    comp_data = {
        "Característica": [
            "Año de introducción",
            "Parámetros (comparativo)",
            "Manejo dependencias largas",
            "Velocidad de entrenamiento",
            "Riesgo vanishing gradient",
            "Mecanismo clave",
            "Uso recomendado",
        ],
        "RNN Vanilla": [
            "1986 (Rumelhart)",
            "⭐ Bajo",
            "❌ Muy limitado",
            "✅ Rápida",
            "🔴 Alto",
            "Estado oculto simple h_t",
            "Secuencias cortas",
        ],
        "LSTM": [
            "1997 (Hochreiter & Schmidhuber)",
            "⭐⭐⭐ Alto",
            "✅ Excelente",
            "🐢 Lenta",
            "🟢 Bajo (celdas de memoria)",
            "Compuertas input/forget/output + celda c_t",
            "NLP clásico, series temporales largas",
        ],
        "GRU": [
            "2014 (Cho et al.)",
            "⭐⭐ Medio",
            "✅ Bueno",
            "⚡ Media-rápida",
            "🟡 Moderado",
            "Compuertas reset/update",
            "Cuando LSTM es excesivo; buen balance",
        ],
    }

    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
    💡 <b>Vanishing gradient en RNN:</b> Al hacer backpropagation a través del tiempo (BPTT),
    los gradientes se multiplican repetidamente por pesos < 1, haciéndose exponencialmente pequeños.
    LSTM mitiga esto con la <b>celda de memoria</b> (c_t) que fluye con cambios aditivos.
    GRU simplifica las compuertas de LSTM manteniendo buen rendimiento con menos parámetros.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LABORATORIO LLM
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🌡️ Laboratorio LLM — Temperatura & Top-p")
    st.markdown("""
    <div class="info-box">
    Experimenta cómo la <b>temperatura</b> y el <b>top-p</b> afectan la creatividad y coherencia
    de las respuestas del modelo. Temperatura → 0 = determinista; → 2 = muy creativo/caótico.
    </div>
    """, unsafe_allow_html=True)

    if not GROQ_AVAILABLE:
        st.error("❌ La librería `groq` no está instalada. Ejecuta: `pip install groq`")
    elif not api_key:
        st.warning("🔑 Ingresa tu Groq API Key en la barra lateral para usar esta sección.")
    else:
        prompt_lab = st.text_area(
            "📝 Prompt para experimentar:",
            value="Escribe una historia corta sobre un robot que descubre las emociones.",
            height=80,
        )

        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("🌡️ Temperatura", 0.0, 2.0, 0.7, 0.1)
        with col2:
            top_p = st.slider("🎯 Top-p", 0.1, 1.0, 0.9, 0.05)

        # Visual de distribución softmax simulada
        st.markdown("#### Visualización de distribución Softmax (simulada)")
        tokens_demo = ["robot", "máquina", "ser", "ente", "criatura", "androide", "autómata", "cosa"]
        logits = np.array([3.2, 2.1, 1.8, 1.5, 1.2, 0.9, 0.5, 0.2])

        def softmax_temp(logits, temp):
            if temp == 0:
                temp = 1e-6
            shifted = logits / temp
            exp_vals = np.exp(shifted - np.max(shifted))
            return exp_vals / exp_vals.sum()

        probs = softmax_temp(logits, temperature if temperature > 0 else 0.01)
        fig_soft = go.Figure(go.Bar(
            x=tokens_demo,
            y=probs,
            marker=dict(
                color=probs,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="P(token)", tickfont=dict(color="#e8eaf0")),
            ),
        ))
        fig_soft.update_layout(
            title=f"Distribución de probabilidad con T={temperature}",
            xaxis_title="Token candidato",
            yaxis_title="Probabilidad",
            paper_bgcolor="#13161f",
            plot_bgcolor="#161923",
            font=dict(color="#e8eaf0", family="DM Sans"),
            title_font=dict(color="#4fc3f7"),
        )
        st.plotly_chart(fig_soft, use_container_width=True)

        if st.button("🚀 Generar Respuesta"):
            try:
                client = Groq(api_key=api_key)
                with st.spinner("Generando respuesta..."):
                    start = time.time()
                    resp = client.chat.completions.create(
                        model=model_choice,
                        messages=[{"role": "user", "content": prompt_lab}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=400,
                    )
                    elapsed = time.time() - start

                answer = resp.choices[0].message.content
                usage = resp.usage

                st.markdown("#### ✅ Respuesta del Modelo")
                st.markdown(f'<div class="nlp-card">{answer}</div>', unsafe_allow_html=True)

                # Métricas
                tps = round(usage.completion_tokens / elapsed, 1) if elapsed > 0 else 0
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱ Latencia", f"{elapsed:.2f}s")
                c2.metric("🔢 Tokens generados", usage.completion_tokens)
                c3.metric("⚡ Tokens/seg", tps)

            except Exception as e:
                st.error(f"Error al llamar a Groq: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AGENTE CONVERSACIONAL
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🤖 Agente Conversacional Especializado")

    if not GROQ_AVAILABLE:
        st.error("❌ La librería `groq` no está instalada.")
    elif not api_key:
        st.warning("🔑 Ingresa tu Groq API Key en la barra lateral.")
    else:
        # Seleccionar persona del agente
        agent_type = st.selectbox("🎭 Tipo de Agente:", [
            "📚 Consultor Académico NLP/IA",
            "💻 Asistente de Código Python",
            "⚽ Experto en Deportes",
            "🔬 Científico de Datos",
        ])

        system_prompts = {
            "📚 Consultor Académico NLP/IA": (
                "Eres un consultor académico especializado en Inteligencia Artificial, "
                "NLP y Machine Learning. Respondes con rigor científico, citas conceptos "
                "clave y usas ejemplos pedagógicos. Eres parte del equipo docente de la "
                "Maestría en Ciencia de Datos de EAFIT. Responde siempre en español."
            ),
            "💻 Asistente de Código Python": (
                "Eres un experto en Python para ciencia de datos. Proporcionas código "
                "limpio, documentado y siguiendo buenas prácticas (PEP8). Siempre explicas "
                "el razonamiento detrás del código. Responde en español."
            ),
            "⚽ Experto en Deportes": (
                "Eres un analista deportivo con amplio conocimiento en fútbol, estadísticas "
                "deportivas y ciencia del deporte. Combinas datos con narrativa apasionante. "
                "Responde en español con entusiasmo."
            ),
            "🔬 Científico de Datos": (
                "Eres un científico de datos senior con experiencia en ML, estadística y "
                "visualización. Explicas conceptos técnicos con claridad, usas analogías "
                "y siempre mencionas casos de uso reales. Responde en español."
            ),
        }

        system_msg = system_prompts[agent_type]

        # Historial
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "metrics_history" not in st.session_state:
            st.session_state.metrics_history = []

        # Renderizar historial
        for msg in st.session_state.messages:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            bg = "#13161f" if msg["role"] == "user" else "#161923"
            border = "#1e2a3a" if msg["role"] == "user" else "#1e3a5f"
            st.markdown(
                f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
                f'padding:0.9rem 1.1rem;margin-bottom:0.6rem">'
                f'<b style="color:#4fc3f7">{role_icon} {msg["role"].capitalize()}</b><br>'
                f'<span style="color:#e8eaf0">{msg["content"]}</span></div>',
                unsafe_allow_html=True,
            )

        user_input = st.chat_input("Escribe tu pregunta al agente...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            try:
                client = Groq(api_key=api_key)
                messages_payload = [{"role": "system", "content": system_msg}] + st.session_state.messages

                with st.spinner("🤔 El agente está pensando..."):
                    t0 = time.time()
                    resp = client.chat.completions.create(
                        model=model_choice,
                        messages=messages_payload,
                        temperature=0.7,
                        max_tokens=600,
                    )
                    latency = time.time() - t0

                answer = resp.choices[0].message.content
                usage = resp.usage
                tps = round(usage.completion_tokens / latency, 1) if latency > 0 else 0

                st.session_state.messages.append({"role": "assistant", "content": answer})

                # LLM-as-a-Judge
                with st.spinner("🧑‍⚖️ Evaluando respuesta con LLM-as-a-Judge..."):
                    judge_prompt = (
                        f"Evalúa la siguiente respuesta de un asistente IA del 1 al 10 "
                        f"basándote en veracidad, claridad y completitud.\n\n"
                        f"Pregunta del usuario: {user_input}\n\n"
                        f"Respuesta del asistente: {answer}\n\n"
                        f"Devuelve SOLO un JSON con este formato exacto, sin texto adicional:\n"
                        f'{{\"score\": <número del 1 al 10>, \"justification\": \"<breve justificación>\"}}'
                    )
                    judge_resp = client.chat.completions.create(
                        model=model_choice,
                        messages=[{"role": "user", "content": judge_prompt}],
                        temperature=0.1,
                        max_tokens=200,
                    )

                judge_text = judge_resp.choices[0].message.content.strip()
                score = None
                justification = ""
                try:
                    import json
                    # Extraer JSON aunque haya texto alrededor
                    json_match = re.search(r'\{[^}]+\}', judge_text, re.DOTALL)
                    if json_match:
                        judge_data = json.loads(json_match.group())
                        score = int(judge_data.get("score", 0))
                        justification = judge_data.get("justification", "")
                except Exception:
                    score = None

                st.session_state.metrics_history.append({
                    "Pregunta": user_input[:40] + "...",
                    "Latencia (s)": round(latency, 2),
                    "TPS": tps,
                    "Tokens": usage.completion_tokens,
                    "Score (1-10)": score if score else "—",
                })

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

        # Panel de métricas
        if st.session_state.metrics_history:
            st.markdown("---")
            st.markdown("### 📊 Métricas de Desempeño del Agente")
            metrics_df = pd.DataFrame(st.session_state.metrics_history)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⏱ Última latencia", f"{metrics_df['Latencia (s)'].iloc[-1]}s")
            c2.metric("⚡ Último TPS", f"{metrics_df['TPS'].iloc[-1]}")
            c3.metric("🔢 Tokens totales", int(metrics_df["Tokens"].sum()))
            last_score = metrics_df["Score (1-10)"].iloc[-1]
            c4.metric("🧑‍⚖️ Score LLM-Judge", f"{last_score}/10" if last_score != "—" else "—")

            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Score history chart
            valid_scores = [(i, r) for i, r in enumerate(metrics_df["Score (1-10)"]) if r != "—"]
            if len(valid_scores) > 1:
                idx, scores = zip(*valid_scores)
                fig_score = go.Figure(go.Scatter(
                    x=list(idx),
                    y=list(scores),
                    mode="lines+markers",
                    line=dict(color="#4fc3f7", width=2),
                    marker=dict(size=8, color="#4fc3f7"),
                ))
                fig_score.update_layout(
                    title="Evolución del Score LLM-as-a-Judge",
                    yaxis=dict(range=[0, 10.5]),
                    paper_bgcolor="#13161f",
                    plot_bgcolor="#161923",
                    font=dict(color="#e8eaf0"),
                    title_font=dict(color="#4fc3f7"),
                    height=260,
                )
                st.plotly_chart(fig_score, use_container_width=True)

        if st.button("🗑️ Limpiar conversación"):
            st.session_state.messages = []
            st.session_state.metrics_history = []
            st.rerun()
