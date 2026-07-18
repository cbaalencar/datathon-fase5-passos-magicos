"""
Passos Mágicos — Preditor de Risco de Defasagem
Datathon Fase 5 · Pós-Tech FIAP

App Streamlit que carrega o modelo treinado (Notebook 02) e estima a
probabilidade de um aluno AGRAVAR sua defasagem no próximo ciclo, com base
nos indicadores do ano atual. Saída em faixas de risco (Baixo/Médio/Alto).
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Risco de Defasagem — Passos Mágicos",
    page_icon="📚",
    layout="wide",
)

# O Streamlit Cloud roda a partir da raiz do repositorio, entao os caminhos sao montados
# a partir da pasta deste arquivo. Procura em models/ (fonte oficial) e, se nao achar,
# ao lado do app.
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)

def _achar(nome):
    for pasta in (os.path.join(RAIZ, "models"), AQUI, RAIZ):
        caminho = os.path.join(pasta, nome)
        if os.path.isfile(caminho):
            return caminho
    return nome

MODELO_PATH = _achar("modelo_risco_defasagem.pkl")
META_PATH = _achar("modelo_meta.json")

# Faixas de cada indicador na base PEDE (min, max, default)
LIMITES = {
    "Defasagem": (-4, 3, -1),
    "fase_num": (0, 7, 2),
    "IDA": (0.0, 10.0, 6.7),
    "IEG": (0.0, 10.0, 8.7),
    "IAA": (0.0, 10.0, 8.8),
    "IPS": (2.5, 10.0, 7.5),
    "IPV": (2.5, 10.0, 7.6),
    "IPP": (0.1, 10.0, 7.5),
}
ROTULOS = {
    "Defasagem": "Defasagem (anos vs. série ideal; negativo = atrasado)",
    "fase_num": "Fase atual (0=ALFA … 7)",
    "IDA": "IDA — Desempenho Acadêmico",
    "IEG": "IEG — Engajamento",
    "IAA": "IAA — Autoavaliação",
    "IPS": "IPS — Psicossocial",
    "IPV": "IPV — Ponto de Virada",
    "IPP": "IPP — Psicopedagógico",
}


# Carregamento do modelo (cacheado)
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(MODELO_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta


def classificar_faixa(prob, corte_medio, corte_alto):
    if prob < corte_medio:
        return "Baixo"
    if prob < corte_alto:
        return "Médio"
    return "Alto"


def cor_faixa(faixa):
    return {"Baixo": "#2E7D32", "Médio": "#F9A825", "Alto": "#C62828"}[faixa]


# P25 de cada indice na base. Abaixo disso, sinaliza foco de acao.
LIMIAR_ATENCAO = {"IDA": 5.1, "IEG": 7.5, "IAA": 7.9, "IPS": 5.6, "IPV": 7.0, "IPP": 6.3}
ACAO_POR_INDICE = {
    "IDA": "Reforço acadêmico direcionado (Mat/Por/Ing) e revisão do plano de estudos.",
    "IEG": "Baixo engajamento: retomar contato com o aluno, entender faltas/entregas e reengajar.",
    "IAA": "Autoavaliação baixa: acolhimento e conversa sobre autoconfiança e pertencimento.",
    "IPS": "Sinal psicossocial: encaminhar para acompanhamento das psicólogas / rede de apoio.",
    "IPV": "Ponto de virada frágil: mentoria e definição de metas de curto prazo com o educador.",
    "IPP": "Avaliação psicopedagógica sugere atenção: revisar estratégias de aprendizagem.",
}


def recomendacoes_acao(entradas):
    """Retorna lista de ações práticas para os índices do aluno abaixo do P25 da base."""
    baixos = [(ind, entradas[ind]) for ind, lim in LIMIAR_ATENCAO.items()
              if ind in entradas and entradas[ind] < lim]
    baixos.sort(key=lambda x: x[1])
    return [(ind, val, LIMIAR_ATENCAO[ind], ACAO_POR_INDICE[ind]) for ind, val in baixos]


try:
    modelo, meta = carregar_modelo()
except Exception as e:
    st.error(
        "Não foi possível carregar o modelo. Verifique se os arquivos "
        "`modelo_risco_defasagem.pkl` e `modelo_meta.json` estão em `models/` "
        f"e se o `requirements.txt` fixa scikit-learn==1.6.1.\n\nDetalhe: {e}"
    )
    st.stop()

NUM = meta["num_features"]
CAT = meta["cat_features"]
CORTE_MEDIO = meta.get("corte_medio", 0.185)
CORTE_ALTO = meta.get("corte_alto", 0.371)


# Cabeçalho
st.title("📚 Preditor de Risco de Defasagem")
st.caption(
    "Associação Passos Mágicos · Datathon Fase 5. "
    "O modelo estima a **probabilidade de o aluno agravar sua defasagem no próximo ciclo**, "
    "a partir dos indicadores do ano atual — permitindo priorizar acompanhamento **antes** da piora."
)

with st.expander("ℹ️ Como interpretar o resultado"):
    st.markdown(
        f"""
- O modelo prevê **agravamento de defasagem** de um ano para o seguinte (alvo temporal).
- A saída é uma **probabilidade** (0 a 100%), traduzida em três faixas:
  - 🟢 **Baixo** — risco abaixo da média da base (< {CORTE_MEDIO:.0%}).
  - 🟡 **Médio** — risco entre a média e o dobro da média.
  - 🔴 **Alto** — risco ≥ **{CORTE_ALTO:.0%}** (pelo menos o dobro da média da base).
- Desempenho do modelo: **ROC-AUC ≈ {meta.get('auc_cv', 0.88)}** (validação cruzada por aluno).
- ⚠️ Ferramenta de **apoio à priorização**, não de decisão automática. A avaliação
  pedagógica e psicossocial da equipe é sempre soberana.
"""
    )

aba_ind, aba_lote = st.tabs(["👤 Aluno individual", "📄 Turma (lote via CSV)"])

# Aba 1 — Aluno individual
with aba_ind:
    st.subheader("Indicadores do aluno no ano atual")
    col1, col2 = st.columns(2)
    entradas = {}

    with col1:
        for c in ["Defasagem", "fase_num", "IDA", "IEG"]:
            lo, hi, dv = LIMITES[c]
            if c in ("Defasagem", "fase_num"):
                entradas[c] = st.slider(ROTULOS[c], int(lo), int(hi), int(dv), step=1)
            else:
                entradas[c] = st.slider(ROTULOS[c], float(lo), float(hi), float(dv), step=0.1)
    with col2:
        for c in ["IAA", "IPS", "IPV", "IPP"]:
            lo, hi, dv = LIMITES[c]
            entradas[c] = st.slider(ROTULOS[c], float(lo), float(hi), float(dv), step=0.1)

    st.markdown("**Perfil**")
    colg, colr = st.columns(2)
    with colg:
        genero = st.selectbox("Gênero", ["F", "M"], index=0)
    with colr:
        rede = st.selectbox("Rede de ensino", ["Publica", "Particular"], index=0)
    entradas["Genero_pad"] = genero
    entradas["rede"] = rede

    if st.button("Calcular risco", type="primary"):
        linha = pd.DataFrame([entradas])[NUM + CAT]
        prob = float(modelo.predict_proba(linha)[:, 1][0])
        faixa = classificar_faixa(prob, CORTE_MEDIO, CORTE_ALTO)

        st.markdown("---")
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Probabilidade de agravar defasagem", f"{prob:.0%}")
        with m2:
            st.markdown(
                f"<div style='padding:1rem;border-radius:8px;background:{cor_faixa(faixa)};"
                f"color:white;font-size:1.4rem;font-weight:600;text-align:center'>"
                f"Risco {faixa}</div>",
                unsafe_allow_html=True,
            )
        st.progress(min(prob, 1.0))

        # Acoes sugeridas pelos indices abaixo do P25
        acoes = recomendacoes_acao(entradas)
        st.markdown("#### 🎯 Recomendação de ação")
        if faixa == "Alto":
            st.warning("Aluno em **risco alto** — acompanhamento prioritário neste ciclo.")
        elif faixa == "Médio":
            st.info("Risco intermediário — monitoramento próximo recomendado.")
        else:
            st.success("Risco baixo de agravamento neste momento.")

        if acoes:
            st.markdown("**Focos sugeridos** (indicadores abaixo do padrão da rede):")
            for ind, val, lim, acao in acoes:
                st.markdown(f"- **{ind} = {val:.1f}** (abaixo de {lim:.1f}): {acao}")
        elif faixa != "Baixo":
            st.markdown(
                "Nenhum indicador isolado está abaixo do padrão, mas a **combinação** eleva o risco. "
                "Recomenda-se acompanhamento integral e reavaliação no próximo ciclo."
            )
        else:
            st.markdown("Todos os indicadores dentro do padrão. Manter acompanhamento regular.")

# Aba 2 — Lote via CSV
with aba_lote:
    st.subheader("Pontuar uma turma inteira")
    st.markdown(
        "Envie um CSV com uma linha por aluno e as colunas: "
        f"`{', '.join(NUM + CAT)}`. Uma coluna identificadora (ex.: `RA`) é opcional "
        "e será mantida no resultado."
    )

    modelo_exemplo = pd.DataFrame(
        [{**{c: LIMITES[c][2] for c in NUM}, "Genero_pad": "F", "rede": "Publica", "RA": "RA-000"}]
    )[["RA"] + NUM + CAT]
    st.download_button(
        "⬇️ Baixar CSV de exemplo (cabeçalho)",
        modelo_exemplo.to_csv(index=False).encode("utf-8"),
        "exemplo_entrada.csv",
        "text/csv",
    )

    arquivo = st.file_uploader("CSV da turma", type=["csv"])
    if arquivo is not None:
        try:
            dados = pd.read_csv(arquivo)
        except Exception as e:
            st.error(f"Não consegui ler o CSV: {e}")
            st.stop()

        faltando = [c for c in NUM + CAT if c not in dados.columns]
        if faltando:
            st.error(f"Faltam colunas obrigatórias: {faltando}")
            st.stop()

        X = dados[NUM + CAT].copy()
        if X.isna().any().any():
            st.warning(
                "Há valores ausentes nas colunas de entrada. Linhas com ausência "
                "não podem ser pontuadas e serão marcadas como vazias."
            )

        validas = X.notna().all(axis=1)
        dados["probabilidade_risco"] = np.nan
        dados.loc[validas, "probabilidade_risco"] = modelo.predict_proba(X[validas])[:, 1]
        dados["faixa_risco"] = dados["probabilidade_risco"].apply(
            lambda p: classificar_faixa(p, CORTE_MEDIO, CORTE_ALTO) if pd.notna(p) else "—"
        )

        # Foco de acao: indices abaixo do P25, mais critico primeiro
        def foco_aluno(r):
            baixos = [(ind, r[ind]) for ind, lim in LIMIAR_ATENCAO.items()
                      if ind in r and pd.notna(r[ind]) and r[ind] < lim]
            baixos.sort(key=lambda x: x[1])
            return ", ".join(f"{ind}" for ind, _ in baixos) if baixos else "—"
        dados["foco_acao"] = dados.apply(foco_aluno, axis=1)

        st.markdown("### Resultado")
        c1, c2, c3 = st.columns(3)
        cont = dados["faixa_risco"].value_counts()
        c1.metric("🔴 Alto", int(cont.get("Alto", 0)))
        c2.metric("🟡 Médio", int(cont.get("Médio", 0)))
        c3.metric("🟢 Baixo", int(cont.get("Baixo", 0)))

        ordenado = dados.sort_values("probabilidade_risco", ascending=False, na_position="last")
        mostrar = [c for c in ["RA"] if c in ordenado.columns] + [
            "probabilidade_risco", "faixa_risco", "foco_acao"
        ] + NUM + CAT
        st.dataframe(
            ordenado[mostrar].style.format({"probabilidade_risco": "{:.0%}"}),
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Baixar resultado (CSV)",
            ordenado.to_csv(index=False).encode("utf-8"),
            "risco_turma.csv",
            "text/csv",
        )

st.markdown("---")
st.caption(
    "Modelo: Random Forest · alvo = agravamento de defasagem (T→T+1) · "
    "features do ano atual. Uso restrito a apoio à decisão da equipe Passos Mágicos."
)
