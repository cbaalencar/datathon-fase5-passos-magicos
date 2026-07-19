"""
Passos Magicos - Risco de Defasagem
Datathon Fase 5 - Pos-Tech FIAP

Carrega o modelo do Notebook 02 e estima, em tempo real, a probabilidade de o
aluno agravar a defasagem no proximo ciclo. Avaliacao individual ou por turma.
"""
import json
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Risco de Defasagem — Passos Mágicos",
    page_icon="📘",
    layout="wide",
)

ANO_ATUAL = date.today().year

# Paleta institucional, extraida do relatorio de atividades 2024 da ONG.
MARINHO = "#001860"
AZUL = "#0060A8"
AZUL_CLARO = "#78A8C0"
VERDE = "#4A8A3D"
AMBAR = "#C68A16"
TERRACOTA = "#A8452C"
TINTA = "#12203D"
PAPEL = "#F7F9FC"

st.markdown(f"""
<style>
.stApp {{ background: {PAPEL}; }}
h1, h2, h3 {{ color: {MARINHO}; letter-spacing: -0.01em; }}
[data-testid="stWidgetLabel"] p {{ color: {TINTA}; font-weight: 600; font-size: .88rem; }}
[data-testid="stSliderTickBar"] {{ color: {AZUL_CLARO}; }}
.rotulo {{
  font-size: .68rem; letter-spacing: .16em; text-transform: uppercase;
  color: {AZUL}; font-weight: 700; margin: 1.1rem 0 .2rem 0;
}}
.prob {{
  font-size: 4rem; font-weight: 800; line-height: 1;
  color: {MARINHO}; font-variant-numeric: tabular-nums;
}}
.selo {{
  display: inline-block; padding: .3rem 1rem; border-radius: 2px;
  color: #fff; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  font-size: .82rem;
}}
.faixa-nota {{ color: {TINTA}; font-size: .9rem; margin-top: .4rem; }}
.acao {{
  border-left: 3px solid {AZUL}; padding: .3rem 0 .3rem .7rem;
  margin: .3rem 0; font-size: .9rem; color: {TINTA};
}}
.acao b {{ color: {MARINHO}; }}
.calc {{ color: {AZUL}; font-size: .85rem; margin-top: .2rem; }}
</style>
""", unsafe_allow_html=True)

# O Streamlit Cloud executa a partir da raiz do repositorio; resolvo os caminhos
# pela pasta deste arquivo, procurando primeiro em models/.
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)

def _achar(nome):
    for pasta in (os.path.join(RAIZ, "models"), AQUI, RAIZ):
        c = os.path.join(pasta, nome)
        if os.path.isfile(c):
            return c
    return nome

MODELO_PATH = _achar("modelo_risco_defasagem.pkl")
META_PATH = _achar("modelo_meta.json")

# Rotulos exibidos -> categorias que o modelo conhece (treinado sem acento/abreviado).
GENERO_UI = {"Feminino": "F", "Masculino": "M"}
REDE_UI = {"Pública": "Publica", "Particular": "Particular"}

FASES = {
    0: "Fase 0 · Alfa (1º–2º ano)", 1: "Fase 1 (3º–4º ano)", 2: "Fase 2 (5º–6º ano)",
    3: "Fase 3 (7º–8º ano)", 4: "Fase 4 (9º ano)", 5: "Fase 5 (1º EM)",
    6: "Fase 6 (2º EM)", 7: "Fase 7 (3º EM)",
}

FASES_IDEAL = {**FASES, 8: "Fase 8 (18+)"}

def fase_ideal_por_idade(idade):
    # Tabela 4 do documento PEDE (idade -> fase esperada).
    if idade <= 8: return 0
    if idade == 9: return 1
    if idade <= 11: return 2
    if idade <= 13: return 3
    if idade == 14: return 4
    if idade == 15: return 5
    if idade == 16: return 6
    if idade == 17: return 7
    return 8

INDICES = {
    "IEG": ("IEG — Engajamento", 0.0, 10.0, 7.5),
    "IAA": ("IAA — Autoavaliação", 0.0, 10.0, 8.0),
    "IPS": ("IPS — Psicossocial", 2.5, 10.0, 6.5),
    "IPV": ("IPV — Ponto de Virada", 2.5, 10.0, 7.0),
    "IPP": ("IPP — Psicopedagógico", 0.1, 10.0, 6.8),
}

@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(MODELO_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta

def classificar_faixa(p, cm, ca):
    return "Baixo" if p < cm else ("Médio" if p < ca else "Alto")

def cor_faixa(faixa):
    return {"Baixo": VERDE, "Médio": AMBAR, "Alto": TERRACOTA}[faixa]

def medidor(prob, cm, ca):
    pos = min(max(prob, 0), 1) * 100
    m, a = cm * 100, ca * 100
    return f"""
<div style="margin:.2rem 0 1rem 0;">
  <div style="position:relative;height:14px;border-radius:2px;overflow:hidden;
              background:linear-gradient(to right,
                {VERDE} 0%, {VERDE} {m}%,
                {AMBAR} {m}%, {AMBAR} {a}%,
                {TERRACOTA} {a}%, {TERRACOTA} 100%);"></div>
  <div style="position:relative;height:0;">
    <div style="position:absolute;left:{pos}%;transform:translateX(-50%);top:-19px;
                width:3px;height:24px;background:{TINTA};border:1px solid #fff;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:.45rem;
              font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:{AZUL_CLARO};">
    <span>0%</span><span>Média da rede {cm:.0%}</span>
    <span>Dobro da média {ca:.0%}</span><span>100%</span>
  </div>
</div>"""

# P25 de cada indice na base; abaixo disso vira foco de acao.
LIMIAR_ATENCAO = {"IDA": 5.1, "IEG": 7.5, "IAA": 7.9, "IPS": 5.6, "IPV": 7.0, "IPP": 6.3}
ACAO_POR_INDICE = {
    "IDA": "Reforço acadêmico direcionado (Mat/Por/Ing) e revisão do plano de estudos.",
    "IEG": "Baixo engajamento: retomar contato, entender faltas e entregas, reengajar.",
    "IAA": "Autoavaliação baixa: acolhimento e conversa sobre autoconfiança e pertencimento.",
    "IPS": "Sinal psicossocial: encaminhar ao acompanhamento das psicólogas / rede de apoio.",
    "IPV": "Ponto de virada frágil: mentoria e metas de curto prazo com o educador.",
    "IPP": "Avaliação psicopedagógica pede atenção: revisar estratégias de aprendizagem.",
}

def recomendacoes_acao(entradas):
    baixos = [(i, entradas[i]) for i, lim in LIMIAR_ATENCAO.items()
              if i in entradas and entradas[i] < lim]
    baixos.sort(key=lambda x: x[1])
    return [(i, v, LIMIAR_ATENCAO[i], ACAO_POR_INDICE[i]) for i, v in baixos]

try:
    modelo, meta = carregar_modelo()
except Exception as e:
    st.error(
        "Não foi possível carregar o modelo. Verifique se `modelo_risco_defasagem.pkl` "
        "e `modelo_meta.json` estão em `models/` e se o `requirements.txt` fixa "
        f"scikit-learn==1.6.1.\n\nDetalhe: {e}"
    )
    st.stop()

NUM = meta["num_features"]
CAT = meta["cat_features"]
CORTE_MEDIO = meta.get("corte_medio", 0.185)
CORTE_ALTO = meta.get("corte_alto", 0.371)

st.markdown(f"""
<div style="border-left:6px solid {AZUL};padding-left:1rem;margin-bottom:.4rem;">
  <div class="rotulo" style="margin-top:0;">Associação Passos Mágicos · PEDE 2022–2024</div>
  <h1 style="margin:.1rem 0 .2rem 0;font-size:2.2rem;">Risco de defasagem</h1>
  <div style="color:{TINTA};font-size:1rem;max-width:72ch;">
    Probabilidade de o aluno agravar sua defasagem no próximo ciclo, a partir dos
    indicadores deste ano ({ANO_ATUAL}). Serve para priorizar acompanhamento antes da piora.
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Como interpretar o resultado"):
    st.markdown(f"""
- O modelo prevê **agravamento de defasagem** de um ano para o seguinte.
- A saída é uma **probabilidade**, traduzida em três faixas:
  **Baixo** (abaixo da média da rede, {CORTE_MEDIO:.0%}) · **Médio** · **Alto** (≥ {CORTE_ALTO:.0%}, o dobro da média).
- Desempenho: **ROC-AUC ≈ {meta.get('auc_cv', 0.87)}** em validação cruzada por aluno.
- Ferramenta de **apoio à priorização** — a avaliação da equipe pedagógica e psicossocial é soberana.
""")

aba_ind, aba_lote = st.tabs(["Aluno individual", "Turma (planilha)"])

# ---------- Aluno individual ----------
with aba_ind:
    col_e, col_r = st.columns([1.5, 1], gap="large")

    with col_e:
        st.markdown('<div class="rotulo">1 · Situação escolar</div>', unsafe_allow_html=True)
        fase = st.selectbox("Fase atual na Associação", options=list(FASES),
                            index=4, format_func=lambda i: FASES[i])

        modo_def = st.radio("Calcular a defasagem a partir de:",
                            ["Defasagem direta", "Fase ideal", "Idade atual", "Ano de nascimento"],
                            horizontal=True)

        def _mostra_calculo(idade_txt, ideal):
            nome_ideal = FASES_IDEAL[ideal].split(" (")[0]
            st.markdown(
                f'<div class="calc">{idade_txt}Fase ideal pela Tabela 4: <b>{nome_ideal}</b>'
                f' · defasagem: <b>{defasagem:+d}</b></div>', unsafe_allow_html=True)

        if modo_def == "Defasagem direta":
            defasagem = st.number_input("Defasagem (Fase atual − Fase ideal; negativo = atrasado)",
                                        min_value=-4, max_value=3, value=0, step=1)
        elif modo_def == "Fase ideal":
            ideal = st.selectbox("Fase ideal para a idade do aluno", options=list(FASES_IDEAL),
                                 index=fase, format_func=lambda i: FASES_IDEAL[i])
            defasagem = int(np.clip(fase - ideal, -4, 3))
            _mostra_calculo("", ideal)
        elif modo_def == "Idade atual":
            idade = st.number_input("Idade do aluno (anos completos em 31/12)",
                                    min_value=6, max_value=22, value=14, step=1)
            ideal = fase_ideal_por_idade(idade)
            defasagem = int(np.clip(fase - ideal, -4, 3))
            _mostra_calculo("", ideal)
            if idade == 8:
                st.caption("Aos 8 anos a Tabela 4 admite Alfa ou Fase 1. Aplicamos Alfa, "
                           "critério usado no tratamento da base (98% dos alunos de 8 anos "
                           "cursam Alfa). Se este aluno já está no 3º ano, use a defasagem direta.")
        else:
            nasc = st.number_input("Ano de nascimento", min_value=ANO_ATUAL - 22,
                                   max_value=ANO_ATUAL - 6, value=ANO_ATUAL - 14, step=1)
            idade = ANO_ATUAL - nasc
            ideal = fase_ideal_por_idade(idade)
            defasagem = int(np.clip(fase - ideal, -4, 3))
            _mostra_calculo(f"Idade em {ANO_ATUAL}: {idade} anos · ", ideal)
            if idade == 8:
                st.caption("Aos 8 anos a Tabela 4 admite Alfa ou Fase 1. Aplicamos Alfa, "
                           "critério usado no tratamento da base (98% dos alunos de 8 anos "
                           "cursam Alfa). Se este aluno já está no 3º ano, use a defasagem direta.")

        st.markdown('<div class="rotulo">2 · Desempenho acadêmico</div>', unsafe_allow_html=True)
        modo_ida = st.radio("IDA", ["Informar o IDA", "Calcular pelas notas"],
                            horizontal=True, label_visibility="collapsed")
        if modo_ida == "Calcular pelas notas":
            tem_ing = fase >= 3   # Ingles nao e avaliado nas Fases 0-2
            ncols = st.columns(3 if tem_ing else 2)
            mat = ncols[0].number_input("Matemática", 0.0, 10.0, 6.0, step=0.1, format="%.1f")
            por = ncols[1].number_input("Português", 0.0, 10.0, 6.0, step=0.1, format="%.1f")
            notas = [mat, por]
            if tem_ing:
                ing = ncols[2].number_input("Inglês", 0.0, 10.0, 6.0, step=0.1, format="%.1f")
                notas.append(ing)
            else:
                st.markdown('<div class="calc">Inglês não é avaliado nas Fases 0–2 — '
                            'média sobre Matemática e Português.</div>', unsafe_allow_html=True)
            ida = round(float(np.mean(notas)), 2)
            st.markdown(f'<div class="calc"><b>IDA calculado: {ida:.2f}</b></div>',
                        unsafe_allow_html=True)
        else:
            ida = st.number_input("IDA — Desempenho Acadêmico (média das notas)",
                                  0.0, 10.0, 6.00, step=0.01, format="%.2f")

        st.markdown('<div class="rotulo">3 · Indicadores de acompanhamento</div>',
                    unsafe_allow_html=True)
        vals = {}
        linha1 = st.columns(3)
        linha2 = st.columns(3)
        for coluna, chave in zip(linha1 + linha2, INDICES):
            rot, lo, hi, dv = INDICES[chave]
            vals[chave] = coluna.number_input(rot, lo, hi, dv, step=0.1, format="%.2f")

        st.markdown('<div class="rotulo">4 · Perfil</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        genero_ui = p1.selectbox("Gênero", list(GENERO_UI))
        rede_ui = p2.selectbox("Rede de ensino", list(REDE_UI))

    entradas = {
        "Defasagem": defasagem, "fase_num": fase, "IDA": ida,
        "IEG": vals["IEG"], "IAA": vals["IAA"], "IPS": vals["IPS"],
        "IPV": vals["IPV"], "IPP": vals["IPP"],
        "Genero_pad": GENERO_UI[genero_ui], "rede": REDE_UI[rede_ui],
    }
    prob = float(modelo.predict_proba(pd.DataFrame([entradas])[NUM + CAT])[:, 1][0])
    faixa = classificar_faixa(prob, CORTE_MEDIO, CORTE_ALTO)
    acoes = recomendacoes_acao(entradas)

    with col_r:
        st.markdown('<div class="rotulo">Probabilidade de agravar</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prob">{prob:.0%}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin:.6rem 0 1rem 0;"><span class="selo" '
            f'style="background:{cor_faixa(faixa)};">Risco {faixa.lower()}</span></div>',
            unsafe_allow_html=True)
        st.markdown('<div class="rotulo">Posição na escala</div>', unsafe_allow_html=True)
        st.markdown(medidor(prob, CORTE_MEDIO, CORTE_ALTO), unsafe_allow_html=True)
        leitura = {"Alto": "Acompanhamento prioritário neste ciclo.",
                   "Médio": "Monitoramento próximo ao longo do ciclo.",
                   "Baixo": "Sem sinal de agravamento no momento."}[faixa]
        st.markdown(f'<div class="faixa-nota">{leitura}</div>', unsafe_allow_html=True)

        st.markdown('<div class="rotulo">Onde agir</div>', unsafe_allow_html=True)
        if acoes:
            for ind, val, lim, acao in acoes:
                st.markdown(
                    f'<div class="acao"><b>{ind} {val:.2f}</b> '
                    f'<span style="color:#7A879E;">(padrão da rede: {lim:.1f})</span><br>{acao}</div>',
                    unsafe_allow_html=True)
        elif faixa != "Baixo":
            st.markdown('<div class="acao">Nenhum indicador isolado abaixo do padrão — a '
                        'combinação deles eleva o risco. Acompanhamento integral e reavaliação '
                        'no próximo ciclo.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="acao">Indicadores dentro do padrão da rede. Manter o '
                        'acompanhamento regular.</div>', unsafe_allow_html=True)

# ---------- Turma ----------
with aba_lote:
    st.markdown('<div class="rotulo">Avaliar vários alunos de uma vez</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Em vez de digitar aluno por aluno, envie uma **planilha da turma** e receba todos "
        "avaliados de uma vez, em ordem de prioridade.\n\n"
        "1. Baixe o **modelo de planilha** abaixo (uma linha por aluno).\n"
        "2. Preencha os indicadores de cada aluno — a coluna `RA` identifica o estudante.\n"
        "3. Envie o arquivo de volta. O resultado traz a **probabilidade**, a **faixa de risco** "
        "e o **foco de ação** de cada aluno, do maior risco para o menor."
    )

    exemplo = pd.DataFrame([{
        "RA": "RA-000", "Defasagem": 0, "fase_num": 4, "IDA": 6.0, "IEG": 7.5,
        "IAA": 8.0, "IPS": 6.5, "IPV": 7.0, "IPP": 6.8,
        "Genero_pad": "Feminino", "rede": "Pública",
    }])[["RA"] + NUM + CAT]
    st.download_button("Baixar modelo de planilha (CSV)",
                       exemplo.to_csv(index=False).encode("utf-8"),
                       "modelo_turma.csv", "text/csv")

    arquivo = st.file_uploader("Planilha da turma (CSV)", type=["csv"])
    if arquivo is not None:
        try:
            dados = pd.read_csv(arquivo)
        except Exception as e:
            st.error(f"Não consegui ler o arquivo: {e}")
            st.stop()

        faltando = [c for c in NUM + CAT if c not in dados.columns]
        if faltando:
            st.error(f"Faltam colunas obrigatórias: {faltando}. Use o modelo de planilha.")
            st.stop()

        # Aceita valores amigaveis e converte para as categorias do modelo.
        dados["Genero_pad"] = dados["Genero_pad"].astype(str).str.strip().replace(
            {"Feminino": "F", "feminino": "F", "Masculino": "M", "masculino": "M"})
        dados["rede"] = dados["rede"].astype(str).str.strip().replace(
            {"Pública": "Publica", "pública": "Publica", "publica": "Publica"})

        X = dados[NUM + CAT].copy()
        validas = X.notna().all(axis=1)
        if (~validas).any():
            st.warning(f"{int((~validas).sum())} linha(s) com valores em branco não puderam "
                       "ser avaliadas e aparecem sem resultado.")

        dados["probabilidade_risco"] = np.nan
        dados.loc[validas, "probabilidade_risco"] = modelo.predict_proba(X[validas])[:, 1]
        dados["faixa_risco"] = dados["probabilidade_risco"].apply(
            lambda p: classificar_faixa(p, CORTE_MEDIO, CORTE_ALTO) if pd.notna(p) else "—")

        def foco(r):
            baixos = [(i, r[i]) for i, lim in LIMIAR_ATENCAO.items()
                      if i in r and pd.notna(r[i]) and r[i] < lim]
            baixos.sort(key=lambda x: x[1])
            return ", ".join(i for i, _ in baixos) if baixos else "—"
        dados["foco_acao"] = dados.apply(foco, axis=1)

        c1, c2, c3 = st.columns(3)
        cont = dados["faixa_risco"].value_counts()
        c1.metric("Risco alto", int(cont.get("Alto", 0)))
        c2.metric("Risco médio", int(cont.get("Médio", 0)))
        c3.metric("Risco baixo", int(cont.get("Baixo", 0)))

        ordenado = dados.sort_values("probabilidade_risco", ascending=False, na_position="last")
        mostrar = [c for c in ["RA"] if c in ordenado.columns] + \
                  ["probabilidade_risco", "faixa_risco", "foco_acao"] + NUM + CAT
        st.dataframe(ordenado[mostrar].style.format({"probabilidade_risco": "{:.0%}"}),
                     use_container_width=True)
        st.download_button("Baixar resultado (CSV)",
                           ordenado.to_csv(index=False).encode("utf-8"),
                           "risco_turma.csv", "text/csv")

st.markdown("---")
st.caption("Modelo: Random Forest · alvo = agravamento de defasagem (T→T+1) · "
           "features do ano atual. Uso restrito a apoio à decisão da equipe Passos Mágicos.")
