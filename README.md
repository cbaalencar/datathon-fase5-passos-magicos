# Datathon Fase 5 — Associação Passos Mágicos

Modelo preditivo de risco de defasagem escolar para a Associação Passos Mágicos, ONG que atende crianças e jovens em vulnerabilidade social em Embu-Guaçu (SP). Pós-Tech FIAP — Data Analytics.

## Objetivo

Estimar a probabilidade de um aluno **agravar sua defasagem escolar no próximo ciclo**, a partir dos indicadores do ano atual, permitindo que a equipe pedagógica priorize acompanhamento **antes** da piora acontecer.

## App

🔗 **https://pede-risco-defasagem.streamlit.app**

Duas formas de uso: pontuação individual (sliders com os indicadores do aluno) e em lote (upload do CSV da turma, com ranking de risco e foco de ação por aluno).

## Estrutura do repositório

```
data/               bases tratadas (painel_limpo, painel_modelo, nasc_override)
notebooks/          01 limpeza e tratamento · 02 modelo preditivo
models/             modelo treinado (.pkl) e metadados (.json)
app/                aplicação Streamlit
requirements.txt    dependências com versões fixas
```

## Tratamento dos dados

Base PEDE 2022–2024 — 3.030 linhas no painel completo, 2.747 na base de modelagem (46 colunas). O notebook 01 documenta cada decisão. As principais:

- **Idade padronizada** como `Ano letivo − ano de nascimento`, com 4 nascimentos corrigidos manualmente por triangulação fase/idade/defasagem.
- **Fase Ideal recalculada** pela Tabela 4 do documento PEDE — divergia da base em 16,6% das linhas, por convenção de idade e por régua de fronteira.
- **Cascata propagada:** Fase Ideal → Defasagem → IAN (Tabela 41) → INDE → Pedra. As colunas originais ficam preservadas com sufixo `_base` para auditoria.
- **Reconstrução do IPP 2022 e recuperação algébrica ancoradas no IAN original** — o INDE da base foi calculado pela ONG com a defasagem antiga, então a engenharia reversa precisa usar esse mesmo IAN. Sem isso o IPP reconstruído absorvia erro de até 7,5 pontos.
- **Zero imputação estatística.** Índices ausentes só são recuperados quando a fórmula do INDE permite solução exata (1 equação, 1 incógnita) — 152 casos. O que não fecha é excluído, não estimado.

## Modelo

**Alvo temporal:** o aluno agrava a defasagem do ano T para o T+1 (`Defasagem_T+1 < Defasagem_T`). Qualquer piora conta, inclusive dentro da mesma faixa do IAN. As features são do ano T — não há vazamento de informação futura.

| | |
|---|---|
| Amostra | 1.155 transições · 759 alunos |
| Algoritmo | Random Forest (Pipeline scikit-learn) |
| ROC-AUC | 0,868 ± 0,016 (validação cruzada agrupada por aluno) |
| Acurácia | 0,814 |
| Balanced accuracy | 0,730 |

Features: Defasagem, fase, IDA, IEG, IAA, IPS, IPV, IPP, gênero e rede de ensino. O IAN fica fora por ser função-degrau da própria defasagem (vazamento); INDE e idade saem por redundância e colinearidade.

A validação é agrupada por aluno (`GroupKFold`), garantindo que nenhum estudante apareça em treino e teste ao mesmo tempo. A acurácia é reportada junto da balanced accuracy porque a base é desbalanceada (21,4% de agravamento) — o baseline trivial já acertaria 78,6%.

### Faixas de risco

Cortes ancorados na taxa média de agravamento da base:

| Faixa | Alunos | Agravamento real | % dos casos |
|---|---|---|---|
| Baixo | 724 | 5% | 15% |
| Médio | 243 | 37% | 36% |
| Alto | 188 | 64% | 49% |

O grupo Alto concentra metade dos casos reais em 188 alunos — taxa 13× maior que a do grupo Baixo. Médio e Alto somados cobrem 85% dos agravamentos.

## Como executar

**Notebooks:** abrir no Google Colab, montar o Drive e ajustar `PASTA_DRIVE` na célula de configuração. Rodar o 01 antes do 02.

**App localmente:**

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

O `requirements.txt` fixa `scikit-learn==1.6.1`, a mesma versão que gerou o `.pkl`. Alterar sem regerar o modelo quebra o carregamento.

## Observações

Ferramenta de apoio à priorização, não de decisão automática — a avaliação da equipe pedagógica e psicossocial é soberana.

Parte do poder preditivo vem da mecânica estrutural da defasagem (fase e idade determinam boa parte da progressão). Os indicadores comportamentais e psicossociais agregam +0,05 de AUC sobre essa base estrutural: é o sinal acionável, aquele sobre o qual a ONG consegue intervir.

---

Caroline Alencar · Pós-Tech FIAP Data Analytics · 2026
