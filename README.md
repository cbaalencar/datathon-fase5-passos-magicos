# Datathon Fase 5 — Associação Passos Mágicos

Modelo preditivo de risco de defasagem escolar para a Associação Passos Mágicos, ONG que atende crianças e jovens em vulnerabilidade social em Embu-Guaçu (SP). Pós-Tech FIAP — Data Analytics.

## Objetivo

Estimar a probabilidade de um aluno **agravar sua defasagem escolar no próximo ciclo**, a partir dos indicadores do ano atual, permitindo que a equipe pedagógica priorize acompanhamento **antes** da piora acontecer.

## App

🔗 **https://pede-risco-defasagem.streamlit.app**

Duas formas de uso: pontuação individual (entradas com os indicadores do aluno) e em lote (upload do CSV da turma, com ranking de risco e foco de ação por aluno).

## Estrutura do repositório

```
data/               três camadas da base + arquivos de override
                    · painel_original.csv  — harmonizada, sem nenhuma correção
                    · painel_limpo.csv     — corrigida, com colunas _base de auditoria
                    · painel_modelo.csv    — recorte apto à modelagem
                    · nasc_override.csv / fase_override.csv — correções manuais auditáveis
notebooks/          01 limpeza e tratamento · 02 modelo preditivo
models/             modelo treinado (.pkl) e metadados (.json)
app/                aplicação Streamlit
requirements.txt    dependências com versões fixas
```

## Tratamento dos dados

Base PEDE 2022–2024 — 3.030 linhas no painel completo, **2.566 na base de modelagem**. O notebook 01 documenta cada decisão em um painel de parâmetros; as principais:

- **Idade padronizada** como `Ano letivo − ano de nascimento`, com 4 nascimentos corrigidos manualmente por triangulação fase/idade/defasagem (`nasc_override.csv`).
- **Fase Ideal recalculada** pela Tabela 4 do documento PEDE (divergia da base em 16,6% das linhas) e **cascata propagada**: Fase Ideal → Defasagem → IAN (Tabela 41, 100% de conformidade) → INDE → Pedra. As colunas originais ficam preservadas com sufixo `_base`.
- **Fase efetiva**: 19 registros além da fase ideal da própria idade (validados um a um) corrigidos por `fase do ano anterior + 1` — 21 correções com a propagação entre anos. Os demais movimentos atípicos (25 saltos em direção à fase da idade e 1 regressão) saem da base de modelo por decisão conservadora, já que nenhuma fonte documenta a mecânica de progressão — flag de reversão previsto.
- **Pisos do IAA por faixa de fase** (Tabela 40 + Figura 10: 3,5 nas Fases 0–2; 2,5 nas 3–8): 251 valores abaixo do piso. Para cada um, o INDE original é testado contra a fórmula documental *usando o próprio valor inválido*: 208 contaminações provadas e 43 indetermináveis — IAA, INDE e Pedra dessas linhas são anulados (originais preservados em `_base`).
- **Reconstrução do IPP 2022 (817 casos) e recuperação algébrica ancoradas no IAN original** — o INDE da base foi calculado pela ONG com a defasagem antiga, então a engenharia reversa usa esse mesmo IAN. A recuperação de índice isolado agora exige valor **acima do piso do indicador**: 3 recuperações exatas; tentativas que devolveriam valores abaixo do piso (ressuscitando o dado inválido) são rejeitadas.
- **Pedra pelos cortes exatos** das Figuras 2–3 do documento (6,110 / 7,154 / 8,198): 803 reclassificações e 287 anulações (INDE não recalculável).
- **Zero imputação estatística.** O que não é real nem algebricamente exato é excluído, não estimado.

## Modelo

**Alvo temporal:** o aluno agrava a defasagem do ano T para o T+1 (`Defasagem_T+1 < Defasagem_T`). Qualquer piora conta, inclusive dentro da mesma faixa do IAN. As features são do ano T — não há vazamento de informação futura.

Features: Defasagem, fase, IDA, IEG, IAA, IPS, IPV, IPP, gênero e rede de ensino. O IAN fica fora por ser função-degrau da própria defasagem (vazamento); INDE e idade saem por redundância e colinearidade. O IPP entra como feature por decisão documentada (reconstrução algébrica exata), com a sensibilidade com/sem registrada no notebook 02.

**Avaliação em dois regimes** — a validação por aluno é a métrica principal (usa todos os anos, sem nenhum estudante em treino e teste ao mesmo tempo); o corte temporal é a prova de realismo (treina 2022→23, testa 2023→24, um ano nunca visto):

| | Amostra | ROC-AUC | Acurácia | Balanced accuracy |
|---|---|---|---|---|
| **Principal — por aluno** | 960 transições · 632 alunos · CV 5-fold | 0,867 ± 0,013 | 0,811 | 0,771 |
| **Robustez — temporal** | treino 426 · teste 534 | 0,792 | 0,811 | 0,619 |

Limiar de decisão 0,30; baseline trivial ("nunca agrava") ≈ 0,78–0,80, por isso a balanced accuracy acompanha sempre a acurácia. No regime temporal, um limiar recalibrado apenas com o treino (0,35) não supera o principal (bal. 0,581 vs 0,619) — evidência de que a diferença entre regimes é *drift* real entre anos, não ajuste de corte; a seção 11 do notebook 02 mede e discute exatamente isso.

### Faixas de risco

Cortes ancorados na taxa média de agravamento da base (21,1%):

| Faixa | Alunos | Agravamento real | % dos casos captados |
|---|---|---|---|
| Baixo (< 0,211) | 628 | 6% | 19% |
| Médio | 189 | 38% | 36% |
| Alto (≥ 0,423) | 143 | 64% | 45% |

## Como reproduzir

1. `notebooks/01_limpeza_base_PEDE.ipynb` — gera as três camadas em `data/` (no Colab, a partir do XLSX no Drive; fora dele, do diretório local).
2. `notebooks/02_modelo_risco_defasagem.ipynb` — treina, avalia nos dois regimes e salva `models/modelo_risco_defasagem.pkl` + `modelo_meta.json`.
3. `app/app.py` — Streamlit Community Cloud, com `requirements.txt` na raiz (scikit-learn fixado em 1.6.1, mesma versão do treino).
