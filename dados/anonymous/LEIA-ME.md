# dados/anonymous/

Cópias anonimizadas de `dados/brutos/` (não versionado — contém e-mail/nome,
LGPD), geradas por `analise/gen_dados_anonymous.py`. Rodar esse script de
novo sempre que `dados/brutos/` mudar.

O que foi removido, e por quê: carimbo de data/hora, nome completo, e-mail —
identificadores diretos — e **todo campo de texto livre não usado por
nenhum `check(...)` da análise** (sugestões de melhoria, expectativas de
aprendizagem, etc.), que é onde estaria o maior risco residual de
reidentificação. Mantidas só as colunas efetivamente lidas pelo pipeline:
itens Likert (1–5), vínculo empregatício, horas de estudo, ano de ingresso,
linguagens já usadas, Bacharelado Interdisciplinar, turma.

A coluna livre "Qual é a sua turma" do questionário de avaliação (que trazia
o código da turma seguido do nome do docente, ex. `"<código> - <nome>"`)
foi resolvida em duas colunas limpas: `turma` e `instrutor` (`B`/`C`, sem
nome real).

**Código de turma pseudonimizado** (2026-09-21), nos 4 arquivos: o código
real da turma (código curto de sala, diferente do código completo
`disciplina-matriz-campus` da planilha institucional) dá pra cruzar com o
sistema público de turmas da instituição e identificar o docente de
2026.1 — mesmo risco e mesmo tratamento de `dados/serie-historica/
pi0505_turmas_2009-2026.csv`. Trocado pelos mesmos pseudônimos de lá,
mapeados em `gen_dados_anonymous.py`: `A1`/`A2` (Exp. 1), `B1` (Exp. 2),
`C1`/`C2` (Exp. 3). (No pacote público de reprodutibilidade, esse script
não é distribuído — só o CSV já com os pseudônimos, pelo mesmo motivo que
`00_serie_historica_extrair.py` também não é: o script em si cita os
códigos reais.)

| arquivo | conteúdo |
|---|---|
| `perfil_diurno.csv` | perfil, Docente B / diurno, Exp. 2 |
| `perfil_noturno.csv` | perfil, Docente A / noturno, Exp. 1 |
| `avaliacao_exp2_exp3.csv` | Likert 1–5, Exp. 2+3, `turma`+`instrutor` já resolvidos |
| `avaliacao_exp1.csv` | Likert 1–5, Exp. 1 |

Usado por `analise/artigo_calculos.ipynb` / `artigo_calculos_en.ipynb`,
`analise/01_percepcao_questionarios.py` e `analise/03_figuras.py` /
`03_figuras_en.py` — nenhum desses lê mais `dados/brutos/` diretamente.

Versionado (exceção em `.gitignore`: `!dados/anonymous/*.csv`), ao contrário
de `dados/brutos/`.
