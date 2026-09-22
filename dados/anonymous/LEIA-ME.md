# dados/anonymous/

De-identified copies of the raw questionnaire exports (not included in this
repository — they carry e-mail/name, personal data under Brazil's LGPD).

What was removed, and why: timestamp, full name, e-mail — direct
identifiers — and **every free-text field not used by any `check(...)` in
the analysis** (improvement suggestions, learning expectations, etc.),
which is where the largest residual re-identification risk would be. Only
the columns actually read by the pipeline are kept: Likert items (1-5),
employment status, study hours, entry year, prior programming languages,
degree programme, class.

The free-text "which class are you in" field of the perception
questionnaire (which carried the instructor's first name, e.g.
`"<class code> - <instructor's first name>"`) was resolved into two clean
columns: `turma` (just the class code) and `instrutor` (`B`/`C`, no real
name).

| file | content |
|---|---|
| `perfil_diurno.csv` | profile, Instructor B / daytime, Exp. 2 |
| `perfil_noturno.csv` | profile, Instructor A / evening, Exp. 1 |
| `avaliacao_exp2_exp3.csv` | Likert 1-5, Exp. 2+3, `turma`+`instrutor` already resolved |
| `avaliacao_exp1.csv` | Likert 1-5, Exp. 1 |

Used by `analise/artigo_calculos.ipynb` and `analise/01_percepcao_questionarios.py`
/ `analise/03_figuras.py` / `03_figuras_en.py` — none of those read the raw
questionnaire exports directly.
