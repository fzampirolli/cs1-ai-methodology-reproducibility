# Reproducibility package — Active Learning with AI Support in CS1

Code and derived data that reproduce every statistic, table, and figure in a
manuscript currently under double-blind review at *Informatics in Education*
(EJMS). Shared here so editors and reviewers can inspect the analysis without
identifying the authors — see "What is withheld, and why" below.

## What this contains

- `analise/artigo_calculos.ipynb` — a single, already-executed Jupyter
  notebook, in English (matching the manuscript's language), that walks
  from the de-identified data through every statistic, table, and figure
  in the manuscript: descriptive counts, Shapiro-Wilk, one-sided Wilcoxon
  signed-rank tests with Holm correction, rank-biserial effect sizes,
  bootstrap CIs, Mann-Whitney homogeneity checks, Cronbach's alpha, and the
  historical failure-rate series by class shift (weighted binomial GLM,
  cluster-robust SEs). Figures 3, 4, and 5 are rendered by calling the
  plotting functions in `03_figuras_en.py` directly, so they are the same
  images published in the manuscript, not simplified stand-ins. Each
  section ends with a note on where that number appears in the manuscript
  (section, table, or figure). All `check(...)` calls in the notebook
  passed against the real data before publication. **Part 1 (the
  historical series) runs end to end from the files in this repository
  alone** — Parts 2-3 (student perception) need the raw questionnaire
  exports, which are not included (see below).
- `analise/02_serie_historica_turno.py`, `03_figuras.py` / `03_figuras_en.py`
  — the standalone scripts (in Portuguese, the authors' working language)
  behind the historical class-shift series and its two figures (PT/EN
  captions). These read only the already de-identified CSVs in this
  repository.
- `analise/01_percepcao_questionarios.py` — the standalone script (also in
  Portuguese) for the student-perception analysis (Wilcoxon/Holm, effect
  sizes, Cronbach's alpha). Reads from `dados/anonymous/` (included).
- `analise/saidas/` — the numeric outputs these scripts produce (CSV/TXT),
  already computed from the actual data.
- `dados/serie-historica/pi0505_turmas_2009-2026.csv` — the class-level
  grade record for the course studied, 2009.3-2026.2, 840 classes
  (pass/fail counts per class, shift, campus, on-schedule-term flag; no
  student-level data). `dados/serie-historica/pi0505_agregado_quadrimestre_turno_2009-2026.csv`
  — the same record aggregated by term and class shift, the basis of the
  manuscript's Table 6 and Figure 3.
- `dados/anonymous/` — de-identified copies of the four questionnaire
  exports used in Parts 2-3 (see its own `LEIA-ME.md`).

## What is withheld, and why

- **Raw questionnaire exports** (Google Forms) are not included: they
  carry e-mail addresses and full names collected as optional fields,
  personal data under Brazil's LGPD. `dados/anonymous/` has de-identified
  copies with only the columns the analysis actually uses (see its
  `LEIA-ME.md`); the notebook and scripts read those, not the raw files,
  and additionally check that no PII column made it through
  (`drop_pii`/`is_pii`). The raw exports are available from the
  corresponding author on reasonable request during review, for
  verification purposes only.
- **Instructor identity.** The manuscript is under double-blind review, so
  the three instructors are referred to as Instructor A/B/C throughout, as
  in the manuscript itself (Instructor A = Experiment 1, evening;
  Instructor B = Experiment 2, daytime; Instructor C = Experiment 3,
  evening). Nothing in this repository ties that letter to a real name.
- **2026.1 class codes.** The institutional class code (e.g.
  `NA1BCM0505-22SB`) is reused every term as a sequential slot number, not
  a name in itself — but for 2026.1 specifically, it can be cross-checked
  against the institution's public class schedule to identify the
  instructor, which would break double-blind review. So
  `dados/serie-historica/pi0505_turmas_2009-2026.csv`'s `turma` column is
  blank for all 840 classes **except** the three 2026.1 classes the
  manuscript's own Table 5 already discusses by number, pseudonymised to
  match: `B1` (Experiment 2), `C1`/`C2` (Experiment 3). Experiment 1's two
  classes are not in this institutional table at all — the manuscript
  sources their numbers from the instructor's own consolidated grade
  sheet instead (see Table 5's note), so there was nothing to redact or
  label for them here. `shift` and `campus` were derived from the real
  code *before* it was redacted, so no number in the series changed — only
  the class label was removed. The two scripts that do this redaction
  (`00_serie_historica_extrair.py`, which also holds the 3 real codes as a
  lookup, and `gen_dados_anonymous.py`, which references the raw
  questionnaire filenames) are intentionally **not** included in this
  repository; only their already-redacted output is.

## Note on `analise/saidas/percepcao_homogeneidade_AxB.csv`

The internal group labels in this one file are the script's own working
codes (`A` = Experiment 2 / Instructor B / daytime, `B` = Experiment 3 /
Instructor C / evening) and predate the Instructor-A/B/C convention used
elsewhere in this package and in the manuscript. No personal data is
involved — it is a labelling mismatch only.

## Running it

```
pip install numpy pandas scipy statsmodels matplotlib jupyter openpyxl
jupyter notebook analise/artigo_calculos.ipynb
```

Part 1 (historical series) and Part 3's figures run from the files
included here. Parts 2-3's statistics need the raw questionnaire exports
described above; without them those specific cells will not execute, but
their already-computed outputs remain visible in the notebook as
published.

## License

Code: MIT. Data: released for reproducibility review; full terms to be
confirmed on publication.
