# Reproducibility package — AI-supported practice ecosystem in one CS1 class

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fzampirolli/cs1-ai-methodology-reproducibility/blob/main/calculations.ipynb)

De-identified data and a single notebook that reproduce **every number,
table and figure** of a manuscript on one CS1 class that adopted a
four-layer AI-supported practice ecosystem (AI-curated material, guided
practice in Colab notebooks, local test validation, parametrised exams with
automated and LLM-proposed grading).

## Contents

| path | what it is |
|---|---|
| `calculations.ipynb` | the notebook, in English, already executed. It walks from the files in `data/` to each published result. Every result is followed by `check(...)`, which compares it with the value printed in the manuscript and stops if they differ; all 205 checks pass. Each part ends with a note on where the numbers appear in the manuscript. |
| `data/perception.csv` | 20-item Likert questionnaire (1–5), anonymous. `group = analysed` (the class studied, N = 34) or `comparison` (two classes of another campus that used the same material with a different assessment design, N = 40). Q01–Q17 substantive, Q18–Q20 validity checks. |
| `data/profile.csv` | profile questionnaire, anonymous: work status, daily hours available for study, programming languages used before, entry programme and year. `group = analysed` or `evening_campus1` (two evening classes of the same campus, used only for the work/study-time proportions). |
| `data/exams.csv` | exam and mock-exam scores (0–100) of the analysed class, one row per student, random IDs `E01…E41`. Columns: `p*_vpl` automatic VPL grade (proportional to the test cases passed); `p1_vpl_reviewed` Exam 1 grade after the instructor's review inside the VPL; `p2_deepseek`, `p3_deepseek_run1…5`, `p3_gemini` LLM-proposed grades (run 5 was sent to students); `p*_final` final grade in the gradebook; `p3_instructor_sheet` the instructor's working column; `mock1…3` mock exams (empty = not taken); `p1_submissions`; `past_exams_practice`. |
| `data/exam1_llm_models.csv` | Exam 1 only: how many graded questions each Groq model handled (the grading tool tried a fixed list of models and fell back to the next when one failed, e.g. at the free plan's rate limit) — counts only. |
| `data/llm_usage.csv` | the LLM provider's usage export (tokens, requests, month-to-date cost) right before and after grading runs 4 and 5; account and key identifiers removed. |
| `exam3/` | material of the LLM grading of Exam 3 (RQ3): `prompt_exam3.txt`, the single prompt sent to the model, which first identifies the question type and then applies that type's rubric (criteria and points for both types are inside it); `statement_variation1.txt`, one of the 50 exam variations (Type A: element-wise matrix operation; Type B: grades and letter grades); `example_llm_output.txt`, one output of the run sent to students, for a Type B answer to which the test-case grading gave 0 points (student code omitted) — the example shown in the manuscript's table `tab:llm_exemplo`. The grading tool itself is open source (link withheld for double-blind review). |
| `data/class_series.csv` | final-grade counts per class of the course, 2009–2026 (840 classes; no student-level data). `on_schedule_term` = the term with the most classes in each year; `instructor_prior` = earlier offerings of the analysed class's instructor; `class_label` = `B1` (analysed class), `C1`/`C2` (comparison classes), empty otherwise. Campuses are pseudonymised as `1` and `2`. |

## How the data were de-identified

- **No direct identifiers.** Names, e-mails, logins, enrolment numbers,
  submitted code, timestamps and free-text answers are not in any file. The
  notebook's first cell also asserts that no such column is present.
- **Questionnaires** were anonymous at collection; the two instruments
  cannot be linked to each other or to exam scores.
- **Exam scores** are keyed by an ID drawn at random when the file was
  built; the order of the rows does not follow the class list, and the key
  to real students is not kept with this package.
- **Class codes** are removed: an institutional class code can be matched
  against public timetables to find the instructor. Only the three labels
  used in the manuscript remain.
- **LLM usage exports** keep only aggregate token, request and cost values.

## Running

```bash
pip install numpy pandas scipy statsmodels matplotlib jupyter
jupyter nbconvert --to notebook --execute calculations.ipynb --output out.ipynb
```

or open it in Colab with the badge above: when `data/` is not next to the
notebook, the first cell downloads the CSVs from this repository. Tested
with Python 3.10 (numpy 2.2, scipy 1.15, pandas 2.3) and Python 3.12 (numpy
2.5, scipy 1.18, pandas 3.0), statsmodels 0.15; all checks pass in both.

Bootstrap intervals and the power simulation use a single seeded random
generator consumed in a fixed order; run the cells top to bottom to get the
published second decimal.

## License

Code: MIT. Data: CC BY 4.0.
