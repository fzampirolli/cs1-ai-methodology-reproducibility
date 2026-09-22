#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise de percepção discente — questionários Likert 1-5 de PI 2026.1.

Entrada (dados/anonymous/, versionado -- já sem carimbo/nome/e-mail/texto
livre; ver analise/gen_dados_anonymous.py):
  - avaliacao_exp2_exp3.csv -> Experimentos 2 (Docente B / diurno) e
    3 (Docente C / noturno), com a turma/instrutor já resolvidos
  - avaliacao_exp1.csv      -> Experimento 1 (Docente A / noturno),
    exploratório

Saída:
  - dados/processados/percepcao_exp23.csv   (anonimizado: id sintético, sem e-mail)
  - dados/processados/percepcao_exp1.csv
  - analise/saidas/percepcao_exp23_itens.csv   (Wilcoxon 1-caudal, Holm, r, IC bootstrap)
  - analise/saidas/percepcao_homogeneidade_AxB.csv (Mann-Whitney U)
  - analise/saidas/percepcao_resumo.txt

Método (conforme rascunho): Shapiro-Wilk (normalidade); Wilcoxon signed-rank
unicaudal H1: mediana > 3; correção de Holm; tamanho de efeito = correlação
rank-biserial; IC 95% da mediana por bootstrap (n=5000); Mann-Whitney U
bicaudal + Holm para homogeneidade B x C (Exp. 2 x Exp. 3); alfa de Cronbach.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(20260901)
ROOT = Path(__file__).resolve().parents[1]
ANON = ROOT / "dados" / "anonymous"
PROC = ROOT / "dados" / "processados"
OUT = ROOT / "analise" / "saidas"
OUT.mkdir(parents=True, exist_ok=True)

F_23 = ANON / "avaliacao_exp2_exp3.csv"
F_1 = ANON / "avaliacao_exp1.csv"

MU0 = 3            # ponto neutro da escala Likert
NBOOT = 5000


def likert_cols(df: pd.DataFrame) -> list[str]:
    """Colunas de item Likert = têm '[' e ']' e domínio {1..5}."""
    out = []
    for c in df.columns:
        if "[" in c and "]" in c:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) and v.isin([1, 2, 3, 4, 5]).all():
                out.append(c)
    return out


def short_id(col: str) -> str:
    """'Bloco A: ... [07. enunciado ...]' -> 'Q07'."""
    import re
    m = re.search(r"\[\s*(\d{1,2})", col)
    return f"Q{int(m.group(1)):02d}" if m else col[:12]


def rank_biserial_signed(x: np.ndarray, mu: float = MU0) -> float:
    """r = (T+ - T-) / (T+ + T-) sobre as diferenças não nulas (x - mu)."""
    d = x - mu
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    tp = r[d > 0].sum()
    tn = r[d < 0].sum()
    return float((tp - tn) / (tp + tn))


def boot_median_ci(x: np.ndarray, nboot=NBOOT, alpha=0.05):
    x = np.asarray(x, float)
    bs = np.array([np.median(RNG.choice(x, x.size, replace=True)) for _ in range(nboot)])
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def cronbach_alpha(frame: pd.DataFrame) -> float:
    m = frame.dropna(axis=0, how="any").to_numpy(float)
    k = m.shape[1]
    if k < 2 or m.shape[0] < 2:
        return np.nan
    var_items = m.var(axis=0, ddof=1).sum()
    var_total = m.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - var_items / var_total))


def analyse_group(df: pd.DataFrame, items: list[str], label: str) -> pd.DataFrame:
    rows = []
    for c in items:
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)
        n = x.size
        sw_p = stats.shapiro(x).pvalue if 3 <= n <= 4999 and np.ptp(x) > 0 else np.nan
        # Wilcoxon signed-rank unicaudal H1: mediana > mu0
        try:
            w = stats.wilcoxon(x - MU0, alternative="greater", zero_method="wilcox")
            wstat, wp = float(w.statistic), float(w.pvalue)
        except ValueError:
            wstat, wp = np.nan, np.nan
        lo, hi = boot_median_ci(x)
        rows.append(dict(
            grupo=label, item=short_id(c), n=n,
            media=round(x.mean(), 3), mediana=float(np.median(x)),
            ic95_med_low=lo, ic95_med_high=hi,
            shapiro_p=sw_p, wilcoxon_stat=wstat, p_raw=wp,
            r_rank_biserial=round(rank_biserial_signed(x), 3),
            enunciado=c,
        ))
    res = pd.DataFrame(rows)
    ok = res["p_raw"].notna()
    res.loc[ok, "p_holm"] = multipletests(res.loc[ok, "p_raw"], method="holm")[1]
    res["signif_holm"] = res["p_holm"] < 0.05
    res["efeito"] = pd.cut(res["r_rank_biserial"].abs(), [-1, .1, .3, .5, 1],
                           labels=["desprezível", "pequeno", "médio", "grande"])
    return res


def main():
    # ---------- Experimentos 2 e 3 (Docentes B e C) ----------
    fr = pd.read_csv(F_23)
    items = likert_cols(fr)
    fr["turno"] = np.where(fr["instrutor"] == "B", "diurno", "noturno")
    fr_proc = fr[["instrutor", "turno"] + items].copy()
    fr_proc.insert(0, "id", [f"E23_{i:03d}" for i in range(1, len(fr_proc) + 1)])
    fr_proc.columns = ["id", "docente_cod", "turno"] + [short_id(c) for c in items]
    PROC.mkdir(parents=True, exist_ok=True)
    fr_proc.to_csv(PROC / "percepcao_exp23.csv", index=False)

    res23 = analyse_group(fr, items, "Exp2+3 (B+C)")
    res23.drop(columns=["enunciado"]).to_csv(OUT / "percepcao_exp23_itens.csv", index=False)

    # Homogeneidade B x C — Mann-Whitney U bicaudal + Holm
    homo = []
    for c in items:
        b = pd.to_numeric(fr.loc[fr.instrutor == "B", c], errors="coerce").dropna()
        cc = pd.to_numeric(fr.loc[fr.instrutor == "C", c], errors="coerce").dropna()
        u = stats.mannwhitneyu(b, cc, alternative="two-sided")
        homo.append(dict(item=short_id(c), n_B=len(b), n_C=len(cc),
                         mediana_B=float(b.median()), mediana_C=float(cc.median()),
                         U=float(u.statistic), p_raw=float(u.pvalue)))
    homo = pd.DataFrame(homo)
    homo["p_holm"] = multipletests(homo["p_raw"], method="holm")[1]
    homo["difere"] = homo["p_holm"] < 0.05
    homo.to_csv(OUT / "percepcao_homogeneidade_AxB.csv", index=False)

    # Cronbach (Q01..Q17 = exclui verificadores Q18..Q20)
    q_subst = [c for c in items if 1 <= int(short_id(c)[1:]) <= 17]
    alpha_all = cronbach_alpha(fr[q_subst].apply(pd.to_numeric, errors="coerce"))
    alpha_B = cronbach_alpha(fr.loc[fr.instrutor == "B", q_subst].apply(pd.to_numeric, errors="coerce"))
    alpha_C = cronbach_alpha(fr.loc[fr.instrutor == "C", q_subst].apply(pd.to_numeric, errors="coerce"))

    # ---------- Experimento 1 (Docente A) — exploratório ----------
    g = pd.read_csv(F_1)
    items_g = likert_cols(g)
    g_proc = g[items_g].copy()
    g_proc.insert(0, "id", [f"E1_{i:03d}" for i in range(1, len(g_proc) + 1)])
    g_proc.columns = ["id"] + [short_id(c) for c in items_g]
    g_proc.to_csv(PROC / "percepcao_exp1.csv", index=False)
    res1 = analyse_group(g, items_g, "Exp1 (A) [exploratório]")
    res1.drop(columns=["enunciado"]).to_csv(OUT / "percepcao_exp1_itens.csv", index=False)

    # ---------- resumo ----------
    L = []
    L.append("=== PERCEPÇÃO DISCENTE — PI 2026.1 ===\n")
    L.append(f"Exp. 2 (Docente B, diurno):  N = {(fr.instrutor=='B').sum()}")
    L.append(f"Exp. 3 (Docente C, noturno): N = {(fr.instrutor=='C').sum()}")
    L.append(f"Exp. 2+3 combinados:         N = {len(fr)}")
    L.append(f"Exp. 1 (Docente A, noturno): N = {len(g)}  (exploratório)\n")
    nviol = int((res23["shapiro_p"] < 0.05).sum())
    L.append(f"Normalidade (Shapiro-Wilk p<0,05): {nviol}/{res23['shapiro_p'].notna().sum()} itens violam\n")
    sig = res23[res23.signif_holm].sort_values("r_rank_biserial", ascending=False)
    L.append(f"Wilcoxon unicaudal (H1: mediana>3) + Holm: "
             f"{res23.signif_holm.sum()}/{len(res23)} itens significativos (p_holm<0,05)")
    L.append("\nItens significativos (ordenados por r):")
    for _, r in sig.iterrows():
        L.append(f"  {r['item']}: mediana={r['mediana']:.0f}  r={r['r_rank_biserial']:+.2f} "
                 f"({r['efeito']})  p_holm={r['p_holm']:.4f}")
    ns = res23[~res23.signif_holm]
    L.append("\nItens NÃO significativos após Holm:")
    for _, r in ns.iterrows():
        L.append(f"  {r['item']}: mediana={r['mediana']:.0f}  r={r['r_rank_biserial']:+.2f}  "
                 f"p_holm={r['p_holm']:.4f}  p_raw={r['p_raw']:.4f}")
    L.append(f"\nHomogeneidade B x C (Mann-Whitney U + Holm): "
             f"{int(homo.difere.sum())}/{len(homo)} itens diferem  "
             f"-> {'agrupamento legítimo' if homo.difere.sum()==0 else 'ATENÇÃO: há diferença'}")
    L.append(f"\nAlfa de Cronbach (Q01-Q17):  B+C={alpha_all:.3f}   B={alpha_B:.3f}   C={alpha_C:.3f}")
    txt = "\n".join(L)
    (OUT / "percepcao_resumo.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
