#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Série histórica de reprovação em PI (PI0505) — efeito de TURNO (diurno x noturno).

Entrada: dados/serie-historica/pi0505_turmas_2009-2026.csv  (840 turmas, 2009.3-2026.2)
Saída:   analise/saidas/serie_turno_resumo.txt
         analise/saidas/serie_turno_glm.csv

PROVISÓRIO: sem `docente_id` (ainda não disponível), a decomposição de
variância docente/seção NÃO é feita aqui — o modelo usa o quadrimestre como
bloco (efeito fixo) e erro-padrão robusto por quadrimestre. O campus (SA/SBC),
agora disponível para ~94% das turmas, entra apenas como covariável de
robustez, não como comparação.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dados" / "serie-historica" / "pi0505_turmas_2009-2026.csv"
OUT = ROOT / "analise" / "saidas"
OUT.mkdir(parents=True, exist_ok=True)

COVID = {f"{y}.{q}" for y in (2020, 2021, 2022) for q in (1, 2, 3)}
COVID = {q for q in COVID if "2020.1" <= q <= "2022.2"}


def main():
    d = pd.read_csv(CSV, dtype={"quadrimestre": str})
    d = d[d["n_avaliados"] > 0].copy()
    d["ano"] = d["quadrimestre"].str[:4].astype(int)
    d["janela"] = np.where(d["quadrimestre"].isin(COVID), "covid",
                    np.where(d["quadrimestre"] < "2020.1", "pre", "pos"))
    d["p_reprov"] = d["n_reprovados"] / d["n_avaliados"]

    L = []
    L.append("=== SÉRIE HISTÓRICA PI0505 — EFEITO DE TURNO (provisório, sem docente) ===\n")
    L.append(f"{len(d)} turmas | {d.quadrimestre.min()}–{d.quadrimestre.max()} | "
             f"{int(d.n_avaliados.sum())} avaliações\n")

    # 1) pooled global
    for t, g in d.groupby("turno"):
        L.append(f"  {t:8}: {len(g):3d} turmas | reprov pooled = "
                 f"{g.n_reprovados.sum()/g.n_avaliados.sum():.4f} | "
                 f"média das taxas por turma = {g.p_reprov.mean():.4f} (dp {g.p_reprov.std():.4f})")

    # 2) teste pareado por quadrimestre (bloco = quadrimestre; robusto)
    piv = (d.groupby(["quadrimestre", "turno"])
             .apply(lambda x: x.n_reprovados.sum() / x.n_avaliados.sum())
             .unstack("turno"))
    piv = piv.dropna()
    diff = piv["noturno"] - piv["diurno"]
    w = stats.wilcoxon(diff, alternative="greater")
    sgn = stats.binomtest((diff > 0).sum(), len(diff), 0.5, alternative="greater")
    L.append(f"\n  Quadrimestres com diurno E noturno: {len(piv)}")
    L.append(f"  noturno > diurno em {(diff>0).sum()}/{len(diff)} quadrimestres "
             f"(Δ mediano = {diff.median()*100:+.1f} p.p.; média {diff.mean()*100:+.1f} p.p.)")
    L.append(f"  Wilcoxon pareado (H1: noturno>diurno): W={w.statistic:.1f}  p={w.pvalue:.2e}")
    L.append(f"  Teste de sinais: p={sgn.pvalue:.2e}")

    # 3) GLM binomial ponderado: reprov ~ turno + C(ano) + janela
    d["succ"] = d["n_reprovados"].astype(int)
    d["fail"] = (d["n_avaliados"] - d["n_reprovados"]).astype(int)
    m = smf.glm("succ + fail ~ C(turno, Treatment('diurno')) + C(ano) + C(janela, Treatment('pre'))",
                data=d, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": d["quadrimestre"]})
    par = m.params.filter(like="turno")
    ci = m.conf_int().loc[par.index]
    or_ = np.exp(par.iloc[0]); lo, hi = np.exp(ci.iloc[0])
    p = m.pvalues[par.index].iloc[0]
    L.append(f"\n  GLM binomial ponderado (bloco=ano, janela COVID; EP robusto por quadrimestre):")
    L.append(f"    OR noturno vs diurno = {or_:.3f}  IC95% [{lo:.3f}; {hi:.3f}]  p = {p:.2e}")
    L.append(f"    -> chance de reprovar {'+' if or_>1 else '-'}{abs(or_-1)*100:.0f}% no noturno, "
             f"ajustado por ano e janela COVID")

    # 3b) robustez: mesmo GLM + campus como covariável (SA/SBC); turmas antigas
    #     sem campus ficam de fora deste ajuste. NÃO é comparação entre campi.
    dc = d[d["campus"].isin(["SA", "SBC"])].copy()
    mc = smf.glm("succ + fail ~ C(turno, Treatment('diurno')) + C(ano) "
                 "+ C(janela, Treatment('pre')) + C(campus, Treatment('SA'))",
                 data=dc, family=sm.families.Binomial()).fit(
                 cov_type="cluster", cov_kwds={"groups": dc["quadrimestre"]})
    pc = mc.params.filter(like="turno")
    cic = mc.conf_int().loc[pc.index]
    orc = np.exp(pc.iloc[0]); loc_, hic = np.exp(cic.iloc[0])
    L.append(f"\n  Robustez — GLM com campus (SA/SBC) como covariável "
             f"({len(dc)} turmas com campus):")
    L.append(f"    OR noturno vs diurno = {orc:.3f}  IC95% [{loc_:.3f}; {hic:.3f}]  "
             f"p = {mc.pvalues[pc.index].iloc[0]:.2e}")
    for cp, gp in dc.groupby("campus"):
        L.append(f"    {cp:4}: reprov pooled diurno "
                 f"{gp[gp.turno=='diurno'].n_reprovados.sum()/gp[gp.turno=='diurno'].n_avaliados.sum():.3f}"
                 f" | noturno "
                 f"{gp[gp.turno=='noturno'].n_reprovados.sum()/gp[gp.turno=='noturno'].n_avaliados.sum():.3f}"
                 f"  ({len(gp)} turmas)")

    # 3c) robustez: GLM + período ideal como covariável
    mp = smf.glm("succ + fail ~ C(turno, Treatment('diurno')) + C(ano) "
                 "+ C(janela, Treatment('pre')) + periodo_ideal",
                 data=d, family=sm.families.Binomial()).fit(
                 cov_type="cluster", cov_kwds={"groups": d["quadrimestre"]})
    pp = mp.params.filter(like="turno")
    cip = mp.conf_int().loc[pp.index]
    orp = np.exp(pp.iloc[0]); lop, hip = np.exp(cip.iloc[0])
    or_pi = np.exp(mp.params["periodo_ideal"])
    L.append(f"\n  Robustez — GLM com período ideal como covariável:")
    L.append(f"    OR noturno vs diurno = {orp:.3f}  IC95% [{lop:.3f}; {hip:.3f}]  "
             f"p = {mp.pvalues[pp.index].iloc[0]:.2e}")
    L.append(f"    OR período ideal vs não-ideal = {or_pi:.3f} "
             f"(reprovação menor no período ideal)")

    # 4) Scheirer–Ray–Hare: turno x janela (não paramétrico, 2 fatores)
    srh = scheirer_ray_hare(d, "p_reprov", "turno", "janela")
    L.append("\n  Scheirer–Ray–Hare (turno × janela COVID) sobre a taxa por turma:")
    for k, v in srh.items():
        L.append(f"    {k:16} H={v['H']:.2f}  df={v['df']}  p={v['p']:.2e}")

    # 5) por janela
    L.append("\n  Reprovação pooled por janela e turno:")
    for j in ["pre", "covid", "pos"]:
        sub = d[d.janela == j]
        for t in ["diurno", "noturno"]:
            s = sub[sub.turno == t]
            if len(s):
                L.append(f"    {j:5} {t:8}: {s.n_reprovados.sum()/s.n_avaliados.sum():.4f}  ({len(s)} turmas)")

    # 5b) período ideal (quadrimestre modal do ano) × turno
    def pool(x):
        return x.n_reprovados.sum() / x.n_avaliados.sum()
    L.append("\n  Reprovação pooled por período ideal e turno "
             "(ideal = quadrimestre com mais turmas no ano):")
    for pi, lab in [(1, "ideal"), (0, "não-ideal")]:
        g = d[d.periodo_ideal == pi]
        L.append(f"    {lab:9} TOTAL   : {pool(g):.4f}  ({len(g)} turmas)")
        for t in ["diurno", "noturno"]:
            gt = g[g.turno == t]
            L.append(f"    {lab:9} {t:8}: {pool(gt):.4f}  ({len(gt)} turmas)")
    di = d[d.periodo_ideal == 1]
    L.append(f"    -> período ideal: gap noturno-diurno = "
             f"{(pool(di[di.turno=='noturno']) - pool(di[di.turno=='diurno']))*100:+.1f} p.p.")

    # 5c) campus (SA/SBC) × turno — cursos pós-BI distintos entre campi podem
    #     confundir a taxa; verifica-se se o efeito de turno resiste dentro
    #     de cada campus. NÃO é comparação entre campi.
    L.append("\n  Reprovação pooled por campus e turno "
             "(cursos pós-BI distintos entre campi; checagem de confusão):")
    for cp in ["SA", "SBC"]:
        g = d[d.campus == cp]
        rd, rn = pool(g[g.turno == "diurno"]), pool(g[g.turno == "noturno"])
        L.append(f"    {cp:4}: diurno {rd:.4f} ({len(g[g.turno=='diurno'])} t) | "
                 f"noturno {rn:.4f} ({len(g[g.turno=='noturno'])} t) | "
                 f"gap {(rn-rd)*100:+.1f} p.p.")
    sem_campus = d["campus"].isna().sum() + (d["campus"] == "").sum()
    L.append(f"    ({sem_campus} turmas de 2009-2010 sem sufixo de campus, fora deste recorte)")

    # 5d) baseline por experimento: período ideal × campus × turno
    #     (Exp2 = SA/diurno; Exp1 = SA/noturno; Exp3 = SBC/noturno)
    L.append("\n  Baseline histórico por experimento (período ideal × campus × turno):")
    for cp in ["SA", "SBC"]:
        for t in ["diurno", "noturno"]:
            g = d[(d.periodo_ideal == 1) & (d.campus == cp) & (d.turno == t)]
            if len(g):
                L.append(f"    {cp:4} {t:8}: pooled {pool(g):.4f} | "
                         f"média turmas {g.p_reprov.mean():.4f} | {len(g)} turmas")

    # 6) tabela por ANO × turno (pooled) — linhas da Tabela do artigo
    d["ano"] = d["quadrimestre"].str[:4].astype(int)
    L.append("\n  Reprovação pooled por ANO e turno (para a Tabela do artigo):")
    L.append(f"    {'ano':4} {'diurno':>7} {'noturno':>8} {'d_pp':>6} {'turmas':>7}")
    for ano, g in d.groupby("ano"):
        gd, gn = g[g.turno == "diurno"], g[g.turno == "noturno"]
        rd = gd.n_reprovados.sum() / gd.n_avaliados.sum() if len(gd) else float("nan")
        rn = gn.n_reprovados.sum() / gn.n_avaliados.sum() if len(gn) else float("nan")
        L.append(f"    {ano:4d} {rd:7.3f} {rn:8.3f} {(rn-rd)*100:+6.1f} {len(g):7d}")
    gd, gn = d[d.turno == "diurno"], d[d.turno == "noturno"]
    rd = gd.n_reprovados.sum() / gd.n_avaliados.sum()
    rn = gn.n_reprovados.sum() / gn.n_avaliados.sum()
    L.append(f"    {'GLB':>4} {rd:7.3f} {rn:8.3f} {(rn-rd)*100:+6.1f} {len(d):7d}")
    n_anos = d["ano"].nunique()
    anos_nott_maior = sum(
        (g[g.turno=='noturno'].n_reprovados.sum()/max(g[g.turno=='noturno'].n_avaliados.sum(),1))
        > (g[g.turno=='diurno'].n_reprovados.sum()/max(g[g.turno=='diurno'].n_avaliados.sum(),1))
        for _, g in d.groupby("ano")
        if len(g[g.turno=='noturno']) and len(g[g.turno=='diurno']))
    L.append(f"    -> noturno > diurno em {anos_nott_maior}/{n_anos} anos")

    txt = "\n".join(L)
    (OUT / "serie_turno_resumo.txt").write_text(txt, encoding="utf-8")
    m.summary2().tables[1].to_csv(OUT / "serie_turno_glm.csv")
    print(txt)


def scheirer_ray_hare(df, y, f1, f2):
    d = df[[y, f1, f2]].dropna().copy()
    N = len(d)
    d["R"] = stats.rankdata(d[y])
    MS_total = d["R"].var(ddof=0) * N / (N - 1)  # ~ variance of ranks
    ss_total = ((d["R"] - d["R"].mean()) ** 2).sum()
    def ss_factor(cols):
        g = d.groupby(cols)["R"]
        return (g.count() * (g.mean() - d["R"].mean()) ** 2).sum()
    ss_a = ss_factor(f1); ss_b = ss_factor(f2)
    ss_ab_cells = ss_factor([f1, f2])
    ss_inter = ss_ab_cells - ss_a - ss_b
    ms_total = ss_total / (N - 1)
    la, lb = d[f1].nunique(), d[f2].nunique()
    out = {}
    for name, ss, df_ in [(f1, ss_a, la - 1), (f2, ss_b, lb - 1),
                          ("interação", ss_inter, (la - 1) * (lb - 1))]:
        H = ss / ms_total
        out[name] = dict(H=H, df=df_, p=stats.chi2.sf(H, df_))
    return out


if __name__ == "__main__":
    main()
