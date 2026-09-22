#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figuras para o artigo. Saída: figuras/*.pdf (vetorial) e .png (rascunho).

- fig_percepcao_dimensoes : todos os 17 itens substantivos (Exp. 2+3, N=74),
  box + pontos + média, ordenados pela média decrescente, cor = dimensão
  a priori (D1..D4), "*" nos significativos (Wilcoxon 1-caudal + Holm).
  Lê os dados BRUTOS direto (rótulos conforme o artigo; descarta colunas de
  identificação, mantendo os itens Likert -- o enunciado de Q09 cita "e-mail").
- fig_serie_turno : série histórica de reprovação por turno.
- fig_serie_turno_periodo : slopegraph ano a ano (diurno -> noturno) para o
  período ideal e o não ideal, com pooled 2009-2026 + IC 95% de Wilson e a
  razao de chances de turno por periodo.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]
BRUTOS = ROOT / "dados" / "brutos"
ANON = ROOT / "dados" / "anonymous"
SERIE = ROOT / "dados" / "serie-historica"
FIG = ROOT / "figuras"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

F_AV = ANON / "avaliacao_exp2_exp3.csv"

# Dimensões a priori (Tabela 5 / Seção 3.4) ---------------------------------
DIM_ITENS = {
    "D1": ["Q02", "Q11", "Q13", "Q16", "Q17"],   # material curado por IA
    "D2": ["Q04", "Q05", "Q06", "Q07", "Q14"],   # prática ativa guiada
    "D3": ["Q03", "Q08", "Q09", "Q10"],          # correção automática e feedback
    "D4": ["Q01", "Q12", "Q15"],                 # uso e autopercepção
}
DIM_DE = {q: d for d, qs in DIM_ITENS.items() for q in qs}
DIM_COR = {"D1": "#d1731f", "D2": "#2f6fb0", "D3": "#3f9b52", "D4": "#8a6fb0"}
DIM_ROT = {"D1": "D1 · material curado por IA", "D2": "D2 · prática ativa guiada",
           "D3": "D3 · correção automática e feedback", "D4": "D4 · uso e autopercepção"}
MNEM = {
    "Q01": "uso do material", "Q02": "formatos do NotebookLM", "Q03": "testes no Colab",
    "Q04": "abertura da aula", "Q05": "tentar sozinho antes", "Q06": "exercícios em aula",
    "Q07": "dose de exercícios", "Q08": "feedback VPL+IA", "Q09": "feedback por e-mail",
    "Q10": "provas aderentes", "Q11": "preparo para provas", "Q12": "evolução na lógica",
    "Q13": "sem NotebookLM, pior", "Q14": "teoria + prática", "Q15": "recomendar o NotebookLM",
    "Q16": "interesse pelo NotebookLM", "Q17": "acompanhar a dificuldade",
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("->", FIG / f"{name}.pdf")


def _carrega_itens():
    """DataFrame N=74 (Exp. 2+3) com colunas Q01..Q17 numéricas."""
    av = pd.read_csv(F_AV)
    def qid(c):
        m = re.search(r"\[\s*(\d{1,2})", str(c))
        return f"Q{int(m.group(1)):02d}" if m else None
    cols = {}
    for c in av.columns:
        if "[" in str(c) and "]" in str(c):
            v = pd.to_numeric(av[c], errors="coerce").dropna()
            if len(v) and v.isin([1, 2, 3, 4, 5]).all():
                q = qid(c)
                if q and 1 <= int(q[1:]) <= 17:
                    cols[q] = pd.to_numeric(av[c], errors="coerce")
    return pd.DataFrame(cols)[[f"Q{i:02d}" for i in range(1, 18)]]


def _rank_biserial(x, mu=3):
    dif = np.asarray(x, float) - mu
    dif = dif[dif != 0]
    r = stats.rankdata(np.abs(dif))
    return (r[dif > 0].sum() - r[dif < 0].sum()) / r.sum()


def _fmt_p(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def fig_percepcao_dimensoes():
    d = _carrega_itens()
    itens = list(d.columns)
    praw = [stats.wilcoxon(d[q].dropna() - 3, alternative="greater").pvalue for q in itens]
    padj = dict(zip(itens, multipletests(praw, method="holm")[1]))
    signif = {q: padj[q] < 0.05 for q in itens}
    rbis = {q: _rank_biserial(d[q].dropna()) for q in itens}
    ordem = list(d.mean().sort_values(ascending=False).index)   # topo = maior média

    X_R, X_P, X_MAX = 5.55, 6.55, 7.6          # colunas de texto à direita do 5
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    trans = blended_transform_factory(ax.transAxes, ax.transData)  # x: fração; y: dados
    for i, q in enumerate(ordem):
        y = len(ordem) - 1 - i
        x = d[q].dropna().to_numpy(float)
        cor = DIM_COR[DIM_DE[q]]
        ns = not signif[q]
        if ns:                                   # faixa cinza nos NÃO significativos
            ax.add_patch(plt.Rectangle((-0.30, y - 0.5), 1.30, 1.0, transform=trans,
                                       facecolor="0.93", edgecolor="none",
                                       zorder=0, clip_on=False))
        bp = ax.boxplot(x, positions=[y], vert=False, widths=0.58, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.6),
                        whiskerprops=dict(color="0.55"), capprops=dict(color="0.55"))
        bp["boxes"][0].set(facecolor=cor, alpha=0.55 if not ns else 0.30, edgecolor=cor)
        ax.scatter(x.mean(), y, marker="D", s=32, color="white", edgecolor="black",
                   lw=1.0, zorder=4)
        ax.annotate(f"{x.mean():.2f}".replace(".", ","), (x.mean(), y),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    va="bottom", fontsize=6.3, color="0.25", zorder=5)
        ax.text(0.86, y, f"{q}  {MNEM[q]}", ha="right", va="center", fontsize=8.2,
                fontweight="bold" if signif[q] else "normal",
                color="black" if signif[q] else "0.35")
        cr = "0.45" if ns else "black"
        ax.text(X_R, y, f"{rbis[q]:+.2f}".replace(".", ","), ha="center", va="center",
                fontsize=8, color=cr, family="monospace")
        ax.text(X_P, y, _fmt_p(padj[q]), ha="center", va="center", fontsize=8,
                color=cr, family="monospace",
                fontweight="bold" if ns else "normal")

    ytop = len(ordem) - 0.25
    ax.text(3, ytop + 0.55, "resposta (Likert 1–5)", ha="center", fontsize=8, style="italic")
    ax.text(X_R, ytop + 0.55, "$r$", ha="center", fontsize=9)
    ax.text(X_P, ytop + 0.55, "$p$ (Holm)", ha="center", fontsize=9)
    ax.axvline(3, color="k", ls="--", lw=0.8, zorder=1)
    ax.axvline(5.25, color="0.8", lw=0.8)         # separador visual
    ax.set_yticks([]); ax.set_ylim(-0.7, ytop + 1.0)
    ax.set_xlim(0.2, X_MAX); ax.set_xticks(range(1, 6))
    ax.set_xlabel("caixa: quartis e mediana  ·  losango: média  ·  "
                  "faixa cinza: $p \\geq 0{,}05$ (Wilcoxon 1-caudal + Holm)", fontsize=8.5)
    ax.set_title("Percepção discente por item e dimensão (Exp. 2 + 3, N = 74)")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=DIM_COR[k],
                          alpha=0.7, label=DIM_ROT[k]) for k in ("D1", "D2", "D3", "D4")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=7.5, frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(1, 5)
    save(fig, "fig_percepcao_dimensoes")


def _serie_periodo_ideal():
    """Série agregada restrita ao PERÍODO IDEAL de cada ano.

    Período ideal = o quadrimestre com mais turmas naquele ano (turnos
    somados) -- o quadrimestre comparável à coorte de 2026.1 do artigo.
    Retorna DataFrame indexado por ano com taxa de reprovação diurno/noturno,
    alunos avaliados e turmas.
    """
    d = pd.read_csv(SERIE / "pi0505_agregado_quadrimestre_turno_2009-2026.csv",
                    dtype={"quadrimestre": str})
    d["ano"] = d["quadrimestre"].str[:4].astype(int)
    cl = d.groupby(["ano", "quadrimestre"])["n_turmas"].sum().reset_index()
    ideal = cl.loc[cl.groupby("ano")["n_turmas"].idxmax()].set_index("ano")["quadrimestre"]
    di = d[d.apply(lambda r: r["quadrimestre"] == ideal[r["ano"]], axis=1)]
    rate = di.pivot(index="ano", columns="turno", values="taxa_reprov_pool")
    nav = di.pivot(index="ano", columns="turno", values="n_avaliados")
    out = pd.DataFrame({"quad": ideal,
                        "day_rate": rate["diurno"], "eve_rate": rate["noturno"],
                        "day_n": nav["diurno"], "eve_n": nav["noturno"]})
    return out.sort_index()


def fig_serie_turno():
    s = _serie_periodo_ideal()
    yrs = s.index.to_numpy()
    x = np.arange(len(yrs))
    covid = (yrs >= 2020) & (yrs <= 2022)
    a, b = x[covid].min(), x[covid].max()
    day_pool = (s["day_n"] * s["day_rate"]).sum() / s["day_n"].sum()
    eve_pool = (s["eve_n"] * s["eve_rate"]).sum() / s["eve_n"].sum()
    pct = lambda v: f"{v*100:.1f}".replace(".", ",") + "%"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1],
                                                "hspace": 0.12})
    ORANGE, BLUE = "#e8710a", "#1f6feb"

    # -- painel de cima: taxa de reprovação, só período ideal ----------------
    ax1.axvspan(a - .5, b + .5, color="0.85", alpha=.6, zorder=0, label="Janela COVID")
    ax1.fill_between(x, s["day_rate"], s["eve_rate"],
                     where=(s["eve_rate"] >= s["day_rate"]), color=BLUE, alpha=.10)
    ax1.plot(x, s["day_rate"], "-o", ms=4, label="Diurno", color=ORANGE)
    ax1.plot(x, s["eve_rate"], "-o", ms=4, label="Noturno", color=BLUE)
    ax1.axhline(day_pool, ls=":", lw=1, color=ORANGE)
    ax1.axhline(eve_pool, ls=":", lw=1, color=BLUE)
    ax1.text(6.5, day_pool - 0.032, f"pooled período ideal: diurno {pct(day_pool)}",
             color=ORANGE, fontsize=7.5, ha="left")
    ax1.text(6.5, eve_pool + 0.012, f"noturno {pct(eve_pool)}",
             color=BLUE, fontsize=7.5, ha="left")
    ax1.set_ylabel("Taxa de reprovação (pooled)")
    ax1.set_ylim(0.10, 0.50)
    ax1.legend(fontsize=8, loc="upper left", ncol=3, frameon=False)
    ax1.set_title("Reprovação em DIP e tamanho da coorte por turno, apenas no período ideal, "
                  "2009–2026\n(noturno > diurno em 18/18 anos)", fontsize=9.5)

    # -- painel de baixo: alunos avaliados, só período ideal ----------------
    w = 0.4
    ax2.bar(x - w/2, s["day_n"], w, label="Diurno", color=ORANGE, alpha=.85)
    ax2.bar(x + w/2, s["eve_n"], w, label="Noturno", color=BLUE, alpha=.85)
    ax2.axvspan(a - .5, b + .5, color="0.85", alpha=.6, zorder=0)
    ax2.set_ylabel("Alunos avaliados")
    ax2.set_xticks(x); ax2.set_xticklabels(yrs, rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
    save(fig, "fig_serie_turno")


def _wilson(k, n, z=1.96):
    "IC de Wilson para uma proporção (k sucessos em n)."
    if not n:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centro = p + z * z / (2 * n)
    meia = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centro - meia) / den, (centro + meia) / den)


def _odds_ratio(p_num, p_den):
    "OR = chance(num) / chance(den)."
    return (p_num / (1 - p_num)) / (p_den / (1 - p_den))


def _serie_periodo_split():
    """Reprovação por turno, separada em período ideal vs. não ideal.

    Mesma definição da Fig. 3 (período ideal = quadrimestre com mais turmas no
    ano). Para cada período devolve, por turno, o *pooled* (k, n, rate) sobre
    todos os quadrimestres daquele período, além de:
      por_ano  -- DataFrame (índice = ano) com as taxas diurna/noturna de cada
                  quadrimestre-ano em que os DOIS turnos foram ofertados;
      anos_noturno_maior / anos_total -- contagem sobre esses anos pareados;
      or_turno -- razão de chances noturno vs. diurno (pooled).
    """
    d = pd.read_csv(SERIE / "pi0505_agregado_quadrimestre_turno_2009-2026.csv",
                    dtype={"quadrimestre": str})
    d["ano"] = d["quadrimestre"].str[:4].astype(int)
    cl = d.groupby(["ano", "quadrimestre"])["n_turmas"].sum().reset_index()
    ideal = cl.loc[cl.groupby("ano")["n_turmas"].idxmax()].set_index("ano")["quadrimestre"]
    d["periodo"] = np.where(
        d.apply(lambda r: r["quadrimestre"] == ideal[r["ano"]], axis=1),
        "ideal", "nao_ideal")
    out = {}
    for per, g in d.groupby("periodo"):
        rec = {}
        for turno, gg in g.groupby("turno"):
            k, n = int(gg["n_reprovados"].sum()), int(gg["n_avaliados"].sum())
            rec[turno] = dict(k=k, n=n, rate=k / n)
        piv = g.pivot_table(index="ano", columns="turno",
                            values=["n_reprovados", "n_avaliados"], aggfunc="sum")
        dr = piv[("n_reprovados", "diurno")] / piv[("n_avaliados", "diurno")]
        er = piv[("n_reprovados", "noturno")] / piv[("n_avaliados", "noturno")]
        pa = pd.DataFrame({"diurno": dr, "noturno": er,
                           "n_diurno": piv[("n_avaliados", "diurno")],
                           "n_noturno": piv[("n_avaliados", "noturno")]}
                          ).dropna().sort_index()
        rec["por_ano"] = pa
        rec["anos_noturno_maior"] = int((pa["noturno"] > pa["diurno"]).sum())
        rec["anos_total"] = int(len(pa))
        rec["or_turno"] = _odds_ratio(rec["noturno"]["rate"], rec["diurno"]["rate"])
        out[per] = rec
    return out


def fig_serie_turno_periodo():
    """Alternativa à Fig. 3: slopegraph ano a ano (diurno -> noturno) para o
    período ideal e o não ideal, com o *pooled* 2009--2026 (losango) e IC 95 %
    de Wilson sobreposto."""
    S = _serie_periodo_split()
    grupos = [("ideal", "Período ideal\n(coorte comparável a 2026.1)"),
              ("nao_ideal", "Período não ideal\n(demais quadrimestres do ano)")]
    ORANGE, BLUE, RED = "#e8710a", "#1f6feb", "#c0392b"
    OFF = 0.17
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for gi, (key, _rot) in enumerate(grupos):
        rec = S[key]
        xd, xe = gi - OFF, gi + OFF
        pa = rec["por_ano"]
        # -- uma linha fina por quadrimestre-ano (diurno -> noturno) -----------
        for ano, r in pa.iterrows():
            inv = r["noturno"] < r["diurno"]
            covid = 2020 <= ano <= 2022
            lc = RED if inv else "0.60"
            j = ((int(ano) % 7) - 3) * 0.013          # deslocamento p/ separar as linhas
            ax.plot([xd + j, xe + j], [r["diurno"], r["noturno"]], "-", color=lc,
                    lw=1.1 if inv else 0.9, alpha=0.85 if inv else 0.4, zorder=2)
            for xx, yy in ((xd + j, r["diurno"]), (xe + j, r["noturno"])):
                ax.scatter(xx, yy, s=14, zorder=3, lw=0.9, edgecolor=lc,
                           facecolor="white" if covid else lc)
            if inv:                                   # rotula os anos com inversão
                nd, ne = int(r["n_diurno"]), int(r["n_noturno"])
                ax.annotate(f"{int(ano)} · {nd} diurno / {ne} noturno",
                            (xe + j, r["noturno"]), textcoords="offset points",
                            xytext=(7, 0), ha="left", va="center", fontsize=6.6,
                            color=RED, zorder=6)
        # -- pooled: losango + IC 95 % de Wilson ------------------------------
        for turno, xx, cor, dx, ha in (("diurno", xd, ORANGE, -13, "right"),
                                       ("noturno", xe, BLUE, 13, "left")):
            rr = rec[turno]
            lo, hi = _wilson(rr["k"], rr["n"])
            ax.errorbar(xx, rr["rate"], yerr=[[rr["rate"] - lo], [hi - rr["rate"]]],
                        fmt="D", ms=8, color=cor, mec="black", mew=1.0,
                        ecolor="black", elinewidth=1.3, capsize=4, zorder=5)
            ax.annotate(f"{rr['rate']*100:.1f}%".replace(".", ","),
                        (xx, rr["rate"]), textcoords="offset points", xytext=(dx, 0),
                        ha=ha, va="center", fontsize=8.5, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8",
                                  alpha=0.92))
            ax.text(xx, 0.035, f"n={rr['n']:,}".replace(",", "."), ha="center",
                    fontsize=6.8, color="0.4")
        ax.text(xd, 0.008, "diurno", ha="center", fontsize=7.5, color=ORANGE)
        ax.text(xe, 0.008, "noturno", ha="center", fontsize=7.5, color=BLUE)
        ax.text(gi, 0.80,
                f"noturno > diurno em {rec['anos_noturno_maior']}/{rec['anos_total']} anos\n"
                f"razão de chances pooled = {rec['or_turno']:.2f}".replace(".", ","),
                ha="center", va="top", fontsize=8, style="italic")
    ax.set_xticks([0, 1]); ax.set_xticklabels([r for _, r in grupos], fontsize=8.5)
    ax.tick_params(axis="x", length=0, pad=20)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Taxa de reprovação em DIP por quadrimestre-ano")
    ax.set_ylim(0, 0.86)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 0.81, 0.1)])
    handles = [
        plt.Line2D([], [], color="0.60", lw=1, alpha=.6, label="1 quadrimestre-ano"),
        plt.Line2D([], [], color=RED, lw=1.2, label="ano com inversão (noturno < diurno)"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.5",
                   label="ano na janela COVID (2020–2022)"),
        plt.Line2D([], [], marker="D", ls="", mfc="0.7", mec="black",
                   label="pooled 2009–2026  ($\\pm$ IC 95% Wilson)"),
    ]
    ax.legend(handles=handles, fontsize=6.9, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.33), frameon=False,
              handletextpad=0.5, columnspacing=1.2)
    ax.set_title("Reprovação por turno, ano a ano: período ideal vs. não ideal",
                 fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "fig_serie_turno_periodo")


if __name__ == "__main__":
    fig_percepcao_dimensoes()
    fig_serie_turno()
    fig_serie_turno_periodo()
