"""Gera as figuras da seção Resultados a partir de resultados/*.csv|json.
Uso: python src/graficos.py dataset/dataset.csv resultados/
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from recomendador import FEATURES_CONTINUAS, Recomendador, carregar_e_tratar  # noqa: E402

AZUL, LARANJA, AQUA, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
TXT, TXT2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRADE,
    "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": TXT,
})

COR_TECNICA = {
    "Aleatório (baseline)": CINZA, "Mais populares (baseline)": CINZA,
    "KNN cosseno, 11 atrib. (Etapa 3)": AZUL, "KNN euclidiana, 11 atrib.": "#86b6ef",
    "KNN cosseno, 9 atrib.": "#86b6ef", "KNN euclidiana, 9 atrib.": LARANJA,
    "Híbrido 5+5, cosseno 11 (Etapa 3)": AZUL, "Híbrido 5+5, euclidiana 9 (proposto)": LARANJA,
}


def barras_h(ax, nomes, valores, fmt, cores):
    y = np.arange(len(nomes))[::-1]
    ax.barh(y, valores, color=cores, height=0.62, edgecolor="white", linewidth=1)
    ax.set_yticks(y, nomes)
    ax.xaxis.grid(True, color=GRADE, linewidth=0.8); ax.set_axisbelow(True)
    for yi, v in zip(y, valores):
        ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", color=TXT, fontsize=9)
    ax.set_xlim(0, max(valores) * 1.18)


def main(csv, pasta):
    out = os.path.join(pasta, "figuras"); os.makedirs(out, exist_ok=True)
    tab = pd.read_csv(os.path.join(pasta, "metricas_comparativas.csv"))
    r = json.load(open(os.path.join(pasta, "resultados.json"), encoding="utf-8"))
    nomes = tab["Técnica"].tolist(); cores = [COR_TECNICA[n] for n in nomes]

    # Fig. 2 - erro acústico por técnica
    fig, ax = plt.subplots(figsize=(8, 4))
    barras_h(ax, nomes, tab["Erro acústico (MAE)"].values, "{:.3f}", cores)
    ax.set_xlabel("Erro absoluto médio por atributo (escala 0-1) — menor é melhor")
    ax.set_title("Proximidade acústica referência × Top-10")
    fig.tight_layout(); fig.savefig(f"{out}/fig2_erro_acustico.png"); plt.close(fig)

    # Fig. 3 - precisão de gênero
    fig, ax = plt.subplots(figsize=(8, 4))
    barras_h(ax, nomes, tab["Precisão de gênero@10"].values * 100, "{:.1f}%", cores)
    ax.set_xlabel("% das recomendações no mesmo gênero da referência")
    ax.set_title("Precisão de gênero@10, por técnica")
    fig.tight_layout(); fig.savefig(f"{out}/fig3_precisao_genero.png"); plt.close(fig)

    # Fig. 4 - erro por atributo: Etapa 3 vs proposto (KNN puro)
    e3 = r["erro_por_atributo"]["KNN cosseno, 11 atrib. (Etapa 3)"]
    pr = r["erro_por_atributo"]["KNN euclidiana, 9 atrib."]
    x = np.arange(len(FEATURES_CONTINUAS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2 - 0.01, e3, w, color=AZUL, label="KNN cosseno, 11 atributos (Etapa 3)")
    ax.bar(x + w / 2 + 0.01, pr, w, color=LARANJA, label="KNN euclidiana, 9 atributos")
    ax.set_xticks(x, FEATURES_CONTINUAS, rotation=30, ha="right")
    ax.set_ylabel("Erro absoluto médio (0-1)")
    ax.yaxis.grid(True, color=GRADE); ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(e3), max(pr)) * 1.3)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    ax.set_title("Erro por atributo de áudio: modelo da Etapa 3 vs. variante euclidiana")
    fig.tight_layout(); fig.savefig(f"{out}/fig4_erro_por_atributo.png"); plt.close(fig)

    # Fig. 5 - saturação do escore de cosseno
    s = np.array(r["saturacao_cosseno"]["sim_10o_vizinho_todas"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(s, bins=np.arange(95, 100.01, 0.1), color=AZUL, edgecolor="white", linewidth=0.6)
    ax.axvline(99, color=TXT2, linestyle="--", linewidth=1)
    ax.text(98.9, ax.get_ylim()[1] * 0.9, f"{(s >= 99).mean()*100:.0f}% das consultas\ncom 10º vizinho ≥ 99% →",
            color=TXT, fontsize=9, va="top", ha="right")
    ax.set_xlabel("Similaridade do cosseno do 10º vizinho mais próximo (%)")
    ax.set_ylabel("Nº de consultas")
    ax.set_title("Saturação do escore de cosseno (modelo da Etapa 3)")
    ax.yaxis.grid(True, color=GRADE); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{out}/fig5_saturacao_cosseno.png"); plt.close(fig)

    # Fig. 6 - viés de popularidade
    fig, ax = plt.subplots(figsize=(8, 4))
    barras_h(ax, nomes, tab["Popularidade média"].values, "{:.1f}", cores)
    cat = r["popularidade_media_catalogo"]
    ax.axvline(cat, color=TXT2, linestyle="--", linewidth=1)
    ax.text(cat, 7.75, f" média do catálogo ({cat:.1f})", color=TXT2, fontsize=8.5, va="center")
    ax.set_ylim(-0.6, 8.2)
    ax.set_xlabel("Popularidade média das faixas recomendadas (0-100)")
    ax.set_title("Viés de popularidade: recomendações vs. catálogo")
    fig.tight_layout(); fig.savefig(f"{out}/fig6_popularidade.png"); plt.close(fig)

    # Fig. 7 - precisão por gênero (estratificada)
    g = pd.Series(r["precisao_por_genero_estratificada_euc9"]).sort_values() * 100
    sel = pd.concat([g.head(10), g.tail(10)])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    cs = [CINZA] * 10 + [AQUA] * 10
    y = np.arange(len(sel))
    ax.barh(y, sel.values, color=cs, height=0.66, edgecolor="white")
    ax.set_yticks(y, sel.index)
    for yi, v in zip(y, sel.values):
        ax.text(v, yi, f" {v:.0f}%", va="center", fontsize=8.5, color=TXT)
    ax.axhline(9.5, color=GRADE, linewidth=1)
    ax.xaxis.grid(True, color=GRADE); ax.set_axisbelow(True)
    ax.set_xlabel("Precisão de gênero@10 (%) — 50 faixas de referência por gênero")
    ax.set_title("Precisão de gênero@10: 10 maiores e 10 menores")
    fig.tight_layout(); fig.savefig(f"{out}/fig7_precisao_por_genero.png"); plt.close(fig)

    # Fig. 8 - estudo de caso Master of Puppets: perfil acústico
    df = carregar_e_tratar(csv)
    Xc = MinMaxScaler().fit_transform(df[FEATURES_CONTINUAS])
    perfis = {}
    for nome, rec in [("cos", Recomendador(df, metrica="cosine")),
                      ("euc", Recomendador(df, FEATURES_CONTINUAS, "euclidean"))]:
        idx = rec.buscar("Master of Puppets", "Metallica")
        perfis["ref"] = Xc[idx]
        perfis[nome] = Xc[rec.recomendar_hibrido_indices(idx)].mean(axis=0)
    x = np.arange(len(FEATURES_CONTINUAS)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w - 0.02, perfis["ref"], w, color="#3d3c39", label="Referência: Master of Puppets")
    ax.bar(x, perfis["cos"], w, color=AZUL, label="Média Top-10 — híbrido cosseno 11 (Etapa 3)")
    ax.bar(x + w + 0.02, perfis["euc"], w, color=LARANJA, label="Média Top-10 — híbrido euclidiana 9 (proposto)")
    ax.set_xticks(x, FEATURES_CONTINUAS, rotation=30, ha="right")
    ax.set_ylabel("Valor normalizado (0-1)"); ax.set_ylim(0, 1.3)
    ax.yaxis.grid(True, color=GRADE); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
    ax.set_title("Estudo de caso: perfil acústico da referência e das recomendações")
    fig.tight_layout(); fig.savefig(f"{out}/fig8_estudo_de_caso.png"); plt.close(fig)
    print("figuras salvas em", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset.csv",
         sys.argv[2] if len(sys.argv) > 2 else "resultados")
