"""
Avaliação offline do sistema de recomendação (Etapa 4).

Como a base não possui interações de usuários, a avaliação usa métricas
baseadas em conteúdo, comuns na literatura de recomendação musical:
  - Erro acústico (MAE): diferença absoluta média entre a referência e as
    recomendações nos 9 atributos contínuos normalizados (0-1). Menor = mais parecido.
  - Precisão de gênero@10: fração das recomendações do mesmo gênero da referência.
  - Diversidade: número de gêneros distintos no Top-10.
  - Popularidade média e % de faixas de "cauda longa" (popularidade < 30).
  - Cobertura de catálogo: % de faixas distintas que aparecem em alguma recomendação.
  - Tempo médio por consulta.

Uso:  python src/avaliacao.py dataset/dataset.csv resultados/
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(__file__))
from recomendador import (FEATURES_COMPLETAS, FEATURES_CONTINUAS,  # noqa: E402
                          Recomendador, carregar_e_tratar)

N_AMOSTRA = 1000
SEMENTE = 42
TOP_N = 10
LIMIAR_CAUDA_LONGA = 30


def avaliar(caminho_csv: str, saida: str) -> None:
    os.makedirs(saida, exist_ok=True)
    df = carregar_e_tratar(caminho_csv)
    print(f"Catálogo após tratamento: {len(df)} faixas, {df.track_genre.nunique()} gêneros")

    # espaço comum de comparação (9 atributos contínuos, 0-1)
    Xc = MinMaxScaler().fit_transform(df[FEATURES_CONTINUAS])
    generos = df["track_genre"].values
    pop = df["popularity"].values

    rng = np.random.default_rng(SEMENTE)
    refs = rng.choice(len(df), size=N_AMOSTRA, replace=False)

    modelos = {
        "cos11": Recomendador(df, FEATURES_COMPLETAS, "cosine"),
        "euc11": Recomendador(df, FEATURES_COMPLETAS, "euclidean"),
        "cos9": Recomendador(df, FEATURES_CONTINUAS, "cosine"),
        "euc9": Recomendador(df, FEATURES_CONTINUAS, "euclidean"),
    }
    mais_populares = np.argsort(-pop)[: TOP_N + 1]

    tecnicas = {
        "Aleatório (baseline)": lambda i: [int(j) for j in rng.choice(len(df), TOP_N + 1, replace=False) if j != i][:TOP_N],
        "Mais populares (baseline)": lambda i: [int(j) for j in mais_populares if j != i][:TOP_N],
        "KNN cosseno, 11 atrib. (Etapa 3)": lambda i: modelos["cos11"].recomendar_indices(i, TOP_N),
        "KNN euclidiana, 11 atrib.": lambda i: modelos["euc11"].recomendar_indices(i, TOP_N),
        "KNN cosseno, 9 atrib.": lambda i: modelos["cos9"].recomendar_indices(i, TOP_N),
        "KNN euclidiana, 9 atrib.": lambda i: modelos["euc9"].recomendar_indices(i, TOP_N),
        "Híbrido 5+5, cosseno 11 (Etapa 3)": lambda i: modelos["cos11"].recomendar_hibrido_indices(i),
        "Híbrido 5+5, euclidiana 9 (proposto)": lambda i: modelos["euc9"].recomendar_hibrido_indices(i),
    }

    linhas, por_atributo, por_genero = [], {}, {}
    for nome, f in tecnicas.items():
        mae, prec, div, popm, cauda, itens, tempos, erro_attr = [], [], [], [], [], set(), [], []
        prec_ref = []
        for i in refs:
            t0 = time.perf_counter()
            recs = f(int(i))
            tempos.append(time.perf_counter() - t0)
            diff = np.abs(Xc[recs] - Xc[i])
            erro_attr.append(diff.mean(axis=0))
            mae.append(diff.mean())
            p = float((generos[recs] == generos[i]).mean())
            prec.append(p); prec_ref.append((generos[i], p))
            div.append(len(set(generos[recs])))
            popm.append(pop[recs].mean())
            cauda.append((pop[recs] < LIMIAR_CAUDA_LONGA).mean())
            itens.update(recs)
        linhas.append({
            "Técnica": nome,
            "Erro acústico (MAE)": np.mean(mae),
            "Precisão de gênero@10": np.mean(prec),
            "Gêneros distintos@10": np.mean(div),
            "Popularidade média": np.mean(popm),
            "% cauda longa": np.mean(cauda) * 100,
            "Cobertura do catálogo (%)": len(itens) / len(df) * 100,
            "Tempo/consulta (ms)": np.mean(tempos) * 1000,
        })
        por_atributo[nome] = np.mean(erro_attr, axis=0).tolist()
        g = pd.DataFrame(prec_ref, columns=["genero", "p"]).groupby("genero").p.mean()
        por_genero[nome] = g.to_dict()
        print(f"  ok: {nome}")

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(os.path.join(saida, "metricas_comparativas.csv"), index=False)
    print("\n", tabela.round(3).to_string(index=False))

    # precisão por gênero com amostra estratificada (50 faixas por gênero), modelo euclidiano 9
    rng_g = np.random.default_rng(SEMENTE)
    prec_genero = {}
    for g, grupo in df.groupby("track_genre"):
        amostra = rng_g.choice(grupo.index.values, size=min(50, len(grupo)), replace=False)
        ps = [(generos[modelos["euc9"].recomendar_indices(int(i), TOP_N)] == g).mean() for i in amostra]
        prec_genero[g] = float(np.mean(ps))

    # saturação do escore de cosseno: similaridade do 10º vizinho (modelo da Etapa 3)
    m = modelos["cos11"]
    sat10, sat1 = [], []
    for i in refs:
        d, _ = m._vizinhos(int(i), TOP_N)
        sat1.append((1 - d[1]) * 100); sat10.append((1 - d[TOP_N]) * 100)
    sat = {"sim_1o_vizinho_media": float(np.mean(sat1)),
           "sim_10o_vizinho_media": float(np.mean(sat10)),
           "pct_consultas_10o_acima_99": float((np.array(sat10) >= 99).mean() * 100),
           "pct_consultas_10o_acima_98": float((np.array(sat10) >= 98).mean() * 100),
           "sim_10o_vizinho_todas": [float(x) for x in sat10]}

    # estudo de caso: Master of Puppets
    casos = {}
    for chave, rec in [("Híbrido 5+5, cosseno 11 (Etapa 3)", modelos["cos11"]),
                       ("Híbrido 5+5, euclidiana 9 (proposto)", modelos["euc9"])]:
        idx = rec.buscar("Master of Puppets", "Metallica")
        ids = rec.recomendar_hibrido_indices(idx)
        t = rec._tabela(idx, ids)
        t["energy"] = df.loc[ids, "energy"].values
        t["loudness"] = df.loc[ids, "loudness"].values
        t["acousticness"] = df.loc[ids, "acousticness"].values
        t["MAE acústico"] = np.abs(Xc[ids] - Xc[idx]).mean(axis=1).round(3)
        casos[chave] = {"referencia": rec.descrever(idx),
                        "ref_energy": float(df.at[idx, "energy"]),
                        "ref_loudness": float(df.at[idx, "loudness"]),
                        "ref_acousticness": float(df.at[idx, "acousticness"]),
                        "tabela": t.to_dict(orient="records")}

    resumo = {
        "catalogo_faixas": int(len(df)),
        "catalogo_generos": int(df.track_genre.nunique()),
        "popularidade_media_catalogo": float(pop.mean()),
        "pct_catalogo_cauda_longa": float((pop < LIMIAR_CAUDA_LONGA).mean() * 100),
        "amostra": N_AMOSTRA, "semente": SEMENTE,
        "features_continuas": FEATURES_CONTINUAS,
        "erro_por_atributo": por_atributo,
        "precisao_por_genero_amostra_geral": por_genero,
        "precisao_por_genero_estratificada_euc9": prec_genero,
        "saturacao_cosseno": sat,
        "estudo_de_caso": casos,
    }
    with open(os.path.join(saida, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=1, default=float)
    print("\nSaturação do cosseno:", {k: v for k, v in sat.items() if k != "sim_10o_vizinho_todas"})


if __name__ == "__main__":
    avaliar(sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset.csv",
            sys.argv[2] if len(sys.argv) > 2 else "resultados")
