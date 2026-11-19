"""
Aplicação de demonstração (Streamlit) do Sistema de Recomendação de Músicas.
Usada na apresentação do projeto e na aplicação com a comunidade.

Execução (na raiz do repositório):
    pip install -r requirements.txt
    streamlit run src/app.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from recomendador import FEATURES_CONTINUAS, Recomendador, carregar_e_tratar  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "dataset", "dataset.csv")

NOMES_PT = {
    "danceability": "Dançabilidade", "energy": "Energia", "loudness": "Volume",
    "speechiness": "Fala", "acousticness": "Acústica", "instrumentalness": "Instrumental",
    "liveness": "Ao vivo", "valence": "Positividade", "tempo": "Andamento",
}


@st.cache_resource(show_spinner="Carregando o catálogo de músicas...")
def carregar():
    return Recomendador(carregar_e_tratar(CSV), FEATURES_CONTINUAS, "euclidean")


st.set_page_config(page_title="Recomendador de Músicas", page_icon="🎵", layout="centered")
st.title("🎵 Recomendador de Músicas por Som")
st.caption("Projeto Aplicado III – Universidade Presbiteriana Mackenzie – 2026. "
           "Escolha uma música e receba sugestões com som parecido: 5 do mesmo gênero e 5 de outros gêneros.")

rec = carregar()

col1, col2 = st.columns([3, 2])
musica = col1.text_input("Nome da música", placeholder="Ex.: Master of Puppets")
artista = col2.text_input("Artista (opcional)", placeholder="Ex.: Metallica")

if musica:
    idx = rec.buscar(musica, artista or None)
    if idx is None:
        st.warning("Música não encontrada no catálogo. Tente outro nome ou confira a grafia.")
    else:
        ref = rec.df.loc[idx]
        st.subheader(f"Referência: {ref.track_name} — {ref.artists}")
        st.write(f"Gênero: **{ref.track_genre}** · Popularidade: **{ref.popularity}**")

        ids = rec.recomendar_hibrido_indices(idx)
        tabela = rec._tabela(idx, ids)
        tabela["Artista"] = tabela["Artista"].str.replace(";", ", ")
        st.markdown("### Recomendações")
        st.dataframe(tabela)

        # perfil acústico: referência x média das recomendações
        ref_v = rec.X[idx]
        rec_v = rec.X[ids].mean(axis=0)
        rotulos = [NOMES_PT[f] for f in rec.features]
        x = np.arange(len(rotulos))
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.bar(x - 0.2, ref_v, 0.38, color="#3d3c39", label="Música escolhida")
        ax.bar(x + 0.2, rec_v, 0.38, color="#eb6834", label="Média das recomendações")
        ax.set_xticks(x, rotulos, rotation=30, ha="right")
        ax.set_ylim(0, 1.15); ax.set_ylabel("0 = baixo · 1 = alto")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        fig.tight_layout()
        st.markdown("### Por que essas músicas?")
        st.pyplot(fig)
        st.caption("O sistema compara 9 características de áudio de cada faixa do catálogo (81 mil músicas) "
                   "e escolhe as mais próximas da música escolhida. A popularidade não é usada no cálculo.")

st.divider()
st.markdown("**Gostou das recomendações?** Avalie de 1 (nada parecidas) a 5 (muito parecidas):")
nota = st.radio("Nota", [1, 2, 3, 4, 5], horizontal=True, index=None, label_visibility="collapsed")
if nota:
    st.success("Obrigado pela avaliação! Anote a nota no formulário da pesquisa.")
