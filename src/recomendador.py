"""
Sistema de Recomendação de Músicas Baseado em Similaridade de Características de Áudio
Projeto Aplicado III - Universidade Presbiteriana Mackenzie - 2026

Módulo principal: carrega e trata o Spotify Tracks Dataset, normaliza os atributos
de áudio e gera recomendações com K-Nearest Neighbors (KNN).

Uso rápido:
    from recomendador import Recomendador
    rec = Recomendador.from_csv("dataset/dataset.csv")
    print(rec.recomendar_hibrido("Master of Puppets"))
"""
from __future__ import annotations

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# 11 atributos acústicos usados na Etapa 3 (modelo original)
FEATURES_COMPLETAS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
# 9 atributos contínuos (sem 'key' e 'mode', que são categóricos)
FEATURES_CONTINUAS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]


def carregar_e_tratar(caminho_csv: str) -> pd.DataFrame:
    """Lê o CSV, remove nulos e deduplica por (nome da faixa + artista)."""
    df = pd.read_csv(caminho_csv)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna()
    chave = (df["track_name"].str.lower().str.strip() + "|"
             + df["artists"].str.lower().str.strip())
    df = df.loc[~chave.duplicated()].reset_index(drop=True)
    return df


class Recomendador:
    """Recomendador baseado em conteúdo (KNN sobre atributos de áudio normalizados)."""

    def __init__(self, df: pd.DataFrame, features: list[str] | None = None,
                 metrica: str = "cosine"):
        self.df = df.reset_index(drop=True)
        self.features = features or FEATURES_COMPLETAS
        self.metrica = metrica
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(self.df[self.features])
        self.modelo = NearestNeighbors(metric=metrica, algorithm="brute")
        self.modelo.fit(self.X)

    @classmethod
    def from_csv(cls, caminho_csv: str, **kwargs) -> "Recomendador":
        return cls(carregar_e_tratar(caminho_csv), **kwargs)

    # ------------------------------------------------------------------ busca
    def buscar(self, musica: str, artista: str | None = None) -> int | None:
        """Retorna o índice da faixa de referência (a mais popular entre as que casam)."""
        m = self.df["track_name"].str.contains(musica, case=False, na=False, regex=False)
        if artista:
            m &= self.df["artists"].str.contains(artista, case=False, na=False, regex=False)
        achados = self.df[m]
        if achados.empty:
            return None
        exatos = achados[achados["track_name"].str.lower() == musica.lower()]
        base = exatos if not exatos.empty else achados
        return int(base["popularity"].idxmax())

    def _vizinhos(self, idx: int, k: int):
        dist, ind = self.modelo.kneighbors(self.X[idx:idx + 1], n_neighbors=k + 1)
        return dist[0], ind[0]

    def _similaridade(self, d: float) -> float:
        """Converte distância em escore de similaridade (0-100%)."""
        if self.metrica == "cosine":
            return (1 - d) * 100
        # euclidiana: normaliza pela maior distância possível no hipercubo [0,1]^n
        return (1 - d / (len(self.features) ** 0.5)) * 100

    # ---------------------------------------------------------- recomendações
    def recomendar_indices(self, idx: int, n: int = 10) -> list[int]:
        """KNN puro: os n vizinhos mais próximos, ignorando a própria faixa e versões."""
        titulo = self.df.at[idx, "track_name"].lower()
        _, ind = self._vizinhos(idx, n + 50)
        vistos, saida = {titulo}, []
        for j in ind:
            t = self.df.at[j, "track_name"].lower()
            if j == idx or t in vistos:
                continue
            vistos.add(t)
            saida.append(int(j))
            if len(saida) == n:
                break
        return saida

    def recomendar_hibrido_indices(self, idx: int, n_mesmo: int = 5,
                                   n_outros: int = 5, k: int = 1000) -> list[int]:
        """Top-N híbrido: n_mesmo faixas do gênero da referência + n_outros de outros gêneros,
        com lista de reserva (fallback) para completar o total."""
        genero = self.df.at[idx, "track_genre"]
        titulo = self.df.at[idx, "track_name"].lower()
        _, ind = self._vizinhos(idx, k)
        mesmo, outros, reserva, vistos = [], [], [], {titulo}
        for j in ind:
            t = self.df.at[j, "track_name"].lower()
            if j == idx or t in vistos:
                continue
            if self.df.at[j, "track_genre"] == genero:
                if len(mesmo) < n_mesmo:
                    mesmo.append(int(j)); vistos.add(t)
                elif len(reserva) < 10:
                    reserva.append(int(j)); vistos.add(t)
            else:
                if len(outros) < n_outros:
                    outros.append(int(j)); vistos.add(t)
                elif len(reserva) < 10:
                    reserva.append(int(j)); vistos.add(t)
            if len(mesmo) == n_mesmo and len(outros) == n_outros:
                break
        saida = mesmo + outros
        saida += reserva[: (n_mesmo + n_outros) - len(saida)]
        return saida

    def _tabela(self, idx: int, indices: list[int]) -> pd.DataFrame:
        d = self.X[indices] - self.X[idx]
        if self.metrica == "cosine":
            from sklearn.metrics.pairwise import cosine_distances
            dist = cosine_distances(self.X[indices], self.X[idx:idx + 1]).ravel()
        else:
            dist = (d ** 2).sum(axis=1) ** 0.5
        t = self.df.loc[indices, ["track_name", "artists", "track_genre", "popularity"]].copy()
        t.columns = ["Música", "Artista", "Gênero", "Popularidade"]
        t["Similaridade (%)"] = [round(self._similaridade(x), 2) for x in dist]
        t.index = range(1, len(t) + 1)
        return t

    def recomendar(self, musica: str, artista: str | None = None, n: int = 10):
        idx = self.buscar(musica, artista)
        if idx is None:
            return None
        return self._tabela(idx, self.recomendar_indices(idx, n))

    def recomendar_hibrido(self, musica: str, artista: str | None = None):
        idx = self.buscar(musica, artista)
        if idx is None:
            return None
        return self._tabela(idx, self.recomendar_hibrido_indices(idx))

    def descrever(self, idx: int) -> str:
        r = self.df.loc[idx]
        return f"{r.track_name} - {r.artists} | Gênero: {r.track_genre}"


if __name__ == "__main__":
    import sys
    caminho = sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset.csv"
    musica = sys.argv[2] if len(sys.argv) > 2 else "Master of Puppets"
    rec = Recomendador.from_csv(caminho, features=FEATURES_CONTINUAS, metrica="euclidean")
    idx = rec.buscar(musica)
    if idx is None:
        print("Música não encontrada.")
    else:
        print("Referência:", rec.descrever(idx), "\n")
        print(rec._tabela(idx, rec.recomendar_hibrido_indices(idx)).to_string())
