# Projeto Aplicado III – Universidade Presbiteriana Mackenzie – 2026

**Sistema de Recomendação de Músicas Baseado em Similaridade de Características de Áudio**

Integrantes: Juan Fiocchi Martins (RA 10732659) · Gustavo de Lima Ribeiro (RA 10424057) · Deborah Silvério Alves Morales (RA 10728563)
Professora: Carolina Toledo Ferraz

## 🎬 Vídeos

| Entrega | Link |
|---|---|
| Apresentação do projeto (5 min) | **[inserir link do YouTube]** |
| Aplicação do projeto na comunidade | **[inserir link do YouTube]** |

## Sobre o projeto

Sistema de recomendação **baseado em conteúdo**: a partir de uma música escolhida pelo usuário, o sistema sugere faixas com som parecido, comparando características de áudio do Spotify (dançabilidade, energia, volume, acústica, positividade, andamento etc.). Não usa histórico de usuários nem popularidade no cálculo, o que permite recomendar artistas pouco conhecidos (ODS 9 e ODS 10).

**Base de dados:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle). Após remover nulos e versões repetidas (mesmo nome + artista), o catálogo final tem **81.206 faixas de 113 gêneros**.

**Modelo final:** normalização Min-Max dos 9 atributos contínuos → K-Nearest Neighbors com distância euclidiana → regra híbrida com **5 faixas do mesmo gênero + 5 de outros gêneros**, sem versões duplicadas.

## Principais resultados (1.000 consultas, Top-10)

| Técnica | Erro acústico (MAE) ↓ | Precisão de gênero@10 | Gêneros distintos | Popularidade média |
|---|---|---|---|---|
| Aleatório (baseline) | 0,222 | 0,9% | 9,6 | 34,8 |
| Mais populares (baseline) | 0,188 | 0,5% | 5,0 | 97,2 |
| KNN cosseno, 11 atributos (Etapa 3) | 0,047 | 9,5% | 7,7 | 34,9 |
| KNN euclidiana, 9 atributos | **0,033** | **10,3%** | 7,5 | 35,1 |
| Híbrido 5+5, cosseno 11 (Etapa 3) | 0,054 | 48,7% | 5,5 | 34,7 |
| **Híbrido 5+5, euclidiana 9 (final)** | **0,039** | 48,8% | 5,4 | 34,7 |

- O KNN deixa as recomendações **4 a 7 vezes mais próximas** do som da referência do que as baselines.
- Trocar o cosseno pela distância euclidiana e retirar `key`/`mode` (categóricos) **reduziu o erro em ~29%**.
- O escore do cosseno **satura** (85% das consultas com o 10º vizinho ≥ 99%), por isso deixou de ser o padrão.
- A popularidade média das recomendações (≈34,7) é igual à do catálogo (34,6): **sem viés de popularidade**.

Tabela completa em [`resultados/metricas_comparativas.csv`](resultados/metricas_comparativas.csv) e gráficos em [`resultados/figuras/`](resultados/figuras/).

## 📂 Estrutura do repositório

```text
├── dataset/
│   └── dataset.csv               # Spotify Tracks Dataset
├── notebooks/
│   └── sistema_recomendacao_v1.ipynb   # EDA e prova de conceito (Etapa 3)
├── src/
│   ├── recomendador.py           # carga, tratamento, modelo KNN e regra híbrida
│   ├── avaliacao.py              # avaliação offline comparando 8 técnicas
│   ├── graficos.py               # gera as figuras da seção Resultados
│   └── app.py                    # aplicação de demonstração (Streamlit)
├── resultados/
│   ├── metricas_comparativas.csv
│   ├── resultados.json
│   ├── log_avaliacao.txt
│   └── figuras/                  # fig2 a fig8 usadas na documentação
├── docs/
│   └── Projeto_Aplicado_III_-_Etapa_4.docx   # documentação final
├── requirements.txt
├── LICENSE
└── README.md
```

## ▶️ Como executar

```bash
pip install -r requirements.txt

# 1) Recomendação pelo terminal
python src/recomendador.py dataset/dataset.csv "Master of Puppets"

# 2) Aplicação web de demonstração
streamlit run src/app.py

# 3) Reproduzir a avaliação e os gráficos (~1 min)
python src/avaliacao.py dataset/dataset.csv resultados
python src/graficos.py dataset/dataset.csv resultados
```

Uso como biblioteca:

```python
from src.recomendador import Recomendador, FEATURES_CONTINUAS
rec = Recomendador.from_csv("dataset/dataset.csv", features=FEATURES_CONTINUAS, metrica="euclidean")
print(rec.recomendar_hibrido("Bohemian Rhapsody", artista="Queen"))
```

## 🛠️ Tecnologias

Python · pandas · NumPy · scikit-learn (`MinMaxScaler`, `NearestNeighbors`) · Matplotlib · Seaborn · Streamlit

## Limitações e trabalhos futuros

Avaliação offline (sem dados de usuários), um gênero por faixa, base estática e restrição da API de *audio features* do Spotify para novas aplicações desde nov/2024. Próximos passos: calibrar o escore exibido, aprender pesos dos atributos com playlists, validar com usuários (testes A/B) e extrair características diretamente do áudio. Detalhes na documentação.
