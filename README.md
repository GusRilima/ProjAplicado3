# Projeto Aplicado III – Universidade Presbiteriana Mackenzie – 2026

Sistema de Recomendação de Músicas Baseado em Similaridade de Características de Áudio

##  Sobre o Projeto

Os sistemas de recomendação estão cada vez mais presentes nas plataformas digitais, auxiliando os usuários na descoberta de novos conteúdos. Em plataformas de streaming musical, como o Spotify, existem milhões de faixas, o que torna praticamente impossível para o usuário conhecer e explorar todas as opções disponíveis sem o auxílio de recomendações automáticas.

Este projeto propõe o desenvolvimento de um sistema de recomendação de músicas baseado exclusivamente na similaridade entre características de áudio, como *dançabilidade, energia, valência e acusticidade*. A partir de uma música de referência informada pelo usuário, o sistema é capaz de identificar e sugerir outras faixas com características sonoras semelhantes, utilizando uma abordagem baseada em conteúdo que não depende do histórico prévio de interações do usuário.

O projeto também está alinhado aos Objetivos de Desenvolvimento Sustentável (ODS), especificamente o ODS 9 (Indústria, Inovação e Infraestrutura) e ODS 10 (Redução das Desigualdades), ao diversificar a descoberta de artistas além daqueles de grande popularidade.

##  Objetivos

**Objetivo Geral**
Desenvolver um sistema de recomendação de músicas capaz de identificar e sugerir faixas semelhantes a partir de uma música de referência escolhida pelo usuário, utilizando a análise e a comparação de características de áudio presentes nas músicas.

**Objetivos Específicos**
- **Tratamento de Dados:** Realizar a coleta e o pré-processamento do *Spotify Tracks Dataset*, incluindo tratamento de valores ausentes, duplicados e normalização das variáveis numéricas.
- **Análise Exploratória (EDA):** Analisar as principais características de áudio presentes, observando sua distribuição e diferenças entre os gêneros musicais.
- **Modelagem:** Desenvolver abordagens de recomendação baseadas em conteúdo utilizando métodos de similaridade (como Similaridade do Cosseno) e algoritmos como K-vizinhos mais próximos (KNN) e agrupamento (K-Means).
- **Avaliação:** Medir a qualidade das recomendações geradas, considerando a similaridade com a música de referência, a diversidade das sugestões e a cobertura de gêneros musicais.
- **Documentação:** Registrar todas as etapas do pipeline, os resultados e as limitações encontradas.

##  Base de Dados e Pipeline

**Dataset Utilizado:** *Spotify Tracks Dataset* (via Kaggle)
- **Volume:** ~114 mil faixas musicais e 125 gêneros.
- **Features Analisadas:** Popularidade, duração, e características de áudio extraídas da API do Spotify (dançabilidade, energia, tom, volume, modo, fala, acusticidade, instrumentalidade, presença de audiência, valência, andamento e assinatura de tempo).

**Visão Geral do Pipeline:**
1. **Coleta e Importação:** Carregamento dos dados no formato tabular (DataFrames).
2. **Pré-processamento:** Limpeza e padronização dos dados.
3. **Análise Exploratória:** Investigação estatística das *features* de áudio.
4. **Modelagem:** Cálculo do grau de similaridade entre as faixas.
5. **Geração das Recomendações:** Retorno de um conjunto de músicas similares à entrada do usuário.
6. **Avaliação dos Resultados:** Análise de coerência e diversidade da saída.

## 🛠️ Tecnologias e Ferramentas

As principais tecnologias utilizadas para o desenvolvimento desta solução incluem:

- **Linguagem:** Python
- **Análise e Manipulação de Dados:** Pandas
- **Visualização de Dados:** Matplotlib, Seaborn
- **Machine Learning (Similaridade e Clusterização):** Scikit-learn
- **Ambiente de Desenvolvimento:** VS Code / Jupyter Notebooks

## 📂 Estrutura do Repositório

```text
├── dataset/            # Diretório contendo o Spotify Tracks Dataset utilizado
├── notebooks/          # Notebooks contendo EDA, pré-processamento e testes de modelos
├── src/                # Códigos-fonte do sistema de recomendação de músicas
├── LICENSE             # Arquivo de licença do projeto (MIT)
└── README.md           # Documentação principal do projeto
