# Classificação de Reclamações de Consumidores com PLN e Deep Learning

Este repositório contém o projeto final da disciplina de **Tópicos em Inteligência Artificial** (UFRRJ). O objetivo principal é desenvolver um pipeline completo de **Processamento de Linguagem Natural (PLN)** e **Deep Learning** para categorizar feedbacks e reclamações de consumidores em 5 classes de sentimento e gravidade, oferecendo uma interface gráfica interativa via **Gradio** e suporte a execução local via executável compilado.

---

## 📋 Sumário
- [Visão Geral do Projeto](#-visão-geral-do-projeto)
- [Categorias de Classificação](#-categorias-de-classificação)
- [Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [Tecnologias e Bibliotecas Utilizadas](#-tecnologias-e-bibliotecas-utilizadas)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Pré-requisitos de Sistema](#-pré-requisitos-de-sistema)
- [Como Executar o Projeto](#-como-executar-o-projeto)
  - [Opção 1: Executando via Código Fonte (Python)](#opção-1-executando-via-código-fonte-python)
  - [Opção 2: Executando via Arquivo Executável (.exe)](#opção-2-executando-via-arquivo-executável-exe)
- [Como Compilar o Executável (PyInstaller / Auto Py to Exe)](#-como-compilar-o-executável-pyinstaller--auto-py-to-exe)
- [Autores e Créditos](#-autores-e-créditos)

---

## 🎯 Visão Geral do Projeto

Com o crescimento das plataformas digitais de e-commerce e atendimento ao cliente, triar manualmente milhares de reclamações é um processo lento e oneroso. Este sistema automatiza o diagnóstico de feedbacks de clientes, unificando datasets (como o corpus do Buscapé, B2W e RePro), tratando o desbalanceamento de classes através de **pesos ponderados na função de perda** (*class weights*) e gerando estatísticas detalhadas de acurácia, precisão, recall e F1-score.

---

## 🏷️ Categorias de Classificação

O modelo analisa o texto de entrada e classifica o feedback em uma das **5 categorias de sentimento e gravidade**:

1. **RECLAMAÇÃO GRAVE** (Notas 0.0 a 1.0 ou termos críticos)
2. **INSATISFEITO / DEFEITO** (Nota 2.0)
3. **ATENDIMENTO / NEUTRO** (Nota 3.0)
4. **SATISFEITO** (Nota 4.0)
5. **ELOGIO / EXCELENTE** (Nota 5.0)

---

## 🧠 Arquitetura do Pipeline

```text
[Entrada: Texto / Dataset] ➔ [Pré-processamento & Limpeza Regex/NLTK]
                                       │
                                       ▼
                         [Vetorização TF-IDF (1,2 n-grams)]
                                       │
                                       ▼
                   [Rede Neural Densa (TensorFlow / Keras)]
                                       │
                                       ▼
                    [Interface Web Interativa com Gradio]
```

### Detalhamento das Etapas:

1. **Ingestão & Unificação**:
   * Leitura automatizada de múltiplos formatos (`.zip` descompactado dinamicamente, `.csv`, `.tsv`, `.txt`).
   * Padronização de rótulos e colunas entre diferentes fontes de dados.

2. **Limpeza e Normalização Textual**:
   * Remoção de pontuações, números, URLs e caracteres especiais via Expressões Regulares (Regex).
   * Conversão de todo o texto para minúsculas.
   * Remoção de *stopwords* da língua portuguesa (preservando termos de negação cruciais para a polaridade, como *não*, *nao*, *sem*, *nunca*, *nem*, *nada*, *jamais*).
   * Descarte de comentários duplicados e remoção de ruídos textuais com menos de 15 caracteres.

3. **Extração de Atributos**:
   * Vetorização com **TF-IDF (Term Frequency - Inverse Document Frequency)** limitando a 3.000 atributos principais com suporte a *unigrams* e *bigrams*.
   * Cálculo de estatísticas descritivas (média, variância e desvio padrão do tamanho dos textos) e normalização linear Min-Max.

4. **Modelagem & Treinamento**:
   * Arquitetura de Rede Neural Densa (*Deep Neural Network*) implementada em TensorFlow / Keras.
   * Camadas de *Dropout* para mitigação de *overfitting* e callback *EarlyStopping* para interrupção otimizada do treino.
   * Pesos balanceados (`compute_class_weight`) para tratar o desbalanceamento das classes e maximizar o *Recall* nas reclamações graves.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O projeto foi construído em **Python 3.12** utilizando o seguinte ecossistema de bibliotecas:

| Categoria | Biblioteca / Ferramenta | Descrição / Uso |
| :--- | :--- | :--- |
| **Interface Web** | `gradio` | Criação da interface gráfica interativa local para inferências em tempo real. |
| **Deep Learning** | `tensorflow` / `keras` | Construção, compilação e treinamento da rede neural densa. |
| **Processamento de Dados** | `pandas`, `numpy` | Manipulação de DataFrames, matrizes numéricas e unificação de datasets. |
| **PLN & Vetorização** | `nltk`, `scikit-learn` | Remoção de stopwords, vetorização TF-IDF e codificação de rótulos. |
| **Métricas & Avaliação** | `scikit-learn` | Cálculo de Acurácia, Precisão, Recall, F1-Score e matriz de confusão. |
| **Download & Automação** | `gdown`, `zipfile` | Download automatizado via Google Drive e extração de repositórios `.zip`. |
| **Empacotamento** | `pyinstaller`, `auto-py-to-exe` | Compilação do código fonte em executável standalone para Windows. |

---

## 📂 Estrutura do Repositório

```text
ProjetoFinal-TopicosIA/
├── app.py                   # Script principal com pipeline NLP, modelo e interface Gradio
├── corpusbuscape.zip        # Dataset comprimido de avaliações do Buscapé
├── .gitignore               # Regras de exclusão do Git (DLLs e builds de compilação)
└── README.md                # Documentação oficial do repositório
```

---

## 💻 Pré-requisitos de Sistema

1. **Python 3.12** (recomendado para total compatibilidade com TensorFlow e PyInstaller).
2. **Microsoft Visual C++ Redistributable (x64)**: Necessário no Windows para a execução das DLLs nativas em C++ do TensorFlow (`msvcp140.dll` e `msvcp140_1.dll`).

---

## 🚀 Como Executar o Projeto

### Opção 1: Executando via Código Fonte (Python)

1. **Clone este repositório:**
   ```bash
   git clone https://github.com/igorcrod
   cd ProjetoFinal-TopicosIA
   ```

2. **Instale as dependências exigidas:**
   ```bash
   pip install pandas numpy tensorflow gradio nltk scikit-learn gdown openpyxl plotly
   ```

3. **Execute o script principal:**
   ```bash
   python app.py
   ```
   *(Caso possua mais de uma versão do Python instalada no Windows, utilize `py -3.12 app.py`)*.

4. **Acesse a interface:**
   O terminal exibirá o link local do Gradio (geralmente `http://localhost:7860`). O navegador abrirá automaticamente a interface do aplicativo.

---

### Opção 2: Executando via Arquivo Executável (.exe)

1. Acesse a aba **Releases** do repositório no GitHub (`[suspicious link removed]`).
2. Baixe a versão compactada **`app.zip`** (ou o arquivo `app.exe`).
3. Extraia o conteúdo em uma pasta no seu computador.
4. Clique duas vezes no arquivo **`app.exe`**.
5. O terminal abrirá inicializando os serviços e a interface carregará no navegador padrão.

---

## ⚙️ Como Compilar o Executável (PyInstaller / Auto Py to Exe)

Caso faça alterações no código e deseje gerar o `.exe` novamente:

1. Instale o compilador no Python 3.12:
   ```bash
   py -3.12 -m pip install auto-py-to-exe pyinstaller
   ```
2. Abra a interface gráfica de compilação:
   ```bash
   py -3.12 -m auto_py_to_exe
   ```
3. Configure os seguintes parâmetros:
   * **Script Location**: Selecione o arquivo `app.py`.
   * **Option**: One Directory (ou One File).
   * **Aba Advanced**:
     * Em `--collect-all`, adicione: `pandas`, `gradio`, `gdown`.
     * Em `--hidden-import`, adicione: `google`.
4. Clique no botão **Convert .PY to .EXE**.
