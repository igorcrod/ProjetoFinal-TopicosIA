# 📋 Sistema de Classificação Automática de Reclamações (PLN + DNN)

Projeto final desenvolvido para a disciplina **Tópicos Especiais em Inteligência Artificial** (UFRRJ).

O sistema realiza o processamento de texto de avaliações e reclamações de clientes (via digitação direta ou upload de arquivos PDF/TXT), classificando-as automaticamente em **5 categorias de sentimento/gravidade** utilizando **Processamento de Linguagem Natural (PLN)** e uma **Rede Neural Fechada Profunda (DNN)** com interface gráfica interativa em **Gradio**.

---

## 🎯 Categorias de Classificação

O modelo analisa o conteúdo semântico do texto e classifica a entrada em uma das 5 categorias mapeadas:

1. **`RECLAMAÇÃO GRAVE`**: Problemas críticos, cancelamento ou falhas graves no serviço/produto.
2. **`INSATISFEITO / DEFEITO`**: Produtos defeituosos, avarias ou insatisfação com a qualidade.
3. **`ATENDIMENTO / NEUTRO`**: Dúvidas, comentários neutros ou questões operacionais/de suporte.
4. **`SATISFEITO`**: Avaliações positivas com pequenos pontos de atenção.
5. **`ELOGIO / EXCELENTE`**: Experiências altamente positivas e elogios explícitos.

---

## 🛠️ Arquitetura do Pipeline

[ Entrada: Texto / PDF / TXT ] │ ▼ [ 1. Pré-processamento PLN ]  ──► Limpeza regex, Lowercase, NLTK Stopwords (preservando negações) │ ▼ [ 2. Vetorização TF-IDF ]     ──► Unigramas & Bigramas (ngram_range=(1,2), max_features=3000) │ ▼ [ 3. Classificação DNN ]      ──► Dense Layers (256 -> 128 -> 64) + Dropout + Softmax + class_weight │ ▼ [ 4. Interface Gradio ]       ──► Predição em tempo real + Distribuição de Probabilidade

### 1. Processamento de Linguagem Natural (PLN)
* **Higienização Textual**: Remoção de URLs, pontuações, caracteres especiais e números via Expressões Regulares (`re`).
* **Stopwords Personalizadas**: Utilização do corpus NLTK em português com preservação explícita de palavras de negação (ex: *"não"*, *"sem"*, *"nunca"*, *"nem"*), essenciais para a detecção de insatisfação.
* **Extração de Características**: Vetorização **TF-IDF** configurada com **unigramas e bigramas** (`ngram_range=(1,2)`), capturando expressões compostas como *"não recebi"* ou *"atraso entrega"*.

### 2. Redes Neurais e Deep Learning
* **Base Unificada**: Fusão e tratamento de múltiplos datasets reais de e-commerce (`B2W-Reviews01`, `RePro` e `Corpus Buscapé`).
* **Arquitetura DNN**:
  * Camadas Densas com ativação `ReLU` (256, 128 e 64 neurônios).
  * Camadas de `Dropout` (0.3 - 0.4) para prevenção de overfitting.
  * Camada final `Softmax` com 5 neurônios de saída.
* **Balanceamento de Classes**: Aplicação de `class_weight='balanced'` via Scikit-Learn para compensar o desbalanceamento nativo entre avaliações positivas e reclamações graves.
* **Regularização**: Implementação de `EarlyStopping` monitorando a perda de validação (`val_loss`).

### 3. Interface Web Interativa (Gradio)
* **Entrada Dupla**: Suporte a entrada por digitação direta de texto ou upload de arquivos nos formatos `.pdf` (com extração via `pypdf`) e `.txt`.
* **Saída Detalhada**: Exibição da categoria atribuída, percentual de confiança da decisão e gráfico de barras com a distribuição de probabilidades de todas as classes.

---

## 🚀 Como Executar no Google Colab

### 1. Requisitos e Bibliotecas

Instale as dependências necessárias no ambiente:

```bash
pip install tensorflow pandas numpy scikit-learn nltk pypdf gradio
