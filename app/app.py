
import io
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

import os
import zipfile
import pandas as pd


"""Para download direto dos datasets de treinamento"""
import gdown

gdown.download(id="1mG117jhV-A_UfE3QGzSAXBAHfSe5SpEe", output="b2w.csv", quiet=False)
gdown.download(id="1rLawuPoSL-NkmWMK-FtdgGyepsF__B5a", output="repro.csv", quiet=False)

nltk.download('stopwords', quiet=True)

id_buscape_drive = "1wsw2DiKeP6vaPO3ywGZ-Jn-eTzkycIHp"

caminhos_locais = [
    "C:/Users/gizelle/OneDrive/Documentos/UFRRJ/TopIA/alunos 2026/igor/Trab3TopicosIA/datasets/corpusbuscape.zip",
    "corpusbuscape.zip",
    "buscape.zip"
]

zip_efetivo = None
for caminho in caminhos_locais:
    if os.path.exists(caminho):
        zip_efetivo = caminho
        break

if zip_efetivo:
    print(f"Arquivo do Buscapé encontrado em: '{zip_efetivo}'")
else:
    print("⏳ Arquivo local não encontrado. Baixando 'corpusbuscape.zip' do drive")
    zip_efetivo = "corpusbuscape.zip"
    gdown.download(id=id_buscape_drive, output=zip_efetivo, quiet=False)

pasta_raiz = "reviews"

if os.path.exists(zip_efetivo):
    with zipfile.ZipFile(zip_efetivo, 'r') as zip_ref:
        zip_ref.extractall(pasta_raiz)
    print(f" Arquivo do Buscapé ('{zip_efetivo}') extraído na pasta '{pasta_raiz}'!")

dados_buscape = []

for root, dirs, files in os.walk(pasta_raiz):
    for file in files:
        if file.endswith(".txt"):
            categoria = os.path.basename(root)
            caminho_completo = os.path.join(root, file)

            with open(caminho_completo, 'r', encoding='utf-8', errors='ignore') as f:
                texto = f.read().strip()
                if len(texto) > 0:
                    dados_buscape.append({'review_text': texto, 'category': categoria})

df_buscape = pd.DataFrame(dados_buscape)
print(f"Total de reclamações do Buscapé carregadas: {len(df_buscape)}")
display(df_buscape.head())

csv_files = [f for f in uploaded.keys() if f.endswith('.csv') or f.endswith('.tsv')]

if not csv_files:
    csv_files = [f for f in uploaded.keys() if f.endswith('.txt') and not f.startswith('reviews_')]

lista_dfs = []

if csv_files:
    print(f"⏳ Arquivos CSV/TSV detectados: {csv_files}\n")

    for arquivo_alvo in csv_files:
        print(f"📄 Lendo arquivo: '{arquivo_alvo}'...")
        conteudo = uploaded[arquivo_alvo]

        primeira_linha = conteudo.decode('utf-8', errors='ignore').split('\n')[0]
        separador = '\t' if '\t' in primeira_linha else ','

        try:
            df_temp = pd.read_csv(io.BytesIO(conteudo), sep=separador, on_bad_lines='skip', encoding='utf-8')
        except Exception:
            df_temp = pd.read_csv(io.BytesIO(conteudo), sep=separador, on_bad_lines='skip', encoding='latin1')

        # Descarta colunas de URL para não contaminar a escolha da coluna de texto
        cols_validas = [c for c in df_temp.columns if not any(ign in str(c).lower() for ign in ['link', 'url', 'href', 'http'])]
        c_texto = 'review_text' if 'review_text' in df_temp.columns else [c for c in cols_validas if 'text' in str(c).lower() or 'review' in str(c).lower()][-1]

        if 'overall_rating' in df_temp.columns:
            c_cat = 'overall_rating'
        elif 'category' in df_temp.columns:
            c_cat = 'category'
        else:
            c_cat = df_temp.columns[-1]

        df_sub = df_temp[[c_texto, c_cat]].dropna().copy()
        df_sub.columns = ['review_text', 'category']
        lista_dfs.append(df_sub)

    df_csv_unificado = pd.concat(lista_dfs, ignore_index=True)
else:
    df_csv_unificado = pd.DataFrame(columns=['review_text', 'category'])

df_completo = pd.concat([df_csv_unificado, df_buscape], ignore_index=True)

# ==========================================================
# 2. ROTULAGEM REFORÇADA DAS 7 CATEGORIAS COM TRAVAS DE SEGURANÇA
# ==========================================================
def rotular_7_categorias(row):
    texto = str(row['review_text']).lower()
    nota = str(row['category']).strip()

    # 🚨 1. TRAVA DE SEGURANÇA: TERMOS EXPLÍCITOS DE RECLAMAÇÃO / INSATISFAÇÃO
    termos_reclamacao_servico_financeiro = [
        'roubou', 'roubo', 'abusiva', 'abusivo', 'indevido', 'indevida', 'irregular',
        'bloquearam', 'bloqueado', 'bloqueio', 'sem sistema', 'sumiu', 'diminuiu',
        'descumprimento', 'sem autorização', 'sem autorizacao', 'cobrança', 'cobranca',
        'cobrado', 'fatura', 'estorno', 'cancelamento', 'cancelar', 'recarga', 'pix',
        'não creditando', 'nao creditando', 'outra empresa', 'sem solução', 'sem solucao',
        'não recebi', 'nao recebi', 'péssimo', 'pessimo', 'errada', 'errado', 'problema',
        'problemas', 'prejuízo', 'prejuizo', 'absurdo', 'vergonha', 'fraude'
    ]

    kw_entrega = [
        'entrega', 'atraso', 'atrasou', 'transportadora', 'prazo', 'demora',
        'demorou', 'rastreio', 'correios', 'chegou', 'envio', 'despacho'
    ]

    kw_defeito = [
        'defeito', 'quebrado', 'trincado', 'parou', 'estragado', 'liga', 'carrega',
        'rasgado', 'amassado', 'ruim', 'danificado', 'aparelho'
    ]

    kw_elogio_expressivo = [
        'excelente', 'ótimo', 'otimo', 'maravilhoso', 'perfeito', 'parabéns', 'parabens',
        'recomendo', 'rápido', 'rapido', 'eficiente', 'atencioso', 'show', 'top', 'nota 10'
    ]

    # SE CONTER QUALQUER TERMO CRÍTICO DE RECLAMAÇÃO -> FORÇA RECLAMAÇÃO
    if any(t in texto for t in termos_reclamacao_servico_financeiro):
        if any(w in texto for w in kw_entrega):
            return 'Reclamação na Entrega do Produto ou Serviço'
        elif any(w in texto for w in kw_defeito):
            return 'Defeito do Produto ou Serviço'
        else:
            return 'Reclamação de Atendimento'

    # AVALIAÇÃO POR NOTA BAIXA (Notas 0, 1 e 2)
    if nota in ['0', '0.0', '1', '1.0', '2', '2.0']:
        if any(w in texto for w in kw_entrega):
            return 'Reclamação na Entrega do Produto ou Serviço'
        elif any(w in texto for w in kw_defeito):
            return 'Defeito do Produto ou Serviço'
        else:
            return 'Reclamação de Atendimento'

    # AVALIAÇÃO NEUTRA (Nota 3)
    elif nota in ['3', '3.0']:
        return 'Neutro'

    # AVALIAÇÃO POSITIVA (Notas 4 e 5) -> EXIGE TERMOS REALMENTE POSITIVOS
    elif nota in ['4', '4.0', '5', '5.0']:
        if any(w in texto for w in kw_elogio_expressivo):
            if any(w in texto for w in kw_entrega):
                return 'Elogio na Entrega'
            elif any(w in texto for w in ['atendimento', 'suporte', 'vendedor', 'atendente', 'loja']):
                return 'Elogio do Atendimento'
            else:
                return 'Elogio da Qualidade do Produto ou Serviço'
        else:
            return 'Neutro'

    return 'Neutro'

# Aplica a nova rotulagem
df_completo['category'] = df_completo.apply(rotular_7_categorias, axis=1)
df_completo = df_completo.dropna(subset=['review_text', 'category']).reset_index(drop=True)

print("=" * 65)
print(f"✅ ROTULAGEM E RE-BALANCEAMENTO CONCLUÍDOS! Total: {len(df_completo)}")
print("=" * 65)
print("\n📌 Distribuição Atualizada das 7 Categorias no Dataset de Treino:")
print(df_completo['category'].value_counts())

# 1. Remoção de registros nulos/ausentes do DataFrame unificado
df_clean = df_completo.dropna(subset=['review_text', 'category']).copy()

# Conversão explícita para string e remoção de espaços nas pontas
df_clean['review_text'] = df_clean['review_text'].astype(str).str.strip()
df_clean['category'] = df_clean['category'].astype(str).str.strip()

# Descarte de valores que viraram texto vazio ou marcadores inválidos
valores_invalidos = ['', 'nan', 'none', 'null', '[]', 'None']
df_clean = df_clean[
    (~df_clean['review_text'].str.lower().isin(valores_invalidos)) &
    (~df_clean['category'].str.lower().isin(valores_invalidos))
]

# 4. Remoção de comentários duplicados no texto da reclamação
df_clean.drop_duplicates(subset=['review_text'], inplace=True)

# 5. Remoção de ruídos (textos extremamente curtos < 15 caracteres)
df_clean = df_clean[df_clean['review_text'].str.len() > 15].reset_index(drop=True)

print(f"Total de registros limpos e validados: {len(df_clean)}")
display(df_clean.head())

stop_words = set(stopwords.words('portuguese'))

negacoes = {'não', 'nao', 'sem', 'nunca', 'nem', 'nada', 'jamais'}
stop_words = stop_words - negacoes

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    palavras = texto.split()

    palavras = [p for p in palavras if p not in stop_words and len(p) > 1]
    return " ".join(palavras)

print("Executando limpeza...")
df_clean['texto_limpo'] = df_clean['review_text'].apply(limpar_texto)
df_clean = df_clean[df_clean['texto_limpo'].str.len() > 5].reset_index(drop=True)
print(f"Limpeza concluída! Total de textos: {len(df_clean)}")

import numpy as np

# Atributo Numérico: Tamanho dos textos (quantidade de palavras)
df_clean['qtd_palavras'] = df_clean['texto_limpo'].apply(lambda x: len(x.split()))

# Estatísticas Descritivas
media_p = df_clean['qtd_palavras'].mean()
var_p = df_clean['qtd_palavras'].var()
std_p = df_clean['qtd_palavras'].std()

# Normalização Linear Min-Max: x_norm = (x - min) / (max - min)
x_min = df_clean['qtd_palavras'].min()
x_max = df_clean['qtd_palavras'].max()
df_clean['qtd_palavras_norm'] = (df_clean['qtd_palavras'] - x_min) / (x_max - x_min)

print("--- ESTATÍSTICAS DE 1ª E 2ª ORDEM (TAMANHO DOS TEXTOS) ---")
print(f"Média (1ª Ordem):     {media_p:.2f} palavras")
print(f"Variância (2ª Ordem): {var_p:.2f}")
print(f"Desvio Padrão:        {std_p:.2f}")
print(f"Intervalo Normalizado: [{df_clean['qtd_palavras_norm'].min():.2f}, {df_clean['qtd_palavras_norm'].max():.2f}]")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Vetorização TF-IDF otimizada para as 7 categorias (Unigramas, Bigramas e Trigramas)
vectorizer = TfidfVectorizer(
    max_features=5000,           # Expandido para 5.000 termos/aspectos
    ngram_range=(1, 3),          # Captura expressões de até 3 palavras
    sublinear_tf=True,           # Escala logarítmica da frequência
    min_df=3,                    # Descarta ruídos extremamente raros
    token_pattern=r'\b\w+\b',
    dtype=np.float32
)

# Transforma os textos limpos em vetores numéricos
X = vectorizer.fit_transform(df_clean['texto_limpo']).astype(np.float32).toarray()

# Mapeia as 7 novas categorias em texto para rótulos numéricos (0 a 6)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_clean['category'])
class_names = list(label_encoder.classes_)

# Divisão estratificada em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"✅ Re-vetorização concluída! Amostras de treino: {X_train.shape[0]} | Atributos por amostra: {X_train.shape[1]}")
print(f"\n📌 Mapeamento Oficial das {len(class_names)} Classes no LabelEncoder:")
for idx, nome in enumerate(class_names):
    print(f"   • Índice {idx}: {nome}")

"""Treinamento"""

import gc
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Limpa a memória e reseta a sessão
gc.collect()
tf.keras.backend.clear_session()
tf.random.set_seed(42)

# Pesos balanceados para compensar desbalanceamento
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

num_classes = len(class_names) # Exatamente 5 classes
input_dim = X_train.shape[1]

# Instancia o modelo DNN do zero
dnn_model = Sequential([
    Input(shape=(input_dim,)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

dnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

print("Treinando a Rede Neural...")
history_dnn = dnn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=35,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

print("Treinamento da rede neural concluído")

import numpy as np
import pandas as pd
from pypdf import PdfReader

def classificar_reclamacao(texto_ou_caminho_pdf):
    if str(texto_ou_caminho_pdf).endswith('.pdf'):
        reader = PdfReader(texto_ou_caminho_pdf)
        texto_bruto = ""
        for page in reader.pages:
            texto_bruto += page.extract_text() + " "
    else:
        texto_bruto = str(texto_ou_caminho_pdf)

    texto_processado = limpar_texto(texto_bruto)
    vetor_entrada = vectorizer.transform([texto_processado]).toarray()

    probs = dnn_model.predict(vetor_entrada, verbose=0)[0]
    idx_pred = np.argmax(probs)
    categoria_predita = str(class_names[idx_pred])
    confianca = float(probs[idx_pred] * 100)

    print("=" * 65)
    print("📋 Classificador de Reclamações")
    print("=" * 65)
    print(f"📄 Texto Analisado: '{texto_bruto[:90]}...'")
    print(f"🎯 Categoria Atribuída:      {categoria_predita.upper()}")
    print(f"📊 Nível de Confiança:        {confianca:.2f}%")
    print("-" * 65)
    print("📈 Probabilidades por Categoria:")

    # Exibe o percentual de cada categoria mapeada no modelo
    for idx, prob in enumerate(probs):
        nome_cat = class_names[idx]
        barra = "█" * int(prob * 20) # Barra gráfica para visualização
        print(f"  • {str(nome_cat).ljust(25)}: {prob*100:6.2f}% {barra}")
    print("=" * 65)

    return categoria_predita, confianca

# Exemplo para Teste
categoria_resultado, nivel_confianca = classificar_reclamacao(
    "Minha encomenda está atrasada há mais de duas semanas e a transportadora não atualiza o rastreio!"
)

"""Protótipo de Interface"""

import gradio as gr
import pandas as pd
import numpy as np
import zipfile
import io
import os
import plotly.express as px
from pypdf import PdfReader

# ==========================================================
# 1. CLASSIFICAÇÃO INDIVIDUAL (Aba 1)
# ==========================================================
def classificar_individual(texto_digitado, arquivo_uploaded):
    texto_final = ""

    if arquivo_uploaded is not None:
        caminho_arquivo = arquivo_uploaded.name
        if caminho_arquivo.endswith('.pdf'):
            reader = PdfReader(caminho_arquivo)
            for page in reader.pages:
                texto_extraido = page.extract_text()
                if texto_extraido:
                    texto_final += texto_extraido + " "
        elif caminho_arquivo.endswith('.txt'):
            with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as f:
                texto_final = f.read()
    elif texto_digitado and texto_digitado.strip() != "":
        texto_final = texto_digitado
    else:
        return "⚠️ Por favor, digite uma reclamação ou faça upload de um arquivo (PDF/TXT).", {}

    if not texto_final.strip():
        return "⚠️ Não foi possível extrair texto legível.", {}

    texto_processado = limpar_texto(texto_final)
    vetor_entrada = vectorizer.transform([texto_processado]).toarray().astype(np.float32)

    probs = dnn_model.predict(vetor_entrada, verbose=0)
    idx_pred = np.argmax(probs)
    categoria_predita = str(class_names[idx_pred])
    confianca_predita = float(probs[idx_pred])

    dic_probabilidades = {str(class_names[i]): float(probs[i]) for i in range(len(class_names))}

    resumo_texto = (
        f"🎯 Categoria Atribuída: {categoria_predita.upper()}\n"
        f"📊 Nível de Confiança: {confianca_predita * 100:.2f}%\n"
        f"📄 Trecho Analisado: '{texto_final[:120]}...'"
    )

    return resumo_texto, dic_probabilidades


# ==========================================================
# 2. CLASSIFICAÇÃO EM LOTE + ANALYTICS (Abas 2 e 3)
# ==========================================================
def classificar_lote_com_analytics(arquivo_lote, prioridade_selecionada):
    if arquivo_lote is None:
        return (
            None, "⚠️ Nenhum arquivo foi enviado.", None, {},
            "### ⚠️ Faça o upload de um lote na Aba 2 para gerar o Dashboard.", None, None, pd.DataFrame(), pd.DataFrame()
        )

    caminho = arquivo_lote.name

    try:
        if caminho.endswith('.csv'):
            df = None
            separadores = ['|', ',', ';', '\t']

            for sep in separadores:
                for enc in ['utf-8', 'latin1']:
                    try:
                        temp_df = pd.read_csv(caminho, sep=sep, on_bad_lines='skip', encoding=enc, engine='python')
                        if len(temp_df.columns) > 1:
                            if any('http' in str(c) or len(str(c)) > 30 for c in temp_df.columns):
                                temp_df = pd.read_csv(caminho, sep=sep, header=None, on_bad_lines='skip', encoding=enc, engine='python')
                            df = temp_df
                            break
                    except Exception:
                        continue
                if df is not None:
                    break

            if df is None:
                return None, "⚠️ Não foi possível identificar a estrutura do CSV.", None, {}, "", None, None, pd.DataFrame(), pd.DataFrame()

        elif caminho.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(caminho)

        elif caminho.endswith('.zip'):
            dados_zip = []
            with zipfile.ZipFile(caminho, 'r') as z:
                for filename in z.namelist():
                    if filename.endswith('.txt') and not filename.startswith('__MACOSX'):
                        with z.open(filename) as f:
                            conteudo = f.read().decode('utf-8', errors='ignore').strip()
                            if conteudo:
                                dados_zip.append({'arquivo': filename, 'texto_reclamacao': conteudo})
            df = pd.DataFrame(dados_zip)
        else:
            return None, "⚠️ Formato não suportado.", None, {}, "", None, None, pd.DataFrame(), pd.DataFrame()

    except Exception as e:
        return None, f"❌ Erro ao ler arquivo: {str(e)}", None, {}, "", None, None, pd.DataFrame(), pd.DataFrame()

    # 🚨 SELEÇÃO INTELIGENTE DE COLUNA: Bloqueia colunas de Link/URL/ID
    colunas_ignoradas = ['link', 'url', 'href', 'http', 'id', 'data', 'date']

    colunas_candidatas = [
        c for c in df.columns
        if any(p in str(c).lower() for p in ['text', 'texto', 'review', 'reclamacao', 'mensagem', 'comentario', 'descricao', 'detalhes', 'titulo', 'reclamacao'])
        and not any(ign in str(c).lower() for ign in colunas_ignoradas)
    ]

    if colunas_candidatas:
        col_texto = colunas_candidatas[0]
    else:
        # Se nenhuma coluna tiver nome padrão, pega a coluna com maior texto descartando links HTTP
        colunas_validas = [c for c in df.columns if not any(ign in str(c).lower() for ign in colunas_ignoradas)]
        if colunas_validas:
            col_texto = max(colunas_validas, key=lambda c: df[c].astype(str).apply(lambda x: 0 if 'http' in x else len(x)).mean())
        else:
            col_texto = df.columns[-1]

    # Processamento em Lote
    df['texto_processado'] = df[col_texto].astype(str).apply(limpar_texto)
    X_lote = vectorizer.transform(df['texto_processado']).toarray().astype(np.float32)

    probs_lote = dnn_model.predict(X_lote, verbose=0)
    idxs_pred = np.argmax(probs_lote, axis=1)

    df['categoria_predita'] = [class_names[i] for i in idxs_pred]
    df['confianca_pct'] = [round(float(probs_lote[row, idxs_pred[row]] * 100), 2) for row in range(len(df))]

    df_resultado = df.drop(columns=['texto_processado'])

    caminho_output = "reclamacoes_classificadas_lote.csv"
    df_resultado.to_csv(caminho_output, index=False, encoding='utf-8-sig')

    total = len(df_resultado)
    contagem_cat = df_resultado['categoria_predita'].value_counts().to_dict()

    reclamacoes_totais = (
        contagem_cat.get('Reclamação na Entrega do Produto ou Serviço', 0) +
        contagem_cat.get('Reclamação de Atendimento', 0) +
        contagem_cat.get('Defeito do Produto ou Serviço', 0)
    )
    pct_reclamacoes = (reclamacoes_totais / total * 100) if total > 0 else 0
    conf_media = df_resultado['confianca_pct'].mean() if total > 0 else 0
    top_cat = df_resultado['categoria_predita'].mode() if not df_resultado.empty else "N/A"

    kpi_cards_html = f"""
    <div style="display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 180px; background-color: #f0f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #2b6cb0; text-align: center;">
            <span style="font-size: 13px; color: #4a5568;">Total de Chamados</span>
            <h2 style="margin: 5px 0; color: #2d3748;">{total}</h2>
        </div>
        <div style="flex: 1; min-width: 180px; background-color: #fff5f5; padding: 15px; border-radius: 10px; border-left: 5px solid #e53e3e; text-align: center;">
            <span style="font-size: 13px; color: #c53030;">Total de Reclamações/Defeitos</span>
            <h2 style="margin: 5px 0; color: #9b2c2c;">{reclamacoes_totais} ({pct_reclamacoes:.1f}%)</h2>
        </div>
        <div style="flex: 1; min-width: 180px; background-color: #f0fff4; padding: 15px; border-radius: 10px; border-left: 5px solid #38a169; text-align: center;">
            <span style="font-size: 13px; color: #276749;">Confiança Média</span>
            <h2 style="margin: 5px 0; color: #22543d;">{conf_media:.1f}%</h2>
        </div>
        <div style="flex: 1; min-width: 180px; background-color: #feefc3; padding: 15px; border-radius: 10px; border-left: 5px solid #d69e2e; text-align: center;">
            <span style="font-size: 13px; color: #744210;">Categoria Predominante</span>
            <h3 style="margin: 5px 0; color: #744210; font-size: 14px;">{top_cat}</h3>
        </div>
    </div>
    """

    resumo_kpi = (
        f"✅ PROCESSAMENTO EM LOTE CONCLUÍDO!\n"
        f"--------------------------------------------------\n"
        f"• Total de Chamados Analisados: {total}\n"
        f"• Reclamações/Defeitos Mapeados: {reclamacoes_totais} ({pct_reclamacoes:.1f}%)\n"
        f"• Nível Médio de Confiança da DNN: {conf_media:.2f}%\n"
        f"• Coluna Utilizada como Texto: '{col_texto}'"
    )

    df_chart = df_resultado['categoria_predita'].value_counts().reset_index()
    df_chart.columns = ['Categoria', 'Quantidade']

    fig_pizza = px.pie(
        df_chart, values='Quantidade', names='Categoria', hole=0.4,
        title="Distribuição Percentual pelas 7 Categorias",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pizza.update_layout(margin=dict(t=40, b=20, l=20, r=20))

    fig_barras = px.bar(
        df_chart, x='Categoria', y='Quantidade', color='Categoria',
        title="Quantidade Absoluta por Categoria",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_barras.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))

    df_prioridade = filtrar_prioridade(prioridade_selecionada, df_resultado)

    return (
        df_resultado.head(15), resumo_kpi, caminho_output, contagem_cat,
        kpi_cards_html, fig_pizza, fig_barras, df_resultado, df_prioridade
    )


# ==========================================================
# 3. FILTRO DE PRIORIDADE (Aba 3)
# ==========================================================
def filtrar_prioridade(categoria_selecionada, df_completo):
    if df_completo is None or len(df_completo) == 0:
        return pd.DataFrame()

    if categoria_selecionada == "TODAS":
        return df_completo
    else:
        return df_completo[df_completo['categoria_predita'] == categoria_selecionada]


# ==========================================================
# 4. INTERFACE GRADIO
# ==========================================================
opcoes_7_categorias = [
    "TODAS",
    "Reclamação na Entrega do Produto ou Serviço",
    "Reclamação de Atendimento",
    "Defeito do Produto ou Serviço",
    "Elogio na Entrega",
    "Elogio do Atendimento",
    "Elogio da Qualidade do Produto ou Serviço",
    "Neutro"
]

with gr.Blocks(theme=gr.themes.Soft(), title="Sistema de Classificação de Reclamações") as demo:
    gr.Markdown(
        """
        # 📋 Sistema Inteligente de Classificação de Reclamações (PLN + DNN)
        Análise de sentimento e aspectos (Entrega, Atendimento, Defeito, Elogios e Neutro).
        """
    )

    state_df = gr.State(pd.DataFrame())

    with gr.Tabs():
        # ABA 1
        with gr.TabItem("🔍 Classificação Individual"):
            with gr.Row():
                with gr.Column():
                    input_texto = gr.Textbox(lines=5, placeholder="Digite ou cole a reclamação do cliente...", label="Entrada de Texto")
                    input_arquivo = gr.File(label="Upload de Documento (.pdf ou .txt)", file_types=[".pdf", ".txt"])
                    btn_classificar = gr.Button("🚀 Classificar Reclamação", variant="primary")
                    btn_limpar = gr.ClearButton([input_texto, input_arquivo])

                with gr.Column():
                    output_resumo = gr.Textbox(label="Resultado da Classificação", lines=4)
                    output_chart = gr.Label(label="Probabilidade por Categoria", num_top_classes=7)

            btn_classificar.click(fn=classificar_individual, inputs=[input_texto, input_arquivo], outputs=[output_resumo, output_chart])

        # ABA 2
        with gr.TabItem("📁 Processamento em Lote"):
            gr.Markdown("### Upload de Arquivos com Múltiplos Chamados (.csv, .xlsx, .zip)")
            with gr.Row():
                with gr.Column(scale=1):
                    file_lote = gr.File(label="Selecione o Arquivo", file_types=[".csv", ".xlsx", ".xls", ".zip"])
                    btn_lote = gr.Button("⚡ Processar Lote Completo", variant="primary")

                with gr.Column(scale=2):
                    output_kpi_lote = gr.Textbox(label="Resumo da Operação", lines=6)
                    file_download = gr.File(label="📥 Baixar Tabela Classificada (.csv)")

            output_table_lote = gr.Dataframe(label="Pré-visualização dos Resultados Classificados", interactive=False)
            output_distrib_lote = gr.Label(label="Distribuição Geral por Categoria no Lote")

        # ABA 3
        with gr.TabItem("📊 Dashboard Analytics & Priorização"):
            gr.Markdown("### 📈 Painel Gerencial de Monitoramento")

            output_kpi_cards = gr.HTML(value="<p style='color: #718096;'>Faça o processamento de um lote na <b>Aba 2</b> para gerar os indicadores do painel.</p>")

            with gr.Row():
                chart_pizza = gr.Plot(label="Distribuição Percentual")
                chart_barras = gr.Plot(label="Quantidade Absoluta")

            gr.Markdown("---")
            gr.Markdown("### 🚨 Fila de Roteamento e Priorização por Categoria")

            dropdown_prioridade = gr.Dropdown(
                choices=opcoes_7_categorias,
                value="Reclamação de Atendimento",
                label="Filtrar Tabela por Categoria Específica"
            )

            table_prioridade = gr.Dataframe(label="Tabela Filtrada para Atendimento Prioritário", interactive=False)

            dropdown_prioridade.change(
                fn=filtrar_prioridade,
                inputs=[dropdown_prioridade, state_df],
                outputs=[table_prioridade]
            )

    btn_lote.click(
        fn=classificar_lote_com_analytics,
        inputs=[file_lote, dropdown_prioridade],
        outputs=[
            output_table_lote, output_kpi_lote, file_download, output_distrib_lote,
            output_kpi_cards, chart_pizza, chart_barras, state_df, table_prioridade
        ]
    )

demo.launch(share=True, debug=True)