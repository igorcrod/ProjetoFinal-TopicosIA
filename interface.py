!pip install gradio pypdf -q

!pip install gradio pypdf openpyxl -q

import gradio as gr
import numpy as np
import plotly.express as px
from pypdf import PdfReader

df_lote_global = pd.DataFrame()

# ==========================================================
# 1. FUNÇÃO DE CLASSIFICAÇÃO INDIVIDUAL (Aba 1)
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

    probs = dnn_model.predict(vetor_entrada, verbose=0)[0]
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
# 2. CLASSIFICAÇÃO EM LOTE + GERADOR DE DASHBOARD (Abas 2 e 3)
# ==========================================================
def classificar_lote_com_analytics(arquivo_lote):
    global df_lote_global

    if arquivo_lote is None:
        return (
            None, "⚠️ Nenhum arquivo foi enviado.", None, {},
            "### ⚠️ Faça o upload de um lote na Aba 2 para gerar o Dashboard.", None, None, None
        )

    caminho = arquivo_lote.name

    try:
        if caminho.endswith('.csv'):
            df = None
            separadores = ['|', ',', ';', '\t']
            for sep in separadores:
                try:
                    temp_df = pd.read_csv(caminho, sep=sep, on_bad_lines='skip', encoding='utf-8', engine='python')
                    if len(temp_df.columns) > 1:
                        df = temp_df
                        break
                except Exception:
                    continue
            if df is None:
                for sep in separadores:
                    try:
                        temp_df = pd.read_csv(caminho, sep=sep, on_bad_lines='skip', encoding='latin1', engine='python')
                        if len(temp_df.columns) > 1:
                            df = temp_df
                            break
                    except Exception:
                        continue

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
            return None, "⚠️ Formato não suportado.", None, {}, "", None, None, None

    except Exception as e:
        return None, f"❌ Erro ao ler arquivo: {str(e)}", None, {}, "", None, None, None

    # Localiza coluna de texto
    colunas_candidatas = [c for c in df.columns if any(p in str(c).lower() for p in ['text', 'texto', 'review', 'reclamacao', 'mensagem', 'comentario', 'descricao'])]
    col_texto = colunas_candidatas[0] if colunas_candidatas else df.columns[-1]

    # Processamento em Bloco via DNN
    df['texto_processado'] = df[col_texto].astype(str).apply(limpar_texto)
    X_lote = vectorizer.transform(df['texto_processado']).toarray().astype(np.float32)

    probs_lote = dnn_model.predict(X_lote, verbose=0)
    idxs_pred = np.argmax(probs_lote, axis=1)

    df['categoria_predita'] = [class_names[i] for i in idxs_pred]
    df['confianca_pct'] = [round(float(probs_lote[row, idxs_pred[row]] * 100), 2) for row in range(len(df))]

    df_resultado = df.drop(columns=['texto_processado'])
    df_lote_global = df_resultado.copy()

    # Exportação para download
    caminho_output = "reclamacoes_classificadas_lote.csv"
    df_resultado.to_csv(caminho_output, index=False, encoding='utf-8-sig')

    # Métricas de Resumo
    total = len(df_resultado)
    contagem_cat = df_resultado['categoria_predita'].value_counts().to_dict()
    graves = contagem_cat.get('RECLAMAÇÃO GRAVE', 0)
    pct_graves = (graves / total * 100) if total > 0 else 0
    conf_media = df_resultado['confianca_pct'].mean()
    top_cat = df_resultado['categoria_predita'].mode()[0] if not df_resultado.empty else "N/A"

    resumo_kpi = (
        f"✅ PROCESSAMENTO EM LOTE CONCLUÍDO!\n"
        f"--------------------------------------------------\n"
        f"• Total de Chamados Analisados: {total}\n"
        f"• Reclamações Graves Identificadas: {graves} ({pct_graves:.1f}%)\n"
        f"• Confiança Médio do Modelo: {conf_media:.2f}%\n"
        f"• Coluna Utilizada como Texto: '{col_texto}'"
    )

    # 📊 GERADORES DE DASHBOARD (PLOTLY)

    # 1. HTML Cards para KPIs
    kpi_cards_html = f"""
    <div style="display: flex; gap: 15px; margin-bottom: 10px;">
        <div style="flex: 1; background-color: #f0f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #2b6cb0; text-align: center;">
            <span style="font-size: 14px; color: #4a5568;">Total de Chamados</span>
            <h2 style="margin: 5px 0; color: #2d3748;">{total}</h2>
        </div>
        <div style="flex: 1; background-color: #fff5f5; padding: 15px; border-radius: 10px; border-left: 5px solid #e53e3e; text-align: center;">
            <span style="font-size: 14px; color: #c53030;">Reclamações Graves</span>
            <h2 style="margin: 5px 0; color: #9b2c2c;">{graves} ({pct_graves:.1f}%)</h2>
        </div>
        <div style="flex: 1; background-color: #f0fff4; padding: 15px; border-radius: 10px; border-left: 5px solid #38a169; text-align: center;">
            <span style="font-size: 14px; color: #276749;">Confiança Média</span>
            <h2 style="margin: 5px 0; color: #22543d;">{conf_media:.1f}%</h2>
        </div>
        <div style="flex: 1; background-color: #feefc3; padding: 15px; border-radius: 10px; border-left: 5px solid #d69e2e; text-align: center;">
            <span style="font-size: 14px; color: #744210;">Categoria Predominante</span>
            <h3 style="margin: 5px 0; color: #744210; font-size: 16px;">{top_cat}</h3>
        </div>
    </div>
    """

    # 2. Gráfico de Pizza (Donut Chart)
    df_chart = df_resultado['categoria_predita'].value_counts().reset_index()
    df_chart.columns = ['Categoria', 'Quantidade']
    fig_pizza = px.pie(
        df_chart, values='Quantidade', names='Categoria', hole=0.4,
        title="Distribuição Percentual por Nível de Insatisfação",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # 3. Gráfico de Barras da Confiança do Modelo
    fig_barras = px.histogram(
        df_resultado, x='confianca_pct', nbins=15,
        title="Distribuição do Nível de Confiança da DNN (%)",
        labels={'confianca_pct': 'Confiança (%)', 'count': 'Frequência'},
        color_discrete_sequence=['#3182ce']
    )

    return df_resultado.head(15), resumo_kpi, caminho_output, contagem_cat, kpi_cards_html, fig_pizza, fig_barras, df_resultado


# ==========================================================
# 3. FUNÇÃO DE FILTRO DE PRIORIDADE (Aba 3)
# ==========================================================
def filtrar_prioridade(categoria_selecionada, df_completo):
    if df_completo is None or len(df_completo) == 0:
        return pd.DataFrame()

    if categoria_selecionada == "TODAS":
        return df_completo
    else:
        return df_completo[df_completo['categoria_predita'] == categoria_selecionada]


# ==========================================================
# 4. CONSTRUÇÃO DA INTERFACE COM 3 ABAS INTEGRADAS
# ==========================================================
with gr.Blocks(theme=gr.themes.Soft(), title="Sistema de Classificação de Reclamações") as demo:
    gr.Markdown(
        """
        # 📋 Sistema de Inteligência para Classificação de Reclamações (PLN + DNN)
        Plataforma corporativa para análise preditiva de insatisfação em atendimentos individuais ou em lote.
        """
    )

    # Estado global para manter o DataFrame entre as abas
    state_df = gr.State(pd.DataFrame())

    with gr.Tabs():
        # ------------------------------------------------------
        # ABA 1: Classificação Individual
        # ------------------------------------------------------
        with gr.TabItem("🔍 Classificação Individual"):
            with gr.Row():
                with gr.Column():
                    input_texto = gr.Textbox(lines=5, placeholder="Digite ou cole a reclamação do cliente...", label="Entrada de Texto")
                    input_arquivo = gr.File(label="Upload de Documento (.pdf ou .txt)", file_types=[".pdf", ".txt"])
                    btn_classificar = gr.Button("🚀 Classificar Reclamação", variant="primary")
                    btn_limpar = gr.ClearButton([input_texto, input_arquivo])

                with gr.Column():
                    output_resumo = gr.Textbox(label="Resultado da Classificação", lines=4)
                    output_chart = gr.Label(label="Probabilidade por Categoria", num_top_classes=5)

            btn_classificar.click(fn=classificar_individual, inputs=[input_texto, input_arquivo], outputs=[output_resumo, output_chart])

        # ------------------------------------------------------
        # ABA 2: Classificação Massiva em Lote
        # ------------------------------------------------------
        with gr.TabItem("📁 Processamento em Lote (CSV/Excel/ZIP)"):
            gr.Markdown("### Upload de Arquivos com Múltiplos Chamados")
            with gr.Row():
                with gr.Column(scale=1):
                    file_lote = gr.File(label="Selecione o Arquivo (.csv, .xlsx, .zip)", file_types=[".csv", ".xlsx", ".xls", ".zip"])
                    btn_lote = gr.Button("⚡ Processar Lote Completo", variant="primary")

                with gr.Column(scale=2):
                    output_kpi_lote = gr.Textbox(label="Resumo da Operação", lines=6)
                    file_download = gr.File(label="📥 Baixar Tabela Classificada (.csv)")

            output_table_lote = gr.Dataframe(label="Pré-visualização dos Primeiros Resultados Classificados", interactive=False)
            output_distrib_lote = gr.Label(label="Distribuição Geral por Categoria no Lote")

        # ------------------------------------------------------
        # ABA 3: Dashboard Analytics & Fila de Prioridades (Funcionalidades 2 e 3)
        # ------------------------------------------------------
        with gr.TabItem("📊 Dashboard Analytics & Priorização"):
            gr.Markdown("### 📈 Painel Gerencial de Monitoramento do Lote")

            output_kpi_cards = gr.HTML(value="<p style='color: #718096;'>Faça o processamento de um lote na <b>Aba 2</b> para visualizar os indicadores do painel.</p>")

            with gr.Row():
                chart_pizza = gr.Plot(label="Gráfico de Distribuição por Categoria")
                chart_barras = gr.Plot(label="Distribuição de Confiança do Modelo")

            gr.Markdown("---")
            gr.Markdown("### 🚨 Fila de Roteamento e Priorização por Categoria")

            with gr.Row():
                dropdown_prioridade = gr.Dropdown(
                    choices=["TODAS", "RECLAMAÇÃO GRAVE", "INSATISFEITO / DEFEITO", "ATENDIMENTO / NEUTRO", "SATISFEITO", "ELOGIO / EXCELENTE"],
                    value="RECLAMAÇÃO GRAVE",
                    label="Filtrar Tabela por Categoria Crítica"
                )

            table_prioridade = gr.Dataframe(label="Tabela Filtrada para Atendimento Prioritário", interactive=False)

            # Evento de filtro ao mudar o Dropdown
            dropdown_prioridade.change(
                fn=filtrar_prioridade,
                inputs=[dropdown_prioridade, state_df],
                outputs=[table_prioridade]
            )

    # Conecta o botão da Aba 2 para alimentar tanto a Aba 2 quanto a Aba 3 (Dashboard)
    btn_lote.click(
        fn=classificar_lote_com_analytics,
        inputs=[file_lote],
        outputs=[
            output_table_lote, output_kpi_lote, file_download, output_distrib_lote,
            output_kpi_cards, chart_pizza, chart_barras, state_df
        ]
    )

demo.launch(share=True, debug=True)