import gradio as gr
import numpy as np
from pypdf import PdfReader

def classificar_interface(texto_digitado, arquivo_uploaded):
    texto_final = ""
    
    # Verifica se o usuário fez upload de um arquivo
    if arquivo_uploaded is not None:
        caminho_arquivo = arquivo_uploaded.name
        
        # Leitura de PDF
        if caminho_arquivo.endswith('.pdf'):
            reader = PdfReader(caminho_arquivo)
            for page in reader.pages:
                texto_extraido = page.extract_text()
                if texto_extraido:
                    texto_final += texto_extraido + " "
                    
        # Leitura de arquivo TXT
        elif caminho_arquivo.endswith('.txt'):
            with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as f:
                texto_final = f.read()
                
    # Se não houver arquivo, utiliza o texto digitado na caixa
    elif texto_digitado and texto_digitado.strip() != "":
        texto_final = texto_digitado
        
    else:
        return "⚠️ Por favor, digite uma reclamação ou faça upload de um arquivo (PDF/TXT).", {}

    if not texto_final.strip():
        return "⚠️ Não foi possível extrair texto legível do arquivo enviado.", {}

    # Pré-processamento e Vetorização com PLN
    texto_processado = limpar_texto(texto_final)
    vetor_entrada = vectorizer.transform([texto_processado]).toarray().astype(np.float32)

    # Predição com DNN
    probs = dnn_model.predict(vetor_entrada, verbose=0)[0]
    idx_pred = np.argmax(probs)
    categoria_predita = str(class_names[idx_pred])
    confianca_predita = float(probs[idx_pred])

    # Dicionário de probabilidades para o gráfico de barras do Gradio
    dic_probabilidades = {str(class_names[i]): float(probs[i]) for i in range(len(class_names))}

    # Mensagem de resumo formatada
    resumo_texto = (
        f"🎯 Categoria Atribuída: {categoria_predita.upper()}\n"
        f"📊 Nível de Confiança: {confianca_predita * 100:.2f}%\n"
        f"📄 Trecho Analisado: '{texto_final[:120]}...'"
    )

    return resumo_texto, dic_probabilidades


with gr.Blocks(theme=gr.themes.Soft(), title="Sistema de Classificação de Reclamações") as demo:
    gr.Markdown(
        """
        # 📋 Sistema Inteligente de Classificação de Reclamações
        Insira o texto da reclamação ou faça upload de um arquivo (**PDF** ou **TXT**) para identificar o assunto e o nível de insatisfação.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_texto = gr.Textbox(
                lines=5, 
                placeholder="Digite ou cole aqui a reclamação do cliente...", 
                label="Input 1: Digitação Direta"
            )
            input_arquivo = gr.File(
                label="Input 2: Upload de Arquivo (.pdf ou .txt)", 
                file_types=[".pdf", ".txt"]
            )
            btn_classificar = gr.Button("🚀 Classificar Reclamação", variant="primary")
            btn_limpar = gr.ClearButton([input_texto, input_arquivo])

        with gr.Column():
            output_resumo = gr.Textbox(label="Resultado da Classificação", lines=4)
            output_chart = gr.Label(label="Probabilidade por Categoria", num_top_classes=5)

    btn_classificar.click(
        fn=classificar_interface, 
        inputs=[input_texto, input_arquivo], 
        outputs=[output_resumo, output_chart]
    )

demo.launch(share=True, debug=True)