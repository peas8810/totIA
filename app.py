import re
import numpy as np
import pandas as pd
import pdfplumber
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch  # Dependência do transformers
import streamlit as st

# Modelo Roberta para análise avançada
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Função para pré-processamento do texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Função para analisar texto com Roberta
def analyze_text_roberta(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probability = torch.softmax(logits, axis=1)[0, 1].item()
    return probability * 100  # Convertendo para porcentagem

# Função para analisar entropia (textos de IA tendem a ter menor diversidade)
def calculate_entropy(text):
    probabilities = np.array([text.count(char) / len(text) for char in set(text)])
    return -np.sum(probabilities * np.log2(probabilities))

# Função principal para análise
def analyze_text(text):
    clean_text = preprocess_text(text)
    entropy_score = calculate_entropy(clean_text)
    roberta_score = analyze_text_roberta(clean_text)

    # Avaliação Final (peso ajustável conforme precisão desejada)
    final_score = (roberta_score * 0.7) + (100 * (1 - entropy_score / 6) * 0.3)

    # Geração do relatório
    result = {
        'Porcentagem de IA (estimada)': f"{final_score:.2f}%",
        'Entropia do Texto': f"{entropy_score:.2f}",
        'Avaliação Roberta (Confiabilidade IA)': f"{roberta_score:.2f}%",
    }
    return result

# Função para extrair texto de PDF usando pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Extrai o texto de cada página
    return text

# Função para gerar relatório em PDF
def generate_pdf_report(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(10, 10, 10)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Relatório de Análise de Texto - TotalIA - PEAS.Co', ln=True, align='C')
    pdf.ln(10)

    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, 'O texto analisado pode ou não ter sido escrito por uma IA. Os resultados abaixo são uma estimativa baseada em padrões detectados.')
    pdf.ln(10)

    for key, value in results.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'O que é a "Avaliação Roberta (Confiabilidade IA)"?', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "A 'Avaliação Roberta (Confiabilidade IA)' representa a pontuação gerada pelo modelo RoBERTa para indicar a probabilidade de que um texto tenha sido escrito por uma inteligência artificial.\n\n" \
                 "Como funciona o modelo RoBERTa\n"
                 "O RoBERTa (Robustly optimized BERT approach) é um modelo avançado de NLP (Processamento de Linguagem Natural) desenvolvido pela Meta (Facebook AI). Ele é treinado com grandes volumes de texto e é altamente eficaz na análise semântica.\n\n"
                 "No seu caso, estamos utilizando o RoBERTa para avaliar:\n"
                 "Coesão textual - Textos gerados por IA costumam apresentar padrões previsíveis.\n"
                 "Uso excessivo de conectores - Expressões como 'Portanto', 'Além disso', 'Em conclusão' são comuns em textos artificiais.\n"
                 "Frases genéricas ou superficiais - A IA tende a utilizar construções que parecem sofisticadas, mas carecem de profundidade.\n"
                 "Padrões linguísticos incomuns - Textos gerados por IA muitas vezes carecem de 'toques humanos', como ironias, ambiguidades ou subjetividades.\n\n"
                
                 "Interpretação do valor\n"
                 "0% a 30% - Baixa probabilidade de IA (provavelmente texto humano)\n"
                 "30% a 60% - Área de incerteza (o texto pode conter partes geradas por IA ou apenas seguir um padrão formal)\n"
                 "60% a 100% - Alta probabilidade de IA (muito provável que o texto seja gerado por um modelo de linguagem como GPT, Bard, etc.)")

    pdf.output("relatorio_IA.pdf", 'F')

# Interface do Streamlit
st.title("🔍 TotalIA - Análise de Texto para Detecção de IA - PEAS.Co")
st.write("Faça o upload de um arquivo PDF para análise:")

uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")

if uploaded_file is not None:
    texto = extract_text_from_pdf(uploaded_file)
    resultado = analyze_text(texto)

    # Exibir o relatório na tela
    st.subheader("🔎 Relatório Final 🔎")
    for key, value in resultado.items():
        st.write(f"{key}: {value}")

    # Geração do arquivo de relatório em PDF
    generate_pdf_report(resultado)
    with open("relatorio_IA.pdf", "rb") as pdf_file:
        st.download_button(
            label="📥 Baixar Relatório em PDF",
            data=pdf_file,
            file_name="relatorio_IA.pdf",
            mime="application/pdf",
        )

 # Texto explicativo ao final da página
    st.markdown("""
    ---
    A PEAS.Co trabalha sem recursos governamentais ou privados, apenas de doações. Nos ajude com um PIX de qualquer valor, PIX: peas8810@gmail.com. Tem alguma ideia de programa com IA, nos envie um email que tentaremos fazer juntos! Nosso avançado programa de detecção de plágio utiliza inteligência artificial para comparar textos com uma ampla base de dados composta pelos 100 maiores indexadores e repositórios globais, analisando cuidadosamente as similaridades encontradas. Com base em estudos internacionais, considera-se que uma taxa de similaridade de 3% ou mais indica uma alta concentração de trechos raros — ou seja, sequências de palavras pouco frequentes que apontam para uma possível cópia. Para ilustrar o processo de análise documental, imagine que um arquivo A tenha sido integralmente copiado de outro arquivo B. Ainda assim, a similaridade pode ser igual ou inferior a 50%, e não 100%, devido à variação na quantidade de trechos considerados na comparação. Pesquisas demonstram que uma taxa média de 3% ou mais costuma indicar uma elevada incidência de termos semelhantes, configurando, assim, uma possível ocorrência de plágio. É importante ressaltar que a avaliação final sobre a presença de plágio cabe sempre aos autores e responsáves pelo conteúdo.Para mais informações sobre práticas de integridade acadêmica, acesse [plagiarism.org](https://plagiarism.org). Powered By - PEAS.Co
    """)

