import re
import numpy as np
import pdfplumber
from fpdf import FPDF
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import streamlit as st
import io
import requests
from datetime import datetime

# 🔗 URL da API gerada no Google Sheets
URL_GOOGLE_SHEETS = "https://script.google.com/macros/s/AKfycbyTpbWDxWkNRh_ZIlHuAVwZaCC2ODqTmo0Un7ZDbgzrVQBmxlYYKuoYf6yDigAPHZiZ/exec"

# =============================
# 📋 Função para Salvar E-mails no Google Sheets
# =============================
def salvar_email_google_sheets(nome, email, codigo="N/A"):
    dados = {"nome": nome, "email": email, "codigo": codigo}
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(URL_GOOGLE_SHEETS, json=dados, headers=headers)
        if response.text.strip() == "Sucesso":
            st.success("✅ Seus dados foram registrados com sucesso!")
        else:
            st.error(f"❌ Falha ao salvar no Google Sheets: {response.text}")
    except Exception as e:
        st.error(f"❌ Erro na conexão com o Google Sheets: {e}")

# =============================
# 💾 Carregamento do Modelo Roberta (recurso pesado)
# =============================
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Falha ao carregar o modelo Roberta: {e}")
    st.stop()

# =============================
# 🔧 Funções de Análise de Texto
# =============================
@st.cache_data
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def analyze_text_roberta(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prob = torch.softmax(outputs.logits, dim=1)[0, 1].item()
    return prob * 100

def calculate_entropy(text: str) -> float:
    probs = np.array([text.count(c) / len(text) for c in set(text)])
    return -np.sum(probs * np.log2(probs))

def analyze_text(text: str) -> dict:
    clean = preprocess_text(text)
    entropy = calculate_entropy(clean)
    roberta_score = analyze_text_roberta(clean)
    final_score = (roberta_score * 0.7) + (100 * (1 - entropy / 6) * 0.3)
    return {
        'IA (estimada)': f"{final_score:.2f}%",
        'Entropia': f"{entropy:.2f}",
        'Roberta (IA)': f"{roberta_score:.2f}%"
    }

# =============================
# 📄 Funções de PDF (com encoding)
# =============================
def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

class PDFReport(FPDF):
    def _encode(self, txt: str) -> str:
        # substitui en-dash e em-dash por hífen simples
        txt = txt.replace('–', '-').replace('—', '-')
        try:
            return txt.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
            return ''.join(c if ord(c) < 256 else '-' for c in txt)

    def header(self):
        title = self._encode('Relatório TotalIA - PEAS.Co')
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, ln=True, align='C')
        self.ln(5)

    def add_results(self, results: dict):
        self.set_font('Arial', '', 12)
        for k, v in results.items():
            line = f"{k}: {v}"
            self.cell(0, 8, self._encode(line), ln=True)
        self.ln(5)

def generate_pdf_report(results: dict) -> str:
    pdf = PDFReport()
    pdf.add_page()

    # Introdução
    intro = 'Este relatório apresenta uma estimativa sobre a probabilidade de o texto ter sido gerado por IA.'
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, pdf._encode(intro))
    pdf.ln(5)

    # Resultados numéricos
    pdf.add_results(results)

    # Explicação detalhada da Avaliação Roberta
    roberta_value = results['Roberta (IA)']
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, pdf._encode('O que é a "Avaliação Roberta (Confiabilidade IA)"?'), ln=True)
    pdf.ln(2)

    explanation = (
        f"A 'Avaliação Roberta (Confiabilidade IA)' representa a pontuação gerada pelo modelo RoBerta "
        f"para indicar a probabilidade de que um texto tenha sido escrito por IA. "
        f"No seu relatório, o modelo atribuiu {roberta_value}.\n\n"
        "Como funciona o RoBerta:\n"
        "O RoBerta (Robustly optimized BERT approach) é um modelo de NLP da Meta (Facebook AI), treinado "
        "com grandes volumes de texto para análises semânticas profundas.\n\n"
        "Critérios avaliados:\n"
        " - Coesão textual: IA costuma seguir padrões previsíveis.\n"
        " - Uso de conectores: expressões como 'Portanto', 'Além disso' são frequentes.\n"
        " - Frases genéricas: construção sofisticada, porém superficial.\n"
        " - Padrões linguísticos: falta de nuances humanas (ironias, ambiguidade).\n\n"
        "Interpretação do valor:\n"
        "0% - 30%    Baixa probabilidade de IA (provavelmente texto humano)\n"
        "30% - 60%   Área de incerteza (o texto pode conter partes geradas por IA ou apenas seguir um padrão formal)\n"
        "60% - 100%  Alta probabilidade de IA (muito provável que o texto seja gerado por um modelo de linguagem como GPT, Bard, etc.)"
    )
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, pdf._encode(explanation))

    filename = "relatorio_IA.pdf"
    pdf.output(filename, 'F')
    return filename


# =============================
# 🖥️ Interface Streamlit
# =============================
st.title("🔍 TotalIA - Detecção de Texto por IA")
st.write("Faça o upload de um PDF para análise:")

uploaded = st.file_uploader("Escolha um arquivo PDF", type="pdf")
if uploaded:
    texto = extract_text_from_pdf(uploaded)
    resultados = analyze_text(texto)

    st.subheader("🔎 Resultados da Análise")
    for key, val in resultados.items():
        st.write(f"**{key}:** {val}")

    report_path = generate_pdf_report(resultados)
    with open(report_path, "rb") as f:
        st.download_button(
            "📥 Baixar Relatório em PDF",
            f.read(),
            "relatorio_IA.pdf",
            "application/pdf"
        )

# =============================
# 📋 Registro de Usuário (ao final)
# =============================
st.markdown("---")
st.subheader("📋 Cadastre-se para Receber Novidades")
nome = st.text_input("Nome completo", key="nome")
email = st.text_input("E-mail", key="email")
if st.button("Registrar meus dados"):
    if nome and email:
        salvar_email_google_sheets(nome, email)
    else:
        st.warning("⚠️ Preencha ambos os campos antes de registrar.")

# =============================
# 📣 Seção de Propaganda
# =============================
st.markdown("---")
st.subheader("Publicidade - Anuncie Aqui")
st.write("📧 Envie sua proposta para: peas8810@gmail.com")
image_url = "https://via.placeholder.com/728x90.png?text=Anuncie+aqui"
st.image(image_url, use_container_width=True)
st.markdown("### Visite nosso site de parceiros")
st.components.v1.iframe("https://example.com", height=200)
