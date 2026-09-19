import os
import streamlit as st
from google import genai
from google.genai import types

# Configuração da página web
st.set_page_config(
    page_title="Tutor Vocálico de Alemão - PPGI/UFPel",
    page_icon="🇩🇪",
    layout="centered"
)

# Estilização com as cores da Alemanha
st.markdown("""
    <style>
    .stAppHeader { border-top: 6px solid #000000; }
    h1 { color: #DD0000 !important; font-weight: 700; }
    .stAlert { border-left: 5px solid #FFCC00 !important; }
    .stButton>button { background-color: #DD0000; color: white; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #000000; color: #FFCC00; }
    </style>
""", unsafe_allow_html=True)

st.title("🇩🇪 Tutor Pedagógico de Fonética do Alemão")
st.caption("Ferramenta de apoio ao estudo do sistema vocálico do alemão padrão (PPGI-UFPel)")

# Informação institucional
st.sidebar.info("Projeto pautado nas diretrizes de integridade do CNPq e MEC.")
st.sidebar.markdown("---")
st.sidebar.caption("Assistente alimentado por Gemini 3.5 Flash Lite")

# Leitura automática da API Key
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Pergunte sobre uma vogal ou par mínimo..."):
    if not api_key:
        st.error("Erro de configuração: Chave de API não encontrada nos Secrets do servidor.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        client = genai.Client(api_key=api_key)

        system_instruction = """You are an expert Pedagogical Tutor and Academic Assistant specialized in Standard German Phonetics. Your role is to interactively guide students in understanding the German vowel triangle using a concise, objective, ethical, and bibliographically-grounded approach.

LANGUAGE REQUIREMENT: All interactions with the user MUST be conducted in Portuguese.

CORE FUNCTIONALITY:
1. Phonetic Description: Explain German vowels based on their phonetic parameters: height/openness, anteriority/backness, roundedness, and tenseness. Keep explanations direct, highly objective, descriptive, and concise. Avoid unnecessary conversational filler.
2. Academic Grounding (Brazilian Literature Focus): Whenever explaining a vowel or concept, reference relevant Brazilian research/literature on German phonetics (e.g., ABRALIN, Brazilian university repositories).
3. Interactive Pedagogy: Conclude phonetic responses with a single, short interactive question.

ETHICAL COMPLIANCE & GOVERNANCE (MEC & CNPq DIRECTIVES):
1. Academic Integrity & Authorship (CNPq): Act strictly as a tutor and study copilot. Never write full academic articles/essays.
2. Data Privacy & Safety (MEC/LGPD): Do not request or collect Personally Identifiable Information (PII).
3. Confidentiality (CNPq): Refuse peer-review generation on unpublished manuscripts.
4. Transparency & Bias Prevention (MEC): Maintain a neutral, transparent tone.

RESPONSE FORMAT (in Portuguese):
- Direct and objective phonetic classification (bullet points for parameters).
- Brief, concise pedagogical context (max 2-3 sentences).
- Citation of 1-2 open-access academic sources prioritizing Brazilian researchers.
- Concise interactive follow-up question."""

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            system_instruction=[types.Part.from_text(text=system_instruction)],
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner("Analisando parâmetros fonéticos..."):
                    for chunk in client.models.generate_content_stream(
                        model="gemini-3.5-flash-lite",
                        contents=user_input,
                        config=config,
                    ):
                        if chunk.text:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error("A cota temporária do servidor foi atingida. Aguarde cerca de 1 minuto e tente novamente.")
