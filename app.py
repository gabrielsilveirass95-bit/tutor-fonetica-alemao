import os
import io
import re
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS

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

# Extrai apenas o conteúdo da tag [ÁUDIO: ...] se ela existir
def extrair_texto_audio(texto):
    match = re.search(r"\[ÁUDIO:\s*(.*?)\]", texto, re.IGNORECASE)
    if match:
        conteudo = match.group(1).strip()
        # Retorna o texto apenas se não estiver vazio ou marcado como NENHUM
        if conteudo and conteudo.upper() != "NENHUM":
            return conteudo
    return None

# Gera o áudio nativo em alemão
def gerar_audio_alemao(texto_alemao):
    try:
        tts = gTTS(text=texto_alemao, lang='de')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio_info" in message and message["audio_info"]:
            st.caption(f"🔊 **Pronúncia dos exemplos:** {message['audio_info']['texto']}")
            st.audio(message["audio_info"]["file"], format="audio/mp3")

if user_input := st.chat_input("Pergunte sobre uma vogal ou par mínimo..."):
    if not api_key:
        st.error("Erro de configuração: Chave de API não encontrada nos Secrets do servidor.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input, "audio_info": None})
        with st.chat_message("user"):
            st.markdown(user_input)

        client = genai.Client(api_key=api_key)

        system_instruction = """You are an expert Pedagogical Tutor and Academic Assistant specialized in Standard German Phonetics. Your role is to interactively guide students in understanding the German vowel triangle using a concise, objective, ethical, and bibliographically-grounded approach.

LANGUAGE REQUIREMENT: All interactions with the user MUST be conducted in Portuguese.

STRICT AUDIO RULE:
- ONLY add an audio tag if the user specifically asks about phonetic concepts, vowels, pronunciation, or German words.
- DO NOT add an audio tag for administrative, general, ethical, or conversational questions (such as privacy questions, greetings, or meta-questions about the app).
- When applicable, write at the very end of your response on a new line: [ÁUDIO: German_Words_Here]
- If no German words are being directly taught or demonstrated, DO NOT include the [ÁUDIO: ...] line at all.

CORE FUNCTIONALITY:
1. Phonetic Description: Explain German vowels based on their phonetic parameters: height/openness, anteriority/backness, roundedness, and tenseness. Keep explanations direct, highly objective, descriptive, and concise.
2. Academic Grounding (Brazilian Literature Focus): Reference relevant Brazilian research/literature on German phonetics (e.g., ABRALIN, Brazilian university repositories).
3. Interactive Pedagogy: Conclude phonetic responses with a single, short interactive question.

ETHICAL COMPLIANCE & GOVERNANCE (MEC & CNPq DIRECTIVES):
1. Academic Integrity & Authorship (CNPq): Act strictly as a tutor and study copilot.
2. Data Privacy & Safety (MEC/LGPD): Do not request or collect Personally Identifiable Information (PII).

RESPONSE FORMAT (in Portuguese):
- Direct and objective phonetic classification (bullet points for parameters).
- Brief, concise pedagogical context with clear German example words.
- Citation of 1-2 open-access academic sources prioritizing Brazilian researchers.
- Concise interactive follow-up question.
- [ÁUDIO: German_Words_Here] (ONLY when demonstrating German vocabulary/phonetics)"""

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            system_instruction=[types.Part.from_text(text=system_instruction)],
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                with st.spinner("Analisando parâmetros..."):
                    for chunk in client.models.generate_content_stream(
                        model="gemini-3.5-flash-lite",
                        contents=user_input,
                        config=config,
                    ):
                        if chunk.text:
                            full_response += chunk.text
                            texto_visivel = re.sub(r"\[ÁUDIO:\s*.*?\]", "", full_response, flags=re.IGNORECASE)
                            response_placeholder.markdown(texto_visivel + "▌")
                
                texto_limpo = re.sub(r"\[ÁUDIO:\s*.*?\]", "", full_response, flags=re.IGNORECASE).strip()
                response_placeholder.markdown(texto_limpo)
                
                texto_para_audio = extrair_texto_audio(full_response)
                
                audio_data = None
                if texto_para_audio:
                    audio_file = gerar_audio_alemao(texto_para_audio)
                    if audio_file:
                        st.caption(f"🔊 **Pronúncia dos exemplos:** {texto_para_audio}")
                        st.audio(audio_file, format="audio/mp3")
                        audio_data = {"texto": texto_para_audio, "file": audio_file}
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": texto_limpo, 
                    "audio_info": audio_data
                })
            
            except Exception:
                st.error("A cota temporária do servidor foi atingida. Aguarde cerca de 1 minuto e tente novamente.")
