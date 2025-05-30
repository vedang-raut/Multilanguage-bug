# app.py - Enhanced Beautiful UI with Streamlit

import streamlit as st
from backend.config import language_codes
from backend.asr import speech_to_text
from backend.translate import translate_text
from backend.tts import text_to_speech
from backend.audio import record_audio, display_audio
from backend.sentiment import analyze_sentiment
from backend.chatbot import get_chatbot_response

st.set_page_config(
    page_title="Multilingual NLP Assistant", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #666;
        font-weight: 400;
        margin-bottom: 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Feature Selection Buttons - Smaller and Better */
    .stButton > button {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 15px !important;
        padding: 0.8rem 0.5rem !important;
        height: 90px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1) !important;
        backdrop-filter: blur(10px) !important;
        color: white !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        line-height: 1.2 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
        background: rgba(255, 255, 255, 0.25) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Active button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%) !important;
        border-color: #667eea !important;
        color: #667eea !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.25) !important;
        font-weight: 600 !important;
    }
    
    .feature-section-header {
        text-align: center;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .feature-section-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .feature-section-subtitle {
        font-size: 1rem;
        opacity: 0.8;
        font-weight: 400;
    }
    
    /* Form inputs with better contrast */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        color: #333 !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        font-size: 1rem;
        padding: 0.75rem !important;
        color: #333 !important;
    }
    
    .stTextArea > div > div > textarea:focus,
    .stTextInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Labels for form inputs */
    .stSelectbox > label,
    .stTextArea > label,
    .stTextInput > label {
        color: white !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    /* Regular buttons for actions */
    .stButton > button:not([kind]) {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        width: 100% !important;
        height: auto !important;
    }
    
    .stButton > button:not([kind]):hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #5a67d8, #6b46c1) !important;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(240, 147, 251, 0.3);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(168, 237, 234, 0.3);
    }
    
    .metric-card {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    color: white;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

/* Hover effects */
.metric-card:hover {
    transform: translateY(-8px) scale(1.05);
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    border-color: rgba(255, 255, 255, 0.6);
    background: rgba(255, 255, 255, 0.25);
}

.metric-card:hover h3 {
    transform: scale(1.1);
    color: #FFD700 !important;
    text-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
}

.metric-card:hover p {
    color: rgba(255, 255, 255, 1) !important;
    transform: translateY(-2px);
}

/* Continuous spark animation */
.metric-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(
        45deg,
        transparent,
        rgba(255, 255, 255, 0.1),
        transparent,
        rgba(255, 215, 0, 0.2),
        transparent,
        rgba(102, 126, 234, 0.2),
        transparent
    );
    animation: sparkle 3s linear infinite;
    pointer-events: none;
}

/* Floating particles effect */
.metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.3) 1px, transparent 1px),
        radial-gradient(circle at 80% 20%, rgba(255, 215, 0, 0.4) 1px, transparent 1px),
        radial-gradient(circle at 40% 40%, rgba(102, 126, 234, 0.3) 1px, transparent 1px);
    background-size: 30px 30px, 40px 40px, 50px 50px;
    animation: floating-particles 4s ease-in-out infinite;
    pointer-events: none;
    opacity: 0.6;
}

/* Pulse effect on numbers */
.metric-card h3 {
    color: white !important;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    transition: all 0.3s ease;
    animation: number-pulse 2s ease-in-out infinite;
}

.metric-card p {
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 0.9rem;
    margin: 0.25rem 0 0 0;
    transition: all 0.3s ease;
}

/* Keyframe animations */
@keyframes sparkle {
    0% {
        transform: rotate(0deg) translate(-50%, -50%);
        opacity: 0;
    }
    50% {
        opacity: 1;
    }
    100% {
        transform: rotate(360deg) translate(-50%, -50%);
        opacity: 0;
    }
}

@keyframes floating-particles {
    0%, 100% {
        transform: translateY(0px) rotate(0deg);
        opacity: 0.6;
    }
    50% {
        transform: translateY(-10px) rotate(180deg);
        opacity: 0.8;
    }
}

@keyframes number-pulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.02);
    }
}

/* Additional glow effect on hover */
.metric-card:hover::before {
    animation-duration: 1.5s;
    background: linear-gradient(
        45deg,
        transparent,
        rgba(255, 255, 255, 0.3),
        transparent,
        rgba(255, 215, 0, 0.5),
        transparent,
        rgba(102, 126, 234, 0.4),
        transparent
    );
}

/* Shimmer effect */
.metric-card:hover::after {
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(255, 255, 255, 0.6) 2px, transparent 2px),
        radial-gradient(circle at 80% 20%, rgba(255, 215, 0, 0.7) 2px, transparent 2px),
        radial-gradient(circle at 40% 40%, rgba(102, 126, 234, 0.6) 2px, transparent 2px);
    animation-duration: 2s;
}

/* Border glow animation */
.metric-card {
    border-image: linear-gradient(45deg, 
        rgba(255, 255, 255, 0.2), 
        rgba(255, 215, 0, 0.4), 
        rgba(102, 126, 234, 0.4), 
        rgba(255, 255, 255, 0.2)
    ) 1;
    animation: border-glow 3s ease-in-out infinite;
}

@keyframes border-glow {
    0%, 100% {
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    50% {
        box-shadow: 0 4px 25px rgba(102, 126, 234, 0.3);
    }
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
    }
}
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Sidebar text colors */
    .sidebar .stMarkdown {
        color: white !important;
    }
    
    .sidebar h3 {
        color: white !important;
        font-weight: 600;
    }
    
    .sidebar .stSelectbox label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Fix sidebar background */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] > div {
        background: rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Warning and info messages */
    .stAlert > div {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: white !important;
    }
    
    /* Success/Error messages */
    .stSuccess > div {
        background: rgba(72, 187, 120, 0.9) !important;
        color: white !important;
    }
    
    .stError > div {
        background: rgba(245, 101, 101, 0.9) !important;
        color: white !important;
    }
    
    .stWarning > div {
        background: rgba(237, 137, 54, 0.9) !important;
        color: white !important;
    }
    
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .stButton > button {
            height: 80px !important;
            font-size: 0.8rem !important;
            padding: 0.6rem 0.4rem !important;
        }
        
        .feature-section-title {
            font-size: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header fade-in">
        <h1 class="main-title">🌍 Multilingual NLP Assistant</h1>
        <p class="main-subtitle">Powered by AI • Speech Recognition • Translation • Sentiment Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for quick settings
with st.sidebar:
    st.markdown("### ⚙️ Quick Settings")
    st.markdown("---")
    
    # Default languages
    default_src = st.selectbox("Default Source Language", list(language_codes.keys()), key="default_src")
    default_tgt = st.selectbox("Default Target Language", list(language_codes.keys()), index=1, key="default_tgt")
    
    st.markdown("---")
    st.markdown("### 📊 App Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><h3>6</h3><p>Features</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>50+</h3><p>Languages</p></div>', unsafe_allow_html=True)

# Initialize active tab in session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# Enhanced Feature Selection Section
st.markdown("""
    <div class="feature-section-header">
        <h2 class="feature-section-title">Choose Your Feature</h2>
        <p class="feature-section-subtitle">Select from our powerful AI-driven language tools</p>
    </div>
""", unsafe_allow_html=True)

# Create feature buttons with smaller size
tab_names = [
    "🎤\nSpeech-to-Speech",
    "📝\nSpeech-to-Text", 
    "🔊\nText-to-Speech",
    "🌐\nText Translation",
    "😊\nSentiment Analysis",
    "🤖\nAI Chatbot"
]

# Create 2 rows of 3 columns each for better layout
row1_cols = st.columns(3)
row2_cols = st.columns(3)
all_cols = row1_cols + row2_cols

for i, tab_name in enumerate(tab_names):
    with all_cols[i]:
        button_type = "primary" if st.session_state.active_tab == i else "secondary"
        
        if st.button(tab_name, key=f"tab_{i}", type=button_type):
            st.session_state.active_tab = i

st.markdown("---")

# Display content based on active tab
if st.session_state.active_tab == 0:
    # Speech-to-Speech content
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 🎤 Speech-to-Speech Translation")
    st.markdown("Record your voice and get instant translation with audio output")
    
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("🗣️ Source Language", language_codes.keys(), 
                               index=list(language_codes.keys()).index(default_src), key="s2s_src")
    with col2:
        tgt_lang = st.selectbox("🎯 Target Language", language_codes.keys(),
                               index=list(language_codes.keys()).index(default_tgt), key="s2s_tgt")
    
    if st.button("🎙️ Record and Translate", key="s2s_btn"):
        with st.spinner("🎤 Recording audio..."):
            audio = record_audio()
        
        with st.spinner("🔍 Converting speech to text..."):
            text = speech_to_text(audio)
            
        if text:
            st.markdown(f'<div class="info-card"><strong>🗣️ You said:</strong> {text}</div>', 
                       unsafe_allow_html=True)
            
            with st.spinner("🌐 Translating..."):
                translated = translate_text(text, src_lang, tgt_lang)
                
            st.markdown(f'<div class="success-card"><strong>✨ Translation:</strong> {translated}</div>', 
                       unsafe_allow_html=True)
            
            with st.spinner("🔊 Generating audio..."):
                path = text_to_speech(translated, tgt_lang)
                display_audio(path)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 1:
    # Speech-to-Text content
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 📝 Speech-to-Text Transcription")
    st.markdown("Convert your spoken words into written text")
    
    if st.button("🎙️ Record and Transcribe", key="s2t_btn"):
        with st.spinner("🎤 Recording audio..."):
            audio = record_audio()
            
        with st.spinner("🔍 Transcribing..."):
            text = speech_to_text(audio)
            
        if text:
            st.markdown(f'<div class="success-card"><strong>📝 Transcription:</strong> {text}</div>', 
                       unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 2:
    # Text-to-Speech content
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 🔊 Text-to-Speech Synthesis")
    st.markdown("Transform text into natural-sounding speech")
    
    text = st.text_area("✍️ Enter your text", "Hello, welcome to our multilingual assistant!", 
                       height=100, key="tts_text")
    
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("🗣️ Source Language", language_codes.keys(),
                               index=list(language_codes.keys()).index(default_src), key="tts_src")
    with col2:
        tgt_lang = st.selectbox("🎯 Target Language", language_codes.keys(),
                               index=list(language_codes.keys()).index(default_tgt), key="tts_tgt")
    
    if st.button("🎵 Translate and Speak", key="tts_btn"):
        if text.strip():
            with st.spinner("🌐 Translating..."):
                translated = translate_text(text, src_lang, tgt_lang)
                
            st.markdown(f'<div class="success-card"><strong>✨ Translation:</strong> {translated}</div>', 
                       unsafe_allow_html=True)
            
            with st.spinner("🔊 Generating audio..."):
                path = text_to_speech(translated, tgt_lang)
                display_audio(path)
        else:
            st.warning("⚠️ Please enter some text first!")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 3:
    # Text Translation content
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 🌐 Text Translation")
    st.markdown("Translate text between different languages instantly")
    
    text = st.text_area("✍️ Enter text to translate", "Hello, how are you today?", 
                       height=120, key="txt_translate")
    
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("🗣️ From Language", language_codes.keys(),
                               index=list(language_codes.keys()).index(default_src), key="txt_src")
    with col2:
        tgt_lang = st.selectbox("🎯 To Language", language_codes.keys(),
                               index=list(language_codes.keys()).index(default_tgt), key="txt_tgt")
    
    if st.button("🌟 Translate Text", key="txt_btn"):
        if text.strip():
            with st.spinner("🌐 Translating..."):
                translated = translate_text(text, src_lang, tgt_lang)
                
            st.markdown(f'<div class="success-card"><strong>✨ Translation:</strong> {translated}</div>', 
                       unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to translate!")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 4:
    # Sentiment Analysis content
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 😊 Sentiment Analysis")
    st.markdown("Analyze the emotional tone and sentiment of your text")
    
    text = st.text_area("✍️ Enter text for analysis", "I absolutely love this new feature! It's amazing!", 
                       height=100, key="sentiment_text")
    
    if st.button("🔍 Analyze Sentiment", key="sentiment_btn"):
        if text.strip():
            with st.spinner("🧠 Analyzing sentiment..."):
                sentiment = analyze_sentiment(text)
                
            # Create sentiment visualization
            sentiment_emoji = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
            emoji = sentiment_emoji.get(sentiment["label"], "🤔")
            
            confidence_percentage = sentiment['score'] * 100
            
            st.markdown(f'''
                <div class="result-card">
                    <h3>{emoji} Sentiment: {sentiment["label"]}</h3>
                    <p><strong>Confidence:</strong> {confidence_percentage:.1f}%</p>
                    <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 20px; margin-top: 1rem;">
                        <div style="background: rgba(255,255,255,0.8); height: 100%; width: {confidence_percentage}%; border-radius: 10px; transition: width 0.5s ease;"></div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 5:
    # AI Chatbot content
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Chatbot")
    st.markdown("Have a conversation with our intelligent AI assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="info-card"><strong>👤 You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-card"><strong>🤖 Assistant:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Type your message...", placeholder="Ask me anything!", key="chat_input")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            send_button = st.form_submit_button("📤 Send Message")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear Chat")
    
    # Handle form submissions
    if send_button and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 Thinking..."):
            # Use session state for chatbot
            response, _ = get_chatbot_response(user_input, st.session_state)
            
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to show new messages
        st.rerun()
    
    if clear_button:
        st.session_state.messages = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);">
        <p>🌟 Multilingual NLP Assistant • Built by Vedang</p>
    </div>
""", unsafe_allow_html=True)