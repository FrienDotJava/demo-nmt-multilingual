import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Page config
st.set_page_config(
    page_title="NLLB Penerjemah Bahasa Daerah",
    page_icon="🌏",
    layout="centered",
)

# Language metadata
INDONESIAN_CODE = "ind_Latn"

LOCAL_LANGS = {
    "jav_Latn": "Bahasa Jawa",
    "sun_Latn": "Bahasa Sunda",
    "ban_Latn": "Bahasa Bali",
    "min_Latn": "Bahasa Minang",
    "bjn_Latn": "Bahasa Dayak",
}

ALL_LANGS = {
    "ind_Latn": "Bahasa Indonesia",
    **LOCAL_LANGS,
}

# label -> code
LABEL_TO_CODE = {v: k for k, v in ALL_LANGS.items()}

SRC_LABELS = list(ALL_LANGS.values())           # all 6 as source

def get_tgt_labels(src_code: str) -> list:
    """Return valid target labels given a source code."""
    if src_code == INDONESIAN_CODE:
        return list(LOCAL_LANGS.values())        # Indo → one of 5 local
    else:
        return [ALL_LANGS[INDONESIAN_CODE]]      # local → Indonesia only

# CSS
st.markdown("""
<style>
    .main-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e94560, #f5a623);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .app-sub {
        color: #a0aec0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-size: 1.1rem;
        color: #f0f0f0;
        min-height: 80px;
        line-height: 1.7;
        word-break: break-word;
    }
    .lang-badge {
        display: inline-block;
        background: rgba(233,69,96,0.15);
        border: 1px solid rgba(233,69,96,0.4);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.82rem;
        color: #e94560;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #e94560, #f5a623) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: opacity 0.2s;
    }
    .stButton button:hover { opacity: 0.85; }
    .footer-note {
        color: #718096;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Model loading (cached)
MODEL_DIR = os.getenv("MODEL_DIR", "BouncingBubble/NLLB-bidirectional-V2")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device    = 0 if torch.cuda.is_available() else -1
    return tokenizer, model, device

def translate(text: str, src_lang: str, tgt_lang: str,
              tokenizer, model, device: int,
              max_new_tokens: int = 256) -> str:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
    if device >= 0:
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.cuda()
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Header
st.markdown("""
<div class="main-card">
  <div class="app-title">Penerjemah Bahasa Daerah</div>
  <div class="app-sub">
    Terjemahan dua arah antara Bahasa Indonesia dan bahasa daerah Nusantara
    menggunakan model NLLB fine-tuned.
  </div>
</div>
""", unsafe_allow_html=True)

# Session state defaults
if "src_label" not in st.session_state:
    st.session_state.src_label = ALL_LANGS[INDONESIAN_CODE]   # Indonesia
if "tgt_label" not in st.session_state:
    st.session_state.tgt_label = ALL_LANGS["jav_Latn"]        # Jawa
if "result" not in st.session_state:
    st.session_state.result = ""

# Language selectors + swap
col_src, col_swap, col_tgt = st.columns([5, 1, 5])

with col_src:
    new_src_label = st.selectbox(
        "Bahasa Sumber",
        SRC_LABELS,
        index=SRC_LABELS.index(st.session_state.src_label),
        key="src_select",
    )
    # When source changes, reset target to first valid option
    if new_src_label != st.session_state.src_label:
        st.session_state.src_label = new_src_label
        new_src_code = LABEL_TO_CODE[new_src_label]
        st.session_state.tgt_label = get_tgt_labels(new_src_code)[0]
        st.session_state.result = ""
        st.rerun()

with col_swap:
    st.write("")
    st.write("")
    if st.button("⇄"):
        old_src = st.session_state.src_label
        old_tgt = st.session_state.tgt_label
        st.session_state.src_label = old_tgt
        st.session_state.tgt_label = old_src
        st.session_state.src_select = old_tgt
        st.session_state.tgt_select = old_src
        st.session_state.result = ""
        st.rerun()

with col_tgt:
    src_code   = LABEL_TO_CODE[st.session_state.src_label]
    valid_tgts = get_tgt_labels(src_code)

    # Guard: ensure current tgt_label is still valid
    if st.session_state.tgt_label not in valid_tgts:
        st.session_state.tgt_label = valid_tgts[0]

    new_tgt_label = st.selectbox(
        "Bahasa Tujuan",
        valid_tgts,
        index=valid_tgts.index(st.session_state.tgt_label),
        key="tgt_select",
    )
    if new_tgt_label != st.session_state.tgt_label:
        st.session_state.tgt_label = new_tgt_label
        st.session_state.result = ""
        st.rerun()

tgt_code = LABEL_TO_CODE[st.session_state.tgt_label]

# Input area
st.markdown("---")
src_name = ALL_LANGS[src_code].replace("Bahasa ", "")
input_text = st.text_area(
    f"Masukkan teks ({src_name})",
    height=140,
    placeholder=f"Tulis teks {src_name} di sini…",
    key="input_text",
)

col_btn, _ = st.columns([3, 8])
with col_btn:
    translate_btn = st.button("Terjemahkan", use_container_width=True)

# Load model & run translation
if translate_btn:
    if not input_text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Memuat model… (hanya sekali)"):
            tokenizer, model, device = load_model(MODEL_DIR)
        with st.spinner("Menerjemahkan…"):
            try:
                st.session_state.result = translate(
                    input_text.strip(), src_code, tgt_code,
                    tokenizer, model, device
                )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menerjemahkan: {e}")
                st.session_state.result = ""

# Output
if st.session_state.result:
    st.markdown("---")
    st.markdown(
        f'<div class="lang-badge">{st.session_state.tgt_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="result-box">{st.session_state.result}</div>',
        unsafe_allow_html=True,
    )

# Footer
st.markdown("""
<div class="footer-note">
  Didukung oleh model NLLB fine-tuned · Bahasa: Indonesia ↔ Jawa, Sunda, Bali, Minang, Dayak
</div>
""", unsafe_allow_html=True)