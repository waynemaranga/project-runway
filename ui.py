import streamlit as st
from main import (
    prompt_oss, prompt_kimi, prompt_zai,
    prompt_azure_openai, prompt_cohere
)

# All model keys (used as swipeable "pages")
MODEL_PROMPTERS = {
    "gpt-4o": lambda p: prompt_azure_openai(p, "gpt-4o"),
    "gpt-4.1": lambda p: prompt_azure_openai(p, "gpt-4.1"),
    "o4-mini": lambda p: prompt_azure_openai(p, "o4-mini"),
    "o1": lambda p: prompt_azure_openai(p, "o1"),
    "oss-120b": lambda p: prompt_oss(p, "120B"),
    "oss-20b": lambda p: prompt_oss(p, "20B"),
    "kimi-k2": prompt_kimi,
    "zai-glm-4.5": prompt_zai,
    "command-r7b": lambda p: prompt_cohere(p, "command-r7b"),
    "command-r+": lambda p: prompt_cohere(p, "command-r+"),
    "command-a": lambda p: prompt_cohere(p, "command-a"),
}

# Model order
model_names = list(MODEL_PROMPTERS.keys())

# Streamlit page setup
st.set_page_config(page_title="Project Runway 🧠", layout="centered")
st.title("🧠 Project Runway — Swipe Through Model Minds")

# Session setup
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "model_index" not in st.session_state:
    st.session_state.model_index = 0
if "prompt_submitted" not in st.session_state:
    st.session_state.prompt_submitted = False

# Prompt input
with st.form("prompt-form"):
    prompt = st.text_area("Enter your prompt", height=150)
    submit = st.form_submit_button("🚀 Run Models")

if submit:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        st.session_state.responses = {}
        st.session_state.prompt_submitted = True
        st.session_state.model_index = 0
        with st.spinner("Querying models..."):
            for model in model_names:
                try:
                    result = MODEL_PROMPTERS[model](prompt)
                    st.session_state.responses[model] = result
                except Exception as e:
                    st.session_state.responses[model] = f"⚠️ Error: {e}"

# After prompt is submitted, show swiper interface
if st.session_state.prompt_submitted and st.session_state.responses:
    index = st.session_state.model_index
    current_model = model_names[index]
    response = st.session_state.responses[current_model]

    st.markdown(f"### 🤖 {current_model}")
    st.code(response.strip(), language="markdown")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=(index == 0)):
            st.session_state.model_index -= 1

    with col3:
        if st.button("➡️ Next", disabled=(index == len(model_names) - 1)):
            st.session_state.model_index += 1

    with col2:
        st.markdown(f"<div style='text-align: center;'>Page {index+1} of {len(model_names)}</div>", unsafe_allow_html=True)

    # Optional: Save responses
    if st.button("💾 Save All to Markdown"):
        from pathlib import Path
        import datetime

        out_path = Path(f"runway_output_{datetime.datetime.now():%Y%m%d_%H%M%S}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Prompt: {prompt}\n\n")
            for model, output in st.session_state.responses.items():
                f.write(f"## Model: {model}\n\n")
                f.write(f"```\n{output}\n```\n\n---\n\n")
        st.success(f"Saved output to `{out_path.name}`")
