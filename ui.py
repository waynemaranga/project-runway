import streamlit as st
import time
from datetime import datetime
from main import (
    prompt_oss, prompt_kimi, prompt_zai,
    prompt_azure_openai, prompt_cohere
)
from extras import prompt_azure_extra


# Simple model list
MODELS = {
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
    "jais-30b": lambda p: prompt_azure_extra(p, "jais-30b"),
    "phi-4": lambda p: prompt_azure_extra(p, "phi-4"),
    "llama-4": lambda p: prompt_azure_extra(p, "llama-4"),
    "grok-3": lambda p: prompt_azure_extra(p, "grok-3"),
    "deepseek-r1": lambda p: prompt_azure_extra(p, "deepseek-r1"),
    "mai-ds-r1": lambda p: prompt_azure_extra(p, "mai-ds-r1"),
}

# Page setup
st.set_page_config(page_title="Project Runway 🧠", layout="wide")
st.title("🧠 Project Runway — AI Model Comparison")

# Initialize session state
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "current_model_index" not in st.session_state:
    st.session_state.current_model_index = 0

# Sidebar - Model Selection
st.sidebar.header("Select Models")
selected_models = []
for model_name in MODELS.keys():
    if st.sidebar.checkbox(model_name, value=True):
        selected_models.append(model_name)

# Main input
prompt = st.text_area("Enter your prompt:", height=100)

# Run button
if st.button("🚀 Run Models", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt")
    elif not selected_models:
        st.warning("Please select at least one model")
    else:
        st.session_state.responses = {}
        
        # Simple progress bar
        progress = st.progress(0)
        status = st.empty()
        
        # Run models one by one
        for i, model_name in enumerate(selected_models):
            status.text(f"Running {model_name}...")
            
            try:
                start_time = time.time()
                response = MODELS[model_name](prompt)
                response_time = time.time() - start_time
                
                st.session_state.responses[model_name] = {
                    "text": response,
                    "time": response_time
                }
            except Exception as e:
                st.session_state.responses[model_name] = {
                    "text": f"❌ Error: {str(e)}",
                    "time": 0
                }
            
            progress.progress((i + 1) / len(selected_models))
        
        progress.empty()
        status.empty()
        st.success(f"✅ Done! Got {len(selected_models)} responses")

# Display results
if st.session_state.responses:
    st.divider()
    
    # View selector
    view_mode = st.radio("View:", ["One at a time", "All together"], horizontal=True)
    
    if view_mode == "One at a time":
        # Swipe view
        model_names = list(st.session_state.responses.keys())
        current_index = st.session_state.current_model_index
        
        if current_index >= len(model_names):
            st.session_state.current_model_index = 0
            current_index = 0
        
        current_model = model_names[current_index]
        response_data = st.session_state.responses[current_model]
        
        # Show current model
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous") and current_index > 0:
                st.session_state.current_model_index -= 1
                st.rerun()
        
        with col2:
            st.markdown(f"### {current_model}")
            st.caption(f"Response time: {response_data['time']:.2f}s")
        
        with col3:
            if st.button("➡️ Next") and current_index < len(model_names) - 1:
                st.session_state.current_model_index += 1
                st.rerun()
        
        # Show response
        if response_data["text"].startswith("❌"):
            st.error(response_data["text"])
        else:
            st.write(response_data["text"])
        
        # Page indicator
        st.caption(f"Page {current_index + 1} of {len(model_names)}")
    
    else:
        # Show all responses
        for model_name, response_data in st.session_state.responses.items():
            with st.expander(f"{model_name} ({response_data['time']:.2f}s)", expanded=True):
                if response_data["text"].startswith("❌"):
                    st.error(response_data["text"])
                else:
                    st.write(response_data["text"])
    
    # Simple export
    if st.button("💾 Save Results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for model, data in st.session_state.responses.items():
                f.write(f"=== {model} ===\n")
                f.write(f"Time: {data['time']:.2f}s\n")
                f.write(f"Response: {data['text']}\n\n")
        
        st.success(f"Saved to {filename}")