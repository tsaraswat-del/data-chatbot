import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ollama

# --- Configuration ---
st.set_page_config(page_title="Local AI Data Bot", layout="wide")
st.title("🏠 Chat with Data (Local - No API Key)")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    schema_file = st.file_uploader("Upload Schema (Optional)", type=["json"])
    data_file = st.file_uploader("Upload Data (JSON)", type=["json"])
    
    # Visual confirmation that we are not using Gemini
    st.success("✅ Connected to Local AI (Qwen)")
    st.info("No internet API required.")

def load_json(uploaded_file):
    if uploaded_file is not None:
        return json.load(uploaded_file)
    return None

def get_local_response(user_query, data_sample, schema_structure):
    """
    Uses Ollama running locally in the Codespace.
    """
    system_prompt = f"""
    You are a Python Data Scientist. 
    Dataset schema: {json.dumps(schema_structure)}
    Dataset sample: {json.dumps(data_sample)}

    Task: Write Python code to answer: "{user_query}"
    
    CRITICAL RULES:
    1. Output ONLY valid Python code. NO explanations. NO markdown.
    2. Use `pd.json_normalize(data)` to flatten nested JSON.
    3. Save tables to `df_result`.
    4. Save charts to `fig`.
    5. Assume imports: pandas as pd, plotly.express as px
    """

    try:
        # We use the model we downloaded earlier
        response = ollama.chat(model='qwen2.5-coder:1.5b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# --- Main App ---
if data_file:
    raw_data = load_json(data_file)
    raw_schema = load_json(schema_file) if schema_file else "Infer from data"

    if isinstance(raw_data, list):
        data_sample = raw_data[:1]
    else:
        data_sample = {k: v for k, v in list(raw_data.items())[:2]}

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Preview")
        st.json(data_sample, expanded=False)

    with col2:
        user_input = st.text_area("Ask a question:", height=100)
        
        if st.button("Generate View"):
            if not user_input:
                st.warning("Please type a request.")
            else:
                with st.spinner("Local AI is thinking..."):
                    try:
                        # 1. Get Code
                        code = get_local_response(user_input, data_sample, raw_schema)
                        
                        # Cleanup formatting
                        code = code.replace("```python", "").replace("```", "").strip()
                        
                        with st.expander("Show Generated Code"):
                            st.code(code, language='python')

                        # 2. Execute Code
                        local_vars = {"data": raw_data, "pd": pd, "px": px, "go": go}
                        exec(code, {}, local_vars)

                        # 3. Show Results
                        if "fig" in local_vars:
                            st.plotly_chart(local_vars["fig"], use_container_width=True)
                        elif "df_result" in local_vars:
                            st.dataframe(local_vars["df_result"], use_container_width=True)
                        else:
                            st.warning("Code ran but produced no 'fig' or 'df_result'.")

                    except Exception as e:
                        st.error(f"Error: {e}")
else:
    st.info("Upload JSON data to start.")
