import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ollama

# --- Configuration ---
st.set_page_config(page_title="Multi-File Data Bot", layout="wide")
st.title("📂 Chat with Multiple JSON Files (Local AI)")

# Sidebar
with st.sidebar:
    st.header("Upload Files")
    # Updated: accept_multiple_files=True
    schema_files = st.file_uploader("Upload Schemas (Optional)", type=["json"], accept_multiple_files=True)
    data_files = st.file_uploader("Upload Data (JSON)", type=["json"], accept_multiple_files=True)
    
    st.success("✅ Connected to Local AI (Qwen)")
    st.info("No internet API required.")

# --- Helper Functions ---
def load_data_registry(uploaded_files):
    registry = {}
    if uploaded_files:
        for f in uploaded_files:
            # Use filename as the key
            registry[f.name] = json.load(f)
    return registry

def get_data_summary(data_registry, schema_registry):
    """
    Creates a text summary of all loaded files for the AI to understand.
    """
    summary = ""
    for name, data in data_registry.items():
        summary += f"\n--- DATASET: {name} ---\n"
        
        # Add Schema info if available
        if name in schema_registry:
            summary += f"Schema: {json.dumps(schema_registry[name])}\n"
        
        # Add Data Sample (First item only to save space)
        if isinstance(data, list) and len(data) > 0:
            summary += f"Sample: {json.dumps(data[:1])}\n"
        elif isinstance(data, dict):
            # Take first 2 keys
            sample = {k: v for k, v in list(data.items())[:2]}
            summary += f"Sample: {json.dumps(sample)}\n"
            
    return summary

def get_local_response(user_query, data_summary):
    """
    Uses Ollama running locally.
    """
    system_prompt = f"""
    You are a Python Data Scientist. 
    You have a dictionary named `datasets` containing multiple JSON files.
    
    The keys of the dictionary are the filenames.
    
    Here is the overview of the loaded data:
    {data_summary}

    Task: Write Python code to answer: "{user_query}"
    
    CRITICAL RULES:
    1. Access data using `datasets['filename.json']`.
    2. To join data, convert them to pandas DataFrames first.
    3. Use `pd.json_normalize()` if data is nested.
    4. Save the final table to `df_result`.
    5. Save the final chart to `fig`.
    6. OUTPUT ONLY CODE. NO MARKDOWN.
    """

    try:
        response = ollama.chat(model='qwen2.5-coder:1.5b', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# --- Main App ---
if data_files:
    # Load all files into dictionaries
    data_registry = load_data_registry(data_files)
    
    # Try to map schemas to data if names match, otherwise just load them
    schema_registry = load_data_registry(schema_files)

    # UI Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Loaded Datasets")
        # Show a tab for each file
        if data_registry:
            tabs = st.tabs(list(data_registry.keys()))
            for i, (name, data) in enumerate(data_registry.items()):
                with tabs[i]:
                    st.json(data if isinstance(data, dict) else data[:1], expanded=False)

    with col2:
        user_input = st.text_area("Ask a question about your files:", height=100, 
                                placeholder="e.g. 'Join users.json and orders.json on user_id and show total spend'")
        
        if st.button("Generate View"):
            if not user_input:
                st.warning("Please type a request.")
            else:
                with st.spinner("Local AI is analyzing multiple files..."):
                    try:
                        # 1. Create Context
                        summary = get_data_summary(data_registry, schema_registry)
                        
                        # 2. Get Code
                        code = get_local_response(user_input, summary)
                        code = code.replace("```python", "").replace("```", "").strip()
                        
                        with st.expander("Show Generated Code"):
                            st.code(code, language='python')

                        # 3. Execute Code
                        # We pass 'datasets' instead of 'data' now
                        local_vars = {"datasets": data_registry, "pd": pd, "px": px, "go": go}
                        exec(code, {}, local_vars)

                        # 4. Show Results
                        if "fig" in local_vars:
                            st.plotly_chart(local_vars["fig"], use_container_width=True)
                        elif "df_result" in local_vars:
                            st.dataframe(local_vars["df_result"], use_container_width=True)
                        else:
                            st.warning("Code ran but produced no 'fig' or 'df_result'.")

                    except Exception as e:
                        st.error(f"Error: {e}")
else:
    st.info("Upload one or more JSON files to start.")
