import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ollama
import os
import glob

# --- Configuration ---
st.set_page_config(page_title="Auto-Detect Data Bot", layout="wide")
st.title("📂 Chat with Local JSON Files")

# Sidebar
with st.sidebar:
    st.header("Data Source")
    st.info("ℹ️ Put your .json files in the file explorer on the left.")
    
    if st.button("🔄 Refresh File List"):
        st.rerun()

    st.success("✅ Connected to Local AI")

# --- Helper Functions ---
def get_json_files():
    """Scans the current directory for JSON files, excluding system files."""
    # Find all .json files
    all_files = glob.glob("*.json")
    
    # Exclude system/config files
    ignore_list = ["package.json", "package-lock.json", "tsconfig.json"]
    valid_files = [f for f in all_files if f not in ignore_list]
    return valid_files

def load_data_registry(filenames):
    registry = {}
    for name in filenames:
        try:
            with open(name, 'r') as f:
                registry[name] = json.load(f)
        except Exception as e:
            st.error(f"Error reading {name}: {e}")
    return registry

def get_data_summary(data_registry):
    summary = ""
    for name, data in data_registry.items():
        summary += f"\n--- FILE: {name} ---\n"
        # Peek at structure
        if isinstance(data, list) and len(data) > 0:
            summary += f"Type: List of Objects. Count: {len(data)}\n"
            summary += f"Sample: {json.dumps(data[:1])}\n"
        elif isinstance(data, dict):
            summary += f"Type: Dictionary. Keys: {list(data.keys())}\n"
            sample = {k: v for k, v in list(data.items())[:2]}
            summary += f"Sample: {json.dumps(sample)}\n"
    return summary

def get_local_response(user_query, data_summary):
    system_prompt = f"""
    You are a Python Data Scientist. 
    You have a dictionary named `datasets`. Keys are filenames.
    
    Data Overview:
    {data_summary}

    Task: Write Python code to answer: "{user_query}"
    
    RULES:
    1. Access data via `datasets['filename.json']`.
    2. Join/Merge data using Pandas if needed.
    3. Use `pd.json_normalize()` for nested data.
    4. Save final table to `df_result`.
    5. Save final chart to `fig`.
    6. OUTPUT ONLY PYTHON CODE. NO MARKDOWN.
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

# 1. Auto-detect files
found_files = get_json_files()

if found_files:
    # Load data
    data_registry = load_data_registry(found_files)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"Found {len(found_files)} Files")
        
        # Create tabs for preview
        if data_registry:
            tabs = st.tabs(list(data_registry.keys()))
            for i, (name, data) in enumerate(data_registry.items()):
                with tabs[i]:
                    st.json(data if isinstance(data, dict) else data[:1], expanded=False)

    with col2:
        user_input = st.text_area("Ask a question about these files:", height=100)
        
        if st.button("Generate View"):
            if not user_input:
                st.warning("Please type a request.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        summary = get_data_summary(data_registry)
                        code = get_local_response(user_input, summary)
                        code = code.replace("```python", "").replace("```", "").strip()
                        
                        with st.expander("Show Logic"):
                            st.code(code, language='python')

                        # Execute
                        local_vars = {"datasets": data_registry, "pd": pd, "px": px, "go": go}
                        exec(code, {}, local_vars)

                        if "fig" in local_vars:
                            st.plotly_chart(local_vars["fig"], use_container_width=True)
                        elif "df_result" in local_vars:
                            st.dataframe(local_vars["df_result"], use_container_width=True)
                        else:
                            st.warning("No result generated.")

                    except Exception as e:
                        st.error(f"Execution Error: {e}")
else:
    st.warning("No .json files found in the directory.")
    st.markdown("👉 **Action:** Drag and drop your `.json` files into the file list on the left side of the screen.")
