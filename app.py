import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import os

st.set_page_config(page_title="Notebook Viewer", layout="wide")

st.title("📘 Jupyter Notebook Viewer")
st.write("### File: A3_44_Giriraj_Pande.ipynb")

# Path to your notebook file
notebook_path = "A3_44_Giriraj_Pande.ipynb"

if not os.path.exists(notebook_path):
    st.error(f"❌ File not found: {notebook_path}")
else:
    # Load and convert notebook to HTML
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    html_exporter.exclude_input = False  # set True to hide code cells
    (body, resources) = html_exporter.from_notebook_node(notebook_content)

    # Display notebook in Streamlit
    st.components.v1.html(body, height=1000, scrolling=True)

st.markdown("---")
st.info("💡 Tip: You can upload a different .ipynb file by replacing it in the same folder.")
