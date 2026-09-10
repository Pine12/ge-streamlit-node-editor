import streamlit as st
from streamlit_node_editor import st_node_editor
import json

st.set_page_config(page_title="Node Editor Demo", layout="wide")
st.title("🔗 streamlit-node-editor demo")
st.caption("Right-click the canvas to add nodes. Drag from an output port to an input port to connect. Click a wire to delete it. Delete key removes the selected node.")
NODE_DEFS = None
with open("data/etr_config.json", "r") as f:
    NODE_DEFS = json.load(f)["nodes"]

# NODE_DEFS = None
graph = st_node_editor(node_defs=NODE_DEFS, initial_nodes=[], height=650, key="demo_graph")

if graph:
    with st.expander(f"Graph JSON — {len(graph['nodes'])} nodes, {len(graph['connections'])} connections"):
        st.json(graph)
