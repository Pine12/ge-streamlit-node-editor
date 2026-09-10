<div align="center">

# 🔗 streamlit-node-editor

**A ComfyUI / Unreal Blueprints-style node graph editor for Streamlit — build visual AI pipelines in pure Python**

<a href="https://pypi.org/project/streamlit-node-editor/"><img src="https://img.shields.io/pypi/v/streamlit-node-editor.svg?style=flat-square&color=818cf8" alt="PyPI version" /></a>
<a href="https://pypi.org/project/streamlit-node-editor/"><img src="https://img.shields.io/pypi/pyversions/streamlit-node-editor.svg?style=flat-square" alt="Python versions" /></a>
<a href="https://pypi.org/project/streamlit-node-editor/"><img src="https://img.shields.io/pypi/dm/streamlit-node-editor.svg?style=flat-square&color=34d399" alt="Downloads" /></a>
<img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License" />

</div>

---

`streamlit-node-editor` is a fully interactive node graph editor for Streamlit. Define typed node types in Python, let users build graphs by dragging ports and connecting wires, and get the full graph state back as structured JSON — no React Flow or external graph library needed.

## ✨ Features

- **Typed ports** — color-coded input/output ports; only compatible types connect
- **Drag-to-wire** — click an output port, drag, drop on a compatible input
- **Type validation** — incompatible connections are silently rejected
- **Animated wires** — smooth Bézier curves with flowing-dot animation
- **Inline parameters** — selects, number inputs, text inputs inside each node
- **Drag to move** — reposition nodes freely on the canvas
- **Collapse nodes** — minimize to save canvas space while preserving connections
- **Pan + scroll-to-zoom** — navigate large graphs with proper SVG transforms
- **Full state round-trip** — graph returned to Python as structured JSON

## 🚀 Quick Start

```bash
pip install streamlit-node-editor
```

```python
import streamlit as st
from streamlit_node_editor import node_editor

NODE_TYPES = {
    "Text Input": {"outputs": [{"id": "text", "type": "string", "label": "Text"}], "params": []},
    "GPT-4":      {"inputs":  [{"id": "prompt", "type": "string", "label": "Prompt"}],
                   "outputs": [{"id": "response", "type": "string", "label": "Response"}], "params": []},
}

result = node_editor(node_types=NODE_TYPES)
st.write(result)
```

## 🆕 Recent Enhancements

*Contributed by [Graham E](mailto:grhmelliott@gmail.com)*

- **Multi-select & bulk delete** — shift/ctrl/cmd-click to select multiple nodes, then remove them via a toolbar **DELETE** button or the Delete/Backspace key
- **Auto-connect** — a toolbar **AUTO-CONNECT** button wires up compatible, unfilled ports left-to-right across the graph while detecting and skipping connections that would form a cycle
- **Safer rewiring** — dropping a wire on an input that's already at its connection limit now warns before replacing or refuses with a toast message, instead of silently dropping the old connection
- **Wire tooltip** — hovering a connection shows "Click to remove"
- **Fixed editable node params** — select/text/number param fields previously couldn't be changed; edits now correctly update node state
- **Larger port hit-areas** — ports have an enlarged, invisible buffer zone so wires snap into place without needing pixel-precise drops
- **Custom numeric steppers** — replaced native number-input spinner arrows (which mis-render under SVG pan/zoom) with custom increment/decrement buttons
- **No more accidental dragging** — interacting with dropdowns, text areas, and number inputs no longer drags the node underneath; dragging a node or wire no longer triggers ghost text-selection highlighting
- **Reduced UI dimming** — node/graph updates are now debounced before being sent back to Streamlit, avoiding rerun-triggered dimming while dragging or typing

## 🛠️ Tech Stack

- **React + TypeScript** — frontend component
- **SVG + Canvas** — node graph rendering
- **Python / Streamlit** — backend integration
- **PyPI** — distributed as `streamlit-node-editor`

## 🤝 Contributing

PRs welcome. Open an issue first for major changes.

## 📄 License

MIT

## 💛 Support

If streamlit-node-editor helps you build visual AI pipelines, consider supporting development:

👉 [Donate via PayPal](https://paypal.me/noodlebake) — @noodlebake

---
<div align="center">Made with ❤️ by <a href="https://github.com/RhythrosaLabs">RhythrosaLabs</a></div>
