import { useState, useRef, useCallback, useEffect } from "react";
import { Streamlit } from "streamlit-component-lib";

// ── Port type system (ComfyUI-style color coding) ─────────────────────────────
let PORT_TYPES = {};
// var PORT_TYPES = {
//   IMAGE:   { color: "#4ade80", label: "IMAGE" },
//   LATENT:  { color: "#c084fc", label: "LATENT" },
//   MODEL:   { color: "#fb923c", label: "MODEL" },
//   CLIP:    { color: "#facc15", label: "CLIP" },
//   VAE:     { color: "#f87171", label: "VAE" },
//   INT:     { color: "#38bdf8", label: "INT" },
//   FLOAT:   { color: "#818cf8", label: "FLOAT" },
//   STRING:  { color: "#a8a29e", label: "STRING" },
//   MASK:    { color: "#2dd4bf", label: "MASK" },
//   ANY:     { color: "#71717a", label: "ANY" },
// };

// ── Node definitions (the "palette") ──────────────────────────────────────────
let NODE_DEFS = {
  "Load Checkpoint": {
    category: "Loaders",
    color: "#1e293b",
    headerColor: "#fb923c",
    inputs: [],
    outputs: [
      { name: "MODEL", type: "MODEL" },
      { name: "CLIP",  type: "CLIP" },
      { name: "VAE",   type: "VAE" },
    ],
    params: [
      { key: "ckpt_name", label: "Checkpoint", type: "select", options: ["v1-5-pruned.ckpt", "sd_xl_base.safetensors", "dreamshaper_8.safetensors"] },
    ],
  },
  "CLIP Text Encode": {
    category: "Conditioning",
    color: "#1e293b",
    headerColor: "#facc15",
    inputs:  [{ name: "clip", type: "CLIP" }],
    outputs: [{ name: "CONDITIONING", type: "LATENT" }],
    params: [
      { key: "text", label: "Prompt", type: "textarea" },
    ],
  },
  "KSampler": {
    category: "Sampling",
    color: "#1e293b",
    headerColor: "#818cf8",
    inputs: [
      { name: "model",           type: "MODEL" },
      { name: "positive",        type: "LATENT" },
      { name: "negative",        type: "LATENT" },
      { name: "latent_image",    type: "LATENT" },
    ],
    outputs: [{ name: "LATENT", type: "LATENT" }],
    params: [
      { key: "seed",      label: "Seed",      type: "int",   default: 42 },
      { key: "steps",     label: "Steps",     type: "int",   default: 20 },
      { key: "cfg",       label: "CFG",       type: "float", default: 7.0 },
      { key: "sampler",   label: "Sampler",   type: "select", options: ["euler", "euler_a", "dpm++2m", "ddim"] },
      { key: "scheduler", label: "Scheduler", type: "select", options: ["normal", "karras", "exponential"] },
      { key: "denoise",   label: "Denoise",   type: "float", default: 1.0 },
    ],
  },
  "Empty Latent Image": {
    category: "Latent",
    color: "#1e293b",
    headerColor: "#c084fc",
    inputs:  [],
    outputs: [{ name: "LATENT", type: "LATENT" }],
    params: [
      { key: "width",  label: "Width",  type: "int", default: 512 },
      { key: "height", label: "Height", type: "int", default: 512 },
      { key: "batch",  label: "Batch",  type: "int", default: 1 },
    ],
  },
  "VAE Decode": {
    category: "Latent",
    color: "#1e293b",
    headerColor: "#f87171",
    inputs:  [{ name: "samples", type: "LATENT" }, { name: "vae", type: "VAE" }],
    outputs: [{ name: "IMAGE",   type: "IMAGE" }],
    params: [],
  },
  "Save Image": {
    category: "Output",
    color: "#1e293b",
    headerColor: "#4ade80",
    inputs:  [{ name: "images", type: "IMAGE" }],
    outputs: [],
    params: [
      { key: "filename_prefix", label: "Filename", type: "string", default: "output" },
    ],
  },
  "Image Scale": {
    category: "Image",
    color: "#1e293b",
    headerColor: "#2dd4bf",
    inputs:  [{ name: "image", type: "IMAGE" }],
    outputs: [{ name: "IMAGE", type: "IMAGE" }],
    params: [
      { key: "upscale_method", label: "Method", type: "select", options: ["nearest", "bilinear", "bicubic", "lanczos"] },
      { key: "width",  label: "Width",  type: "int", default: 1024 },
      { key: "height", label: "Height", type: "int", default: 1024 },
    ],
  },
  "Integer": {
    category: "Primitives",
    color: "#1e293b",
    headerColor: "#38bdf8",
    inputs:  [],
    outputs: [{ name: "INT", type: "INT" }],
    params: [{ key: "value", label: "Value", type: "int", default: 0 }],
  },
};

// ── Utilities ─────────────────────────────────────────────────────────────────
let _id = 1;
const uid = () => `node_${_id++}`;

function makeNode(type, x, y, paramValues = {}) {
  const def = NODE_DEFS[type] || { params: [] };
  const defaultParams = Object.fromEntries(
    (def.params || []).map(p => [p.key, p.default ?? ""])
  );
  return {
    id: uid(),
    type,
    x, y,
    width: 240,
    // merge defaults with any provided values
    params: { ...defaultParams, ...(paramValues || {}) },
    collapsed: false,
  };
}

function portHeight(index) { return 44 + index * 28; }
function nodeHeight(node) {
  const def = NODE_DEFS[node.type];
  if (node.collapsed) return 36;
  const portRows = Math.max(def.inputs.length, def.outputs.length);
  const paramH = def.params.reduce((sum, p) => sum + (p.type === "textarea" ? 60 : 34), 0);
  return 44 + portRows * 28 + paramH + 16;
}

function getPortPos(node, side, index) {
  const y = node.y + portHeight(index);
  const x = side === "output" ? node.x + node.width + 1 : node.x - 1;
  return { x, y };
}

function typesCompatible(a, b) {
  if (a === "ANY" || b === "ANY") return true;
  return a === b;
}

function inputMaxConnections(nodeType, portIndex) {
  const port = (NODE_DEFS[nodeType]?.inputs || [])[portIndex];
  return port?.maxConnections ?? 1;
}

// ── Wire SVG path (cubic bezier) ─────────────────────────────────────────────
function wirePath(x1, y1, x2, y2) {
  const dx = Math.abs(x2 - x1) * 0.6 + 60;
  return `M ${x1} ${y1} C ${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}`;
}

// ── Node component ─────────────────────────────────────────────────────────────
function GraphNode({ node, selected, wiring, onSelect, onDragStart,
  onPortMouseDown, onPortMouseUp, onParamChange, connections }) {
  const def = NODE_DEFS[node.type];
  const h = nodeHeight(node);

  const connectedInputs  = new Set(connections.filter(c => c.toNode === node.id).map(c => c.toPort));
  const connectedOutputs = new Set(connections.filter(c => c.fromNode === node.id).map(c => c.fromPort));

  return (
    <g transform={`translate(${node.x},${node.y})`}
      onMouseDown={e => { e.preventDefault(); e.stopPropagation(); onDragStart(e, node.id); onSelect(node.id); }}>

      {/* Shadow */}
      <rect x={3} y={3} width={node.width} height={h} rx={8}
        fill="rgba(0,0,0,0.45)" />

      {/* Body */}
      <rect width={node.width} height={h} rx={8}
        fill={node.collapsed ? def.headerColor + "22" : "#13131f"}
        stroke={selected ? "white" : "rgba(255,255,255,0.12)"}
        strokeWidth={selected ? 1.5 : 1} />

      {/* Header */}
      <rect width={node.width} height={32} rx={8} ry={8} fill={def.headerColor} opacity={0.9} />
      <rect y={24} width={node.width} height={12} fill={def.headerColor} opacity={0.9} />

      {/* Title */}
      <text x={10} y={21} fontFamily="'Fira Code', monospace" fontSize={12}
        fontWeight={600} fill="rgba(0,0,0,0.85)" style={{ userSelect: "none", pointerEvents: "none" }}>
        {node.type}
      </text>

      {/* Collapse toggle */}
      <text x={node.width - 16} y={21} fontFamily="monospace" fontSize={12}
        fill="rgba(0,0,0,0.6)" style={{ cursor: "pointer", userSelect: "none" }}
        onMouseDown={e => { e.stopPropagation(); }}>
        {node.collapsed ? "+" : "−"}
      </text>

      {!node.collapsed && (
        <>
          {/* Input ports */}
          {def.inputs.map((port, i) => {
            const relY = 44 + i * 28;
            const c = PORT_TYPES[port.type]?.color ?? "#71717a";
            const isConn = connectedInputs.has(i);
            return (
              <g key={`in-${i}`}>
                <circle cx={-1} cy={relY} r={14}
                  fill="transparent"
                  style={{ cursor: "crosshair" }}
                  onMouseDown={e => { e.stopPropagation(); onPortMouseDown(e, node.id, "input", i, port.type); }}
                  onMouseUp={e => { e.stopPropagation(); onPortMouseUp(e, node.id, "input", i, port.type); }}
                />
                <circle cx={-1} cy={relY} r={5.5}
                  fill={isConn ? c : "#13131f"} stroke={c} strokeWidth={2}
                  style={{ pointerEvents: "none" }}
                />
                <text x={14} y={relY + 4} fontFamily="'Fira Code', monospace"
                  fontSize={10} fill={c} style={{ userSelect: "none", pointerEvents: "none" }}>
                  {port.name}
                </text>
              </g>
            );
          })}

          {/* Output ports */}
          {def.outputs.map((port, i) => {
            const relY = 44 + i * 28;
            const c = PORT_TYPES[port.type]?.color ?? "#71717a";
            const isConn = connectedOutputs.has(i);
            return (
              <g key={`out-${i}`}>
                <circle cx={node.width + 1} cy={relY} r={14}
                  fill="transparent"
                  style={{ cursor: "crosshair" }}
                  onMouseDown={e => { e.stopPropagation(); onPortMouseDown(e, node.id, "output", i, port.type); }}
                  onMouseUp={e => { e.stopPropagation(); onPortMouseUp(e, node.id, "output", i, port.type); }}
                />
                <circle cx={node.width + 1} cy={relY} r={5.5}
                  fill={isConn ? c : "#13131f"} stroke={c} strokeWidth={2}
                  style={{ pointerEvents: "none" }}
                />
                <text x={node.width - 14} y={relY + 4} fontFamily="'Fira Code', monospace"
                  fontSize={10} fill={c} textAnchor="end"
                  style={{ userSelect: "none", pointerEvents: "none" }}>
                  {port.name}
                </text>
              </g>
            );
          })}

          {/* Params */}
          {def.params.map((param, i, arr) => {
            const baseY = 44 + Math.max(def.inputs.length, def.outputs.length) * 28
              + arr.slice(0, i).reduce((s, p) => s + (p.type === "textarea" ? 60 : 34), 0) + 8;
            const fHeight = param.type === "textarea" ? 56 : 30;
            return (
              <foreignObject key={param.key} x={8} y={baseY} width={node.width - 16} height={fHeight}
                style={{ pointerEvents: "all" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <label style={{ fontFamily: "'Fira Code', monospace", fontSize: 9, color: "#64748b",
                    minWidth: 52, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    {param.label}
                  </label>
                  {param.type === "select" ? (
                    <select
                      value={node.params[param.key] ?? ""}
                      onChange={e => onParamChange(node.id, param.key, e.target.value)}
                      onClick={e => e.stopPropagation()}
                      onMouseDown={e => e.stopPropagation()}
                      style={{ flex: 1, background: "#0f0f1a", border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: 4, color: "#e2e8f0", fontFamily: "'Fira Code', monospace",
                        fontSize: 10, padding: "2px 4px", outline: "none" }}>
                      {param.options.map(o => <option key={o}>{o}</option>)}
                    </select>
                  ) : param.type === "textarea" ? (
                    <textarea rows={2}
                      value={node.params[param.key] ?? ""}
                      onChange={e => onParamChange(node.id, param.key, e.target.value)}
                      onClick={e => e.stopPropagation()}
                      onMouseDown={e => e.stopPropagation()}
                      style={{ flex: 1, background: "#0f0f1a", border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: 4, color: "#e2e8f0", fontFamily: "'Fira Code', monospace",
                        fontSize: 10, padding: "2px 4px", outline: "none", resize: "none" }} />
                  ) : (
                    <div style={{ flex: 1, display: "flex", alignItems: "stretch" }}>
                      <input type={param.type === "float" || param.type === "int" ? "number" : "text"}
                        value={node.params[param.key] ?? ""}
                        step={param.type === "float" ? 0.1 : 1}
                        onChange={e => onParamChange(node.id, param.key, e.target.value)}
                        onClick={e => e.stopPropagation()}
                        onMouseDown={e => e.stopPropagation()}
                        style={{ flex: 1, minWidth: 0, background: "#0f0f1a", border: "1px solid rgba(255,255,255,0.1)",
                          borderRadius: param.type === "float" || param.type === "int" ? "4px 0 0 4px" : 4,
                          color: "#e2e8f0", fontFamily: "'Fira Code', monospace",
                          fontSize: 10, padding: "2px 4px", outline: "none", MozAppearance: "textfield" }} />
                      {(param.type === "float" || param.type === "int") && (
                        <div style={{ display: "flex", flexDirection: "column", width: 14 }}>
                          {[1, -1].map(dir => (
                            <button key={dir} type="button"
                              onClick={e => {
                                e.stopPropagation();
                                const cur = parseFloat(node.params[param.key]) || 0;
                                const step = param.type === "float" ? 0.1 : 1;
                                const next = param.type === "float"
                                  ? Math.round((cur + dir * step) * 10) / 10
                                  : cur + dir * step;
                                onParamChange(node.id, param.key, next);
                              }}
                              onMouseDown={e => e.stopPropagation()}
                              style={{ flex: 1, background: "#1a1a2e", border: "1px solid rgba(255,255,255,0.1)",
                                borderLeft: "none", borderBottom: dir === 1 ? "none" : undefined,
                                color: "#64748b", cursor: "pointer", fontSize: 7, lineHeight: 1, padding: 0 }}>
                              {dir === 1 ? "▲" : "▼"}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </foreignObject>
            );
          })}
        </>
      )}
    </g>
  );
}

function generateColor(index, total) {
  const hue = Math.round((index / Math.max(total, 1)) * 360);
  return `hsl(${hue}, 70%, 60%)`;
}


// ── Main Editor ───────────────────────────────────────────────────────────────
export default function NodeEditor(props) {
  // const [nodes, setNodes] = useState(() => {
  //   const n1 = makeNode("Load Checkpoint", 60, 80);
  //   const n2 = makeNode("CLIP Text Encode", 360, 60);
  //   const n3 = makeNode("CLIP Text Encode", 360, 240);
  //   const n4 = makeNode("Empty Latent Image", 360, 440);
  //   const n5 = makeNode("KSampler", 680, 200);
  //   const n6 = makeNode("VAE Decode", 1000, 300);
  //   const n7 = makeNode("Save Image", 1260, 320);
  //   n3.params.text = "blurry, bad quality, ugly";
  //   return [n1, n2, n3, n4, n5, n6, n7];
  // });

  const [connections, setConnections] = useState([]);
  const [selected, setSelected] = useState([]); // array of selected node ids
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const [wiring, setWiring] = useState(null); // { fromNode, fromPort, fromSide, fromType, x, y }
  const [wirePos, setWirePos] = useState({ x: 0, y: 0 });
  const [contextMenu, setContextMenu] = useState(null); // { x, y }
  const [palette, setPalette] = useState(false);
  const [paletteSearch, setPaletteSearch] = useState("");
  const [toast, setToast] = useState(null);
  const svgRef = useRef(null);

  // Interaction refs (avoid re-renders during drag)
  const dragNodeRef = useRef(null);   // { id, offsetX, offsetY }
  const panDragRef = useRef(null);    // { startX, startY }
  const panRef = useRef({ x: 0, y: 0 });
  const zoomRef = useRef(1);
  const rafRef = useRef(null);
  useEffect(() => { panRef.current = pan; }, [pan]);
  useEffect(() => { zoomRef.current = zoom; }, [zoom]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3000);
    return () => clearTimeout(t);
  }, [toast]);

  // ── Streamlit lifecycle ──────────────────────────────────────────────────
  const readyRef = useRef(false);
  const [streamlitArgs, setStreamlitArgs] = useState({
    node_defs: null,
    height: 700,
    initial_nodes: [],
    initial_connections: [],
    key: null
  });

  useEffect(() => {
    const onRender = (event) => {
      const args = event.detail.args || {};
      const newArgs = {
        node_defs: args.node_defs || {},
        height: args.height,
        initial_nodes: args.initial_nodes || [],
        initial_connections: args.initial_connections || [],
        key: args.key
      };
      // console.log(JSON.stringify(newArgs.node_defs));
      // console.log(JSON.stringify(NODE_DEFS));
      setStreamlitArgs(newArgs);

      // ── Resolve which NODE_DEFS to use ──────────────────────────────
      const incomingDefs = (newArgs.node_defs && Object.keys(newArgs.node_defs).length > 0)
        ? newArgs.node_defs
        : NODE_DEFS;

      // Update the global synchronously so makeNode() calls below see it
      NODE_DEFS = incomingDefs;

      // Update React state so components re-render with new defs
      setNodeDefs(incomingDefs);

      // ── Build port type color map ────────────────────────────────────
      const allPortTypes = new Set();
      Object.values(incomingDefs).forEach(def => {
        (def.inputs  || []).forEach(i => allPortTypes.add(i.type));
        (def.outputs || []).forEach(o => allPortTypes.add(o.type));
      });
      const typeArray = [...allPortTypes];
      const portTypeMap = Object.fromEntries(
        typeArray.map((type, index) => [
          type,
          { color: generateColor(index, typeArray.length), label: type }
        ])
      );
      PORT_TYPES = portTypeMap;
      setPortTypes(portTypeMap);

      // ── Initialize nodes/connections only once ───────────────────────
      if (!readyRef.current) {
        if (newArgs.initial_nodes?.length) {
          setNodes(newArgs.initial_nodes.map(n => ({
            ...makeNode(n.type, n.x ?? 0, n.y ?? 0, n.params),
            id: n.id,
          })));
        } else {
          // Auto-layout nodes from defs if no initial_nodes provided
          let x_pos = -140;
          let y_pos = -40;
          setNodes(Object.entries(incomingDefs).map(([name, def]) => {
            x_pos += 150;
            y_pos += 50;
            return makeNode(name, def.x || x_pos, def.y || y_pos, {});
          }));
        }

        if (newArgs.initial_connections?.length) {
          setConnections(newArgs.initial_connections);
        }

        readyRef.current = true;
      }

      Streamlit.setFrameHeight(newArgs.height || 700);
    };

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
    return () => Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, onRender);
  }, []); // empty deps — register once only

  // console.log("Global Node Defs: " + JSON.stringify(NODE_DEFS))

  const [nodeDefs, setNodeDefs] = useState(NODE_DEFS); // tracks current definitions
  const [nodes, setNodes] = useState([]); // setNodes is defined here
  const [portTypes, setPortTypes] = useState([]);

  // debounce so rapid drag/typing updates don't trigger a Streamlit rerun (and dimming) on every frame
  const commitTimeoutRef = useRef(null);
  useEffect(() => {
    if (!readyRef.current) return;
    if (commitTimeoutRef.current) clearTimeout(commitTimeoutRef.current);
    commitTimeoutRef.current = setTimeout(() => {
      Streamlit.setComponentValue({
        nodes: nodes.map(n => ({ id: n.id, type: n.type, x: n.x, y: n.y, params: n.params })),
        connections,
      });
    }, 150);
    return () => clearTimeout(commitTimeoutRef.current);
  }, [nodes, connections]);

  useEffect(() => { Streamlit.setFrameHeight(); });

  // ── SVG coordinate helpers ──────────────────────────────────────────────────
  const svgPoint = useCallback((e) => {
    const rect = svgRef.current.getBoundingClientRect();
    const p = panRef.current, z = zoomRef.current;
    return {
      x: (e.clientX - rect.left - p.x) / z,
      y: (e.clientY - rect.top  - p.y) / z,
    };
  }, []);

  const svgRaw = useCallback((e) => {
    const rect = svgRef.current.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  // ── Drag node ──────────────────────────────────────────────────────────────
  const handleNodeDragStart = useCallback((e, id) => {
    if (wiring) return;
    const pt = svgPoint(e);
    const node = nodes.find(n => n.id === id);
    dragNodeRef.current = { id, offsetX: pt.x - node.x, offsetY: pt.y - node.y };
  }, [nodes, svgPoint, wiring]);

  // ── Pan canvas ────────────────────────────────────────────────────────────
  const handleCanvasMouseDown = useCallback((e) => {
    if (e.button === 1 || e.button === 2 || (e.button === 0 && !wiring)) {
      if (e.button !== 2) {
        setIsPanning(true);
        panDragRef.current = { startX: e.clientX - panRef.current.x, startY: e.clientY - panRef.current.y };
      }
      setSelected([]);
    }
  }, [wiring]);

  // ── Mouse move (rAF-throttled) ─────────────────────────────────────────────
  const handleMouseMove = useCallback((e) => {
    if (dragNodeRef.current || panDragRef.current || wiring) e.preventDefault();
    const cx = e.clientX, cy = e.clientY;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      if (dragNodeRef.current) {
        const rect = svgRef.current.getBoundingClientRect();
        const p = panRef.current, z = zoomRef.current;
        const ptx = (cx - rect.left - p.x) / z;
        const pty = (cy - rect.top  - p.y) / z;
        const d = dragNodeRef.current;
        setNodes(ns => ns.map(n => n.id === d.id
          ? { ...n, x: ptx - d.offsetX, y: pty - d.offsetY }
          : n
        ));
      }
      if (panDragRef.current) {
        const newPan = { x: cx - panDragRef.current.startX, y: cy - panDragRef.current.startY };
        panRef.current = newPan;
        setPan(newPan);
      }
      if (wiring) {
        const rect = svgRef.current.getBoundingClientRect();
        setWirePos({ x: cx - rect.left, y: cy - rect.top });
      }
    });
  }, [wiring]);

  const handleMouseUp = useCallback(() => {
    dragNodeRef.current = null;
    panDragRef.current = null;
    setIsPanning(false);
    if (wiring) setWiring(null);
  }, [wiring]);

  // ── Zoom ──────────────────────────────────────────────────────────────────
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(0.2, Math.min(3, z * delta)));
  }, []);

  // ── Port interactions ──────────────────────────────────────────────────────
  const handlePortMouseDown = useCallback((e, nodeId, side, portIndex, portType) => {
    e.preventDefault();
    const raw = svgRaw(e);
    setWiring({ fromNode: nodeId, fromPort: portIndex, fromSide: side, fromType: portType });
    setWirePos(raw);
  }, [svgRaw]);

  const handlePortMouseUp = useCallback((e, nodeId, side, portIndex, portType) => {
    if (!wiring) return;
    // Must connect output → input
    const isOutputToInput = wiring.fromSide === "output" && side === "input";
    const isInputToOutput = wiring.fromSide === "input"  && side === "output";
    if (!(isOutputToInput || isInputToOutput)) { setWiring(null); return; }
    if (wiring.fromNode === nodeId) { setWiring(null); return; }
    if (!typesCompatible(wiring.fromType, portType)) { setWiring(null); return; }

    const fromNode = isOutputToInput ? wiring.fromNode : nodeId;
    const fromPort = isOutputToInput ? wiring.fromPort : portIndex;
    const toNode   = isOutputToInput ? nodeId : wiring.fromNode;
    const toPort   = isOutputToInput ? portIndex : wiring.fromPort;
    const maxConnections = inputMaxConnections(
      nodes.find(node => node.id === toNode)?.type,
      toPort,
    );
    setWiring(null);

    if (maxConnections <= 0) {
      setToast("This input does not accept connections.");
      return;
    }

    const inputConnections = connections.filter(c => c.toNode === toNode && c.toPort === toPort);
    if (inputConnections.length >= maxConnections) {
      if (maxConnections > 1) {
        setToast(`This input already has the maximum of ${maxConnections} connections.`);
        return;
      }
      // single-input port: warn before replacing the existing wire
      const replace = window.confirm("This input already has a connection. Replace it?");
      if (!replace) return;
    }

    setConnections(cs => [
      ...cs.filter(c => maxConnections > 1 || !(c.toNode === toNode && c.toPort === toPort)),
      { id: `w${Date.now()}`, fromNode, fromPort, toNode, toPort },
    ]);
  }, [nodes, wiring, connections]);

  // ── Context menu (right-click canvas) ────────────────────────────────────
  const handleContextMenu = useCallback((e) => {
    e.preventDefault();
    const raw = svgRaw(e);
    setContextMenu({ x: e.clientX, y: e.clientY, svgX: raw.x, svgY: raw.y });
  }, [svgRaw]);

  // ── Add node ──────────────────────────────────────────────────────────────
  const addNode = useCallback((type, rawX, rawY) => {
    const pt = {
      x: (rawX - pan.x) / zoom,
      y: (rawY - pan.y) / zoom,
    };
    setNodes(ns => [...ns, makeNode(type, pt.x - 120, pt.y - 18)]);
    setContextMenu(null);
    setPalette(false);
  }, [pan, zoom]);

  // ── Update a node's param value ──────────────────────────────────────────
  const handleParamChange = useCallback((nodeId, key, value) => {
    setNodes(ns => ns.map(n => n.id === nodeId
      ? { ...n, params: { ...n.params, [key]: value } }
      : n
    ));
  }, []);

  // ── Delete selected node(s) ───────────────────────────────────────────────
  const handleDeleteSelected = useCallback(() => {
    setNodes(ns => ns.filter(n => !selected.includes(n.id)));
    setConnections(cs => cs.filter(c => !selected.includes(c.fromNode) && !selected.includes(c.toNode)));
    setSelected([]);
  }, [selected]);

  useEffect(() => {
    const handler = (e) => {
      if ((e.key === "Delete" || e.key === "Backspace") && selected.length) {
        handleDeleteSelected();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selected, handleDeleteSelected]);

  // ── Auto-connect matching ports, left-to-right, skipping cycles ──────────
  const handleAutoConnect = useCallback(() => {
    // true if `fromId` can already reach `toId` via existing connections
    const canReach = (fromId, toId, conns) => {
      const visited = new Set();
      const stack = [fromId];
      while (stack.length) {
        const cur = stack.pop();
        if (cur === toId) return true;
        if (visited.has(cur)) continue;
        visited.add(cur);
        conns.filter(c => c.fromNode === cur).forEach(c => stack.push(c.toNode));
      }
      return false;
    };

    setConnections(cs => {
      const result = [...cs];
      const orderedNodes = [...nodes].sort((a, b) => a.x - b.x);

      orderedNodes.forEach(toNodeObj => {
        const toDef = NODE_DEFS[toNodeObj.type];
        if (!toDef) return;

        toDef.inputs.forEach((inputPort, toPort) => {
          const maxConn = inputMaxConnections(toNodeObj.type, toPort);

          // keep adding matching outputs until this input's connection limit is reached
          for (;;) {
            const existing = result.filter(c => c.toNode === toNodeObj.id && c.toPort === toPort);
            if (existing.length >= maxConn) break;

            const fromNodeObj = orderedNodes.find(candidate => {
              if (candidate.id === toNodeObj.id) return false;
              const fromDef = NODE_DEFS[candidate.type];
              if (!fromDef) return false;
              return fromDef.outputs.some((outputPort, fromPort) =>
                typesCompatible(outputPort.type, inputPort.type) &&
                !canReach(toNodeObj.id, candidate.id, result) &&
                !existing.some(c => c.fromNode === candidate.id && c.fromPort === fromPort)
              );
            });
            if (!fromNodeObj) break;

            const fromDef = NODE_DEFS[fromNodeObj.type];
            const fromPort = fromDef.outputs.findIndex((outputPort, idx) =>
              typesCompatible(outputPort.type, inputPort.type) &&
              !canReach(toNodeObj.id, fromNodeObj.id, result) &&
              !existing.some(c => c.fromNode === fromNodeObj.id && c.fromPort === idx)
            );
            if (fromPort === -1) break;

            result.push({
              id: `w${Date.now()}_${toNodeObj.id}_${toPort}_${fromNodeObj.id}_${fromPort}`,
              fromNode: fromNodeObj.id,
              fromPort,
              toNode: toNodeObj.id,
              toPort,
            });
          }
        });
      });

      return result;
    });
  }, [nodes]);

  // ── Wire geometry ─────────────────────────────────────────────────────────
  const getWireEndpoints = (conn) => {
    const fromNode = nodes.find(n => n.id === conn.fromNode);
    const toNode   = nodes.find(n => n.id === conn.toNode);
    if (!fromNode || !toNode) return null;
    const from = getPortPos(fromNode, "output", conn.fromPort);
    const to   = getPortPos(toNode,   "input",  conn.toPort);
    return { from, to };
  };

  const getWireColor = (conn) => {
    const fromNode = nodes.find(n => n.id === conn.fromNode);
    if (!fromNode) return "#71717a";
    const def = NODE_DEFS[fromNode.type];
    const port = def.outputs[conn.fromPort];
    return PORT_TYPES[port?.type]?.color ?? "#71717a";
  };

  // ── Filtered palette ──────────────────────────────────────────────────────
  const filteredDefs = Object.entries(nodeDefs).filter(([name]) =>
    name.toLowerCase().includes(paletteSearch.toLowerCase())
  );
  const categories = [...new Set(filteredDefs.map(([, d]) => d.category))];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
        select option { background: #0f0f1a; }
        /* native spinner chrome mis-positions under SVG pan/zoom transforms */
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
      `}</style>

      <div style={{ width: "100vw", height: "100vh", background: "#080810", overflow: "hidden",
        fontFamily: "'Fira Code', monospace", position: "relative", userSelect: "none" }}>

        {/* ── Toolbar ───────────────────────────────────────────────────── */}
        <div style={{ position: "absolute", top: 10, right: 12,
          display: "flex", gap: 6, zIndex: 100, alignItems: "center" }}>
          <div style={{ background: "rgba(15,15,26,0.85)", border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 8, padding: "5px 10px", display: "flex", gap: 8, alignItems: "center",
            backdropFilter: "blur(8px)" }}>
            <button onClick={() => { setPalette(p => !p); setContextMenu(null); }} style={{
              background: palette ? "rgba(99,102,241,0.3)" : "rgba(255,255,255,0.05)",
              border: `1px solid ${palette ? "rgba(99,102,241,0.5)" : "rgba(255,255,255,0.1)"}`,
              borderRadius: 5, color: palette ? "#818cf8" : "#94a3b8",
              padding: "3px 10px", cursor: "pointer", fontSize: 10, letterSpacing: "0.05em"
            }}>+ ADD</button>
            <div style={{ width: 1, height: 14, background: "rgba(255,255,255,0.08)" }} />
            <button onClick={handleAutoConnect} style={{
              background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 5, color: "#94a3b8",
              padding: "3px 10px", cursor: "pointer", fontSize: 10, letterSpacing: "0.05em"
            }}>AUTO-CONNECT</button>
            <div style={{ width: 1, height: 14, background: "rgba(255,255,255,0.08)" }} />
            <span style={{ fontSize: 10, color: "#475569" }}>{Math.round(zoom * 100)}%</span>
            <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); panRef.current = { x: 0, y: 0 }; zoomRef.current = 1; }}
              style={{ background: "none", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 4,
                color: "#475569", padding: "2px 6px", cursor: "pointer", fontSize: 9 }}>RESET</button>
            <div style={{ width: 1, height: 14, background: "rgba(255,255,255,0.08)" }} />
            <button onClick={handleDeleteSelected} disabled={!selected.length} style={{
              background: selected.length ? "rgba(248,113,113,0.15)" : "rgba(255,255,255,0.05)",
              border: `1px solid ${selected.length ? "rgba(248,113,113,0.4)" : "rgba(255,255,255,0.1)"}`,
              borderRadius: 5, color: selected.length ? "#f87171" : "#475569",
              padding: "3px 10px", cursor: selected.length ? "pointer" : "not-allowed",
              fontSize: 10, letterSpacing: "0.05em"
            }}>DELETE{selected.length > 1 ? ` (${selected.length})` : ""}</button>
          </div>
        </div>
        {/* ── Node palette ──────────────────────────────────────────────── */}
        {palette && (
          <div style={{ position: "absolute", top: 60, left: 20, width: 220,
            background: "rgba(13,13,25,0.98)", border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 12, zIndex: 200, overflow: "hidden", boxShadow: "0 16px 48px rgba(0,0,0,0.6)" }}>
            <div style={{ padding: "10px 12px 8px" }}>
              <input autoFocus value={paletteSearch} onChange={e => setPaletteSearch(e.target.value)}
                placeholder="Search nodes…"
                style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 6, color: "#e2e8f0", fontFamily: "'Fira Code', monospace", fontSize: 11,
                  padding: "6px 10px", outline: "none" }} />
            </div>
            <div style={{ maxHeight: 400, overflowY: "auto", paddingBottom: 8 }}>
              {categories.map(cat => (
                <div key={cat}>
                  <div style={{ padding: "4px 12px 2px", fontSize: 9, color: "#475569",
                    letterSpacing: "0.15em", textTransform: "uppercase" }}>{cat}</div>
                  {filteredDefs.filter(([, d]) => d.category === cat).map(([name, def]) => (
                    <div key={name} onClick={() => addNode(name, window.innerWidth / 2, window.innerHeight / 2)}
                      style={{ padding: "6px 12px", cursor: "pointer", display: "flex",
                        alignItems: "center", gap: 8, transition: "background 0.1s" }}
                      onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.05)"}
                      onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: def.headerColor, flexShrink: 0 }} />
                      <span style={{ fontSize: 11, color: "#cbd5e1" }}>{name}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Toast notification ───────────────────────────────────────── */}
        {toast && (
          <div style={{ position: "absolute", top: 60, left: "50%", transform: "translateX(-50%)",
            background: "rgba(127,29,29,0.95)", border: "1px solid rgba(248,113,113,0.4)",
            borderRadius: 8, padding: "8px 16px", zIndex: 400, color: "#fecaca", fontSize: 11,
            boxShadow: "0 8px 24px rgba(0,0,0,0.5)" }}>
            {toast}
          </div>
        )}

        {/* ── Port type legend ──────────────────────────────────────────── */}
        <div style={{ position: "absolute", bottom: 16, left: 16, zIndex: 100,
          background: "rgba(13,13,25,0.9)", border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 8, padding: "8px 12px", display: "flex", gap: 12, flexWrap: "wrap", maxWidth: 500 }}>
          {Object.entries(PORT_TYPES).filter(([k]) => k !== "ANY").map(([k, v]) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: v.color }} />
              <span style={{ fontSize: 9, color: "#475569", letterSpacing: "0.05em" }}>{k}</span>
            </div>
          ))}
        </div>

        {/* ── Context menu ─────────────────────────────────────────────── */}
        {contextMenu && (
          <div style={{ position: "fixed", left: contextMenu.x, top: contextMenu.y,
            background: "rgba(13,13,25,0.98)", border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 10, zIndex: 300, overflow: "hidden", minWidth: 200,
            boxShadow: "0 16px 48px rgba(0,0,0,0.6)" }}
            onMouseLeave={() => setContextMenu(null)}>
            <div style={{ padding: "6px 12px 4px", fontSize: 9, color: "#475569", letterSpacing: "0.15em" }}>
              ADD NODE
            </div>
            {Object.entries(nodeDefs).map(([name, def]) => (
              <div key={name}
                onClick={() => addNode(name, contextMenu.svgX, contextMenu.svgY)}
                style={{ padding: "6px 14px", cursor: "pointer", fontSize: 11, color: "#cbd5e1",
                  display: "flex", alignItems: "center", gap: 8 }}
                onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"}
                onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: def.headerColor, flexShrink: 0 }} />
                {name}
              </div>
            ))}
          </div>
        )}

        {/* ── Canvas SVG ───────────────────────────────────────────────── */}
        <svg ref={svgRef} width="100%" height="100%"
          style={{ cursor: isPanning ? "grabbing" : wiring ? "crosshair" : "default" }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
          onContextMenu={handleContextMenu}
          onClick={() => setContextMenu(null)}>

          <defs>
            <pattern id="grid" width={40 * zoom} height={40 * zoom} patternUnits="userSpaceOnUse"
              x={pan.x % (40 * zoom)} y={pan.y % (40 * zoom)}>
              <path d={`M ${40 * zoom} 0 L 0 0 0 ${40 * zoom}`}
                fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
            </pattern>
          </defs>

          {/* Grid background */}
          <rect width="100%" height="100%" fill="url(#grid)" />

          <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>

            {/* Connections */}
            {connections.map(conn => {
              const pts = getWireEndpoints(conn);
              if (!pts) return null;
              const color = getWireColor(conn);
              return (
                <path key={conn.id}
                  d={wirePath(pts.from.x, pts.from.y, pts.to.x, pts.to.y)}
                  fill="none" stroke={color} strokeWidth={2.5} strokeOpacity={0.85}
                  style={{ cursor: "pointer" }}
                  onClick={e => { e.stopPropagation(); setConnections(cs => cs.filter(c => c.id !== conn.id)); }}>
                  <title>Click to remove</title>
                </path>
              );
            })}

            {/* Nodes */}
            {nodes.map(node => (
              <GraphNode key={node.id} node={node}
                selected={selected.includes(node.id)}
                wiring={wiring}
                connections={connections}
                onSelect={(id, additive) => setSelected(sel => additive
                  ? (sel.includes(id) ? sel.filter(s => s !== id) : [...sel, id])
                  : [id]
                )}
                onDragStart={handleNodeDragStart}
                onPortMouseDown={handlePortMouseDown}
                onPortMouseUp={handlePortMouseUp}
                onParamChange={handleParamChange}
              />
            ))}
          </g>

          {/* Live wire being dragged */}
          {wiring && (() => {
            const fromNode = nodes.find(n => n.id === wiring.fromNode);
            if (!fromNode) return null;
            const from = getPortPos(fromNode, wiring.fromSide, wiring.fromPort);
            const fx = from.x * zoom + pan.x;
            const fy = from.y * zoom + pan.y;
            const color = PORT_TYPES[wiring.fromType]?.color ?? "#71717a";
            return (
              <path d={wirePath(fx, fy, wirePos.x, wirePos.y)}
                fill="none" stroke={color} strokeWidth={2.5} strokeOpacity={0.7}
                strokeDasharray="6 3" />
            );
          })()}
        </svg>
      </div>
    </>
  );
}
