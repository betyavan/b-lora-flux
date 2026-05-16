"""Generate FLUX B-LoRA architecture diagram using graphviz."""
import graphviz
from pathlib import Path

OUT = Path("/Users/i.betev/Desktop/DIPLOMA/output/presentation_figures")
OUT.mkdir(parents=True, exist_ok=True)

dot = graphviz.Digraph(
    name="flux_blora",
    format="png",
    graph_attr={
        "rankdir": "LR",
        "bgcolor": "#F8F8F8",
        "fontname": "DejaVu Sans",
        "splines": "ortho",
        "nodesep": "0.5",
        "ranksep": "1.0",
        "pad": "0.4",
        "dpi": "150",
        "label": "Архитектура FLUX.1-dev и расположение адаптеров B-LoRA\n(MMDiT: 19 Double-Stream + 38 Single-Stream блоков)",
        "labelloc": "t",
        "fontsize": "16",
        "fontcolor": "#222222",
        "newrank": "true",
    },
    node_attr={
        "fontname": "DejaVu Sans",
        "fontsize": "11",
        "style": "filled,rounded",
        "shape": "box",
        "margin": "0.2,0.12",
        "penwidth": "1.5",
        "width": "1.8",
    },
    edge_attr={
        "color": "#555555",
        "penwidth": "1.8",
        "arrowsize": "0.9",
    },
)

# ── Nodes ─────────────────────────────────────────────────────────────────────
dot.node("input", "Промпт\n+ шум",
         fillcolor="#4A4A5A", fontcolor="white")

with dot.subgraph(name="cluster_style") as c:
    c.attr(label="LoRA-Style", style="dashed", color="#C0392B",
           penwidth="2.5", fontcolor="#C0392B", fontsize="11", bgcolor="#FFF0EE")
    c.node("ds_01", "DS 00–01\n★ СТИЛЬ",
           fillcolor="#C0392B", fontcolor="white")

dot.node("ds_02", "DS 02–04\n(нейтральные)",
         fillcolor="#7F8C8D", fontcolor="white")

with dot.subgraph(name="cluster_content") as c:
    c.attr(label="LoRA-Content", style="dashed", color="#2980B9",
           penwidth="2.5", fontcolor="#2980B9", fontsize="11", bgcolor="#EEF4FF")
    c.node("ds_05", "DS 05–12\n★ КОНТЕНТ",
           fillcolor="#2980B9", fontcolor="white")

dot.node("ds_13", "DS 13–18",
         fillcolor="#7F8C8D", fontcolor="white")

dot.node("ss", "SS 00–37\n(38 блоков)",
         fillcolor="#546E7A", fontcolor="white")

dot.node("vae", "Декодер VAE\n→ изображение",
         fillcolor="#4A4A5A", fontcolor="white")

# ── Two-row layout via rank=same columns ──────────────────────────────────────
# Row 1 (top):    input → ds_01 → ds_02 → ds_05
# Row 2 (bottom): ds_13 → ss    → vae
# Columns:        col0    col1    col2    col3
#                 input   ds_01  ds_02   ds_05
#                 ds_13   ss     vae

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("input")
    s.node("ds_13")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("ds_01")
    s.node("ss")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("ds_02")
    s.node("vae")

# ── Edges ─────────────────────────────────────────────────────────────────────
# Row 1 (forward)
dot.edge("input", "ds_01")
dot.edge("ds_01", "ds_02")
dot.edge("ds_02", "ds_05")
# Snake: ds_05 → ds_13 (wrap to row 2, constraint=false so it doesn't shift ranks)
dot.edge("ds_05", "ds_13", constraint="false",
         style="dashed", color="#888888", label="  ↓")
# Row 2 (forward)
dot.edge("ds_13", "ss")
dot.edge("ss",    "vae")

out_path = str(OUT / "flux_blora_architecture")
dot.render(out_path, cleanup=True)
print(f"✓ flux_blora_architecture.png → {out_path}.png")
