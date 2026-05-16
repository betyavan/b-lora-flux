"""Generate figures for VKR presentation: bar chart + architecture diagram."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("/Users/i.betev/Desktop/DIPLOMA/output/presentation_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. BLOCK ANALYSIS BAR CHART ──────────────────────────────────────────────
# SS blocks from JSON
data = json.load(open("/Users/i.betev/Desktop/DIPLOMA/output/analysis/block_analysis_results.json"))
blocks = {}
for key, val in data["results"].items():
    name = key.rsplit("_", 1)[0]
    direction = val["direction"]
    blocks.setdefault(name, {})
    if direction == "style_inject":
        blocks[name]["style"] = val["clip_sim"]
    else:
        blocks[name]["content"] = val["clip_sim"]

# DS blocks from report (hardcoded — ds_00..ds_18, only available in markdown)
ds_raw = {
    "ds_00": (0.2603, 0.2428), "ds_01": (0.2293, 0.2249),
    "ds_02": (0.1865, 0.1890), "ds_03": (0.1864, 0.1897),
    "ds_04": (0.1861, 0.1907), "ds_05": (0.1291, 0.1457),
    "ds_06": (0.1104, 0.1240), "ds_07": (0.1207, 0.1349),
    "ds_08": (0.1210, 0.1329), "ds_09": (0.1372, 0.1453),
    "ds_10": (0.1404, 0.1557), "ds_11": (0.1247, 0.1346),
    "ds_12": (0.1333, 0.1538), "ds_13": (0.1531, 0.1570),
    "ds_14": (0.1343, 0.1439), "ds_15": (0.1358, 0.1423),
    "ds_16": (0.1486, 0.1467), "ds_17": (0.1202, 0.1317),
    "ds_18": (0.1626, 0.1594),
}
for name, (s, c) in ds_raw.items():
    blocks[name] = {"style": s, "content": c}

ds = sorted([(k, v) for k, v in blocks.items() if k.startswith("ds_")],
            key=lambda x: int(x[0].split("_")[1]))
ss = sorted([(k, v) for k, v in blocks.items() if k.startswith("ss_")],
            key=lambda x: int(x[0].split("_")[1]))

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Чувствительность блоков FLUX.1-dev к стилю и контенту\n(CLIP-embedding injection, n=200)",
             fontsize=13, fontweight="bold")

COLOR_STYLE = "#E07B39"
COLOR_CONTENT = "#4A90D9"

for ax, group, title in [(axes[0], ds, "Double-Stream блоки (DS 00–18)"),
                          (axes[1], ss, "Single-Stream блоки (SS 00–37)")]:
    names = [k for k, _ in group]
    style_scores = [v.get("style", 0) for _, v in group]
    content_scores = [v.get("content", 0) for _, v in group]
    x = np.arange(len(names))
    w = 0.38

    bars_s = ax.bar(x - w/2, style_scores, w, color=COLOR_STYLE, alpha=0.85, label="Style score")
    bars_c = ax.bar(x + w/2, content_scores, w, color=COLOR_CONTENT, alpha=0.85, label="Content score")

    # Highlight key blocks
    for i, name in enumerate(names):
        if name in ("ds_00", "ds_01"):
            ax.bar(i - w/2, style_scores[i], w, color=COLOR_STYLE, alpha=1.0, linewidth=2,
                   edgecolor="red")
            ax.annotate("Стиль ★", (i - w/2, style_scores[i] + 0.002), ha="center",
                        fontsize=7, color="red", fontweight="bold")
        if name in ("ds_12",):
            ax.bar(i + w/2, content_scores[i], w, color=COLOR_CONTENT, alpha=1.0, linewidth=2,
                   edgecolor="darkblue")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CLIP cosine similarity")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(style_scores), max(content_scores)) * 1.12)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "block_analysis_chart.png", dpi=150, bbox_inches="tight")
print("✓ block_analysis_chart.png")
plt.close()

# ── 2. RESULTS BAR CHART (Phase 1 metrics) ──────────────────────────────────
experiments = ["e00\n(без LoRA)", "d01\n(1K шагов\nr=16)", "d02\n(2K шагов\nr=16)", "d03\n(1K шагов\nr=32)"]
dino_style  = [0.046, 0.191, 0.162, 0.102]
clip_style  = [0.463, 0.486, 0.483, 0.467]
fid         = [283.3, 246.3, 257.5, 271.8]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Метрики Phase 1 — Диагностическая серия D (стиль: Ван Гог)", fontsize=12, fontweight="bold")

x = np.arange(len(experiments))
colors = ["#AAAAAA", "#E07B39", "#F5C06B", "#F0A080"]

for ax, vals, ylabel, title, higher_better in [
    (axes[0], dino_style, "DINO-style ↑", "DINO-style\n(стилевое сходство)", True),
    (axes[1], clip_style, "CLIP-style ↑", "CLIP-style\n(семантика стиля)", True),
    (axes[2], fid,        "FID ↓",         "FID ↓\n(качество распределения)", False),
]:
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.2)
    # Highlight best
    best_idx = np.argmin(vals) if not higher_better else np.argmax(vals)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(vals)*0.01),
                f"{val:.3f}" if val < 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ypad = max(vals) * 0.15
    ax.set_ylim(0, max(vals) + ypad)

plt.tight_layout()
plt.savefig(OUT / "phase1_metrics.png", dpi=150, bbox_inches="tight")
print("✓ phase1_metrics.png")
plt.close()

# ── 3. FLUX ARCHITECTURE + B-LoRA DIAGRAM ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_facecolor("#F8F8F8")
fig.patch.set_facecolor("#F8F8F8")

ax.text(6, 6.6, "Архитектура FLUX.1 и расположение B-LoRA адаптеров",
        ha="center", va="center", fontsize=13, fontweight="bold")

def box(ax, x, y, w, h, color, text, fontsize=9, textcolor="white"):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="white", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold", wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

# Input
box(ax, 0.3, 3.0, 1.6, 0.7, "#555566", "Текстовый\nпромпт + шум", fontsize=8)
arrow(ax, 1.9, 3.35, 2.3, 3.35)

# DS blocks 00-01 — style zone
box(ax, 2.3, 4.2, 1.8, 1.3, "#C0392B", "DS 00–01\n(Double-Stream)\n★ СТИЛЬ", fontsize=8)
arrow(ax, 3.2, 3.0, 3.2, 4.2)   # from input up

# DS blocks 02-04 — neutral
box(ax, 2.3, 3.0, 1.8, 1.0, "#7F8C8D", "DS 02–04\n(нейтральные)", fontsize=8)

# DS blocks 05-12 — content zone
box(ax, 2.3, 1.5, 1.8, 1.3, "#2980B9", "DS 05–12\n(Double-Stream)\n★ КОНТЕНТ", fontsize=8)
arrow(ax, 3.2, 3.0, 3.2, 2.8)   # down

# DS blocks 13-18
box(ax, 2.3, 0.5, 1.8, 0.8, "#7F8C8D", "DS 13–18\n(остальные DS)", fontsize=8)

# arrow DS→SS
arrow(ax, 4.1, 3.0, 4.7, 3.35)

# SS blocks
box(ax, 4.7, 2.5, 1.9, 1.7, "#616A6B", "SS 00–37\n(Single-Stream\n38 блоков)\nсмешанная роль", fontsize=8)

# Output
arrow(ax, 6.6, 3.35, 7.0, 3.35)
box(ax, 7.0, 3.0, 1.5, 0.7, "#555566", "Декодер VAE\n→ изображение", fontsize=8)

# B-LoRA style adapter annotation
rect_style = mpatches.FancyBboxPatch((2.15, 4.1), 2.1, 1.55,
    boxstyle="round,pad=0.1", facecolor="none", edgecolor="#C0392B", linewidth=2.5, linestyle="--")
ax.add_patch(rect_style)
ax.text(4.4, 5.15, "LoRA-Style\n(адаптер стиля)", ha="left", va="center",
        fontsize=9, color="#C0392B", fontweight="bold")

# B-LoRA content adapter annotation
rect_content = mpatches.FancyBboxPatch((2.15, 1.4), 2.1, 1.45,
    boxstyle="round,pad=0.1", facecolor="none", edgecolor="#2980B9", linewidth=2.5, linestyle="--")
ax.add_patch(rect_content)
ax.text(4.4, 2.1, "LoRA-Content\n(адаптер контента)", ha="left", va="center",
        fontsize=9, color="#2980B9", fontweight="bold")

# Legend
ax.text(7.2, 5.8, "B-LoRA для FLUX.1", fontsize=11, fontweight="bold", color="#333")
ax.text(7.2, 5.3, "Стиль → ds_00, ds_01", fontsize=9, color="#C0392B")
ax.text(7.2, 4.9, "Контент → ds_05–ds_12", fontsize=9, color="#2980B9")
ax.text(7.2, 4.4, "Основание: блок-анализ\n(CLIP-embedding injection\nn=200 промпт-пар)", fontsize=8, color="#555")

# FLUX label
ax.text(3.2, 6.3, "FLUX.1-dev  (MMDiT: 19 DS + 38 SS блоков)", ha="center",
        fontsize=9, color="#333", style="italic")

plt.tight_layout()
plt.savefig(OUT / "flux_blora_architecture.png", dpi=150, bbox_inches="tight")
print("✓ flux_blora_architecture.png")
plt.close()

print(f"\nВсе файлы сохранены в {OUT}")
