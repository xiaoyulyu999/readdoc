import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure target static asset directory exists
output_dir = "docs/_static"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "representational_drift_diagram.png")

# Configure figure canvas
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# Title and Subtitles
plt.title(
    "Structural Prior Degradation: Full Fine-Tuning vs. LoRA Parameter-Efficient Adaptation",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

# -------------------------------------------------------------
# Panel A: Full Fine-Tuning (Structural Collapse & Drift)
# -------------------------------------------------------------
ax.text(
    0.5,
    6.3,
    "(A) Standard Full Fine-Tuning (Global Gradient Backpropagation)",
    fontsize=11,
    fontweight="bold",
    color="#B22222",
)

# Pipeline A boxes
boxes_a = [
    ("Input 3D Backbone\n(Coordinates X)", 0.5, 4.3, "#E0E0E0", "black"),
    ("ProteinFeatures\n[UNFROZEN]", 3.2, 4.3, "#FFCCCC", "#B22222"),
    ("EncLayer (GNN)\n[SE(3) DRIFT]", 5.9, 4.3, "#FFCCCC", "#B22222"),
    ("DecLayer (FFN)\n[OVERFITTED]", 8.6, 4.3, "#FFCCCC", "#B22222"),
    ("Degraded Folding\n(pLDDT < 70, RMSD > 3.5Å)", 11.3, 4.3, "#FFE5E5", "#8B0000"),
]

for label, x, y, facecolor, edgecolor in boxes_a:
    rect = patches.FancyBboxPatch(
        (x, y),
        2.2,
        1.2,
        boxstyle="round,pad=0.2",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(
        x + 1.1,
        y + 0.6,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="semibold",
    )

# Arrows A
for x in [2.7, 5.4, 8.1, 10.8]:
    ax.annotate(
        "",
        xy=(x + 0.5, 4.9),
        xytext=(x, 4.9),
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=2),
    )

# -------------------------------------------------------------
# Panel B: Proposed LoRA Adaptation (Preserved Geometric Invariants)
# -------------------------------------------------------------
ax.text(
    0.5,
    2.8,
    "(B) Proposed PEFT Architecture (Frozen Backbone + LoRA Decoder Injection)",
    fontsize=11,
    fontweight="bold",
    color="#006400",
)

# Pipeline B boxes
boxes_b = [
    ("Input 3D Backbone\n(Coordinates X)", 0.5, 0.8, "#E0E0E0", "black"),
    ("ProteinFeatures\n[100% FROZEN]", 3.2, 0.8, "#E6F2FF", "#004080"),
    ("EncLayer (GNN)\n[100% FROZEN]", 5.9, 0.8, "#E6F2FF", "#004080"),
    ("DecLayer (Base)\n[100% FROZEN]", 8.6, 0.8, "#E6F2FF", "#004080"),
    ("Stable Foldability\n(pLDDT > 80, RMSD < 2.0Å)", 11.3, 0.8, "#E6FFE6", "#006400"),
]

for label, x, y, facecolor, edgecolor in boxes_b:
    rect = patches.FancyBboxPatch(
        (x, y),
        2.2,
        1.2,
        boxstyle="round,pad=0.2",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(
        x + 1.1,
        y + 0.6,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="semibold",
    )

# Arrows B
for x in [2.7, 5.4, 8.1, 10.8]:
    ax.annotate(
        "",
        xy=(x + 0.5, 1.4),
        xytext=(x, 1.4),
        arrowprops=dict(arrowstyle="->", color="#004080", lw=2),
    )

# LoRA Adapter Branch on Decoder
lora_box = patches.FancyBboxPatch(
    (8.6, 2.3),
    2.2,
    0.8,
    boxstyle="round,pad=0.2",
    linewidth=1.5,
    edgecolor="#006400",
    facecolor="#CCFFCC",
)
ax.add_patch(lora_box)
ax.text(
    9.7,
    2.7,
    "LoRA (FFN W_in/out)\nΔW = B·A (< 0.2% params)",
    ha="center",
    va="center",
    fontsize=8,
    fontweight="bold",
    color="#006400",
)

# Connect LoRA to Decoder
ax.annotate(
    "",
    xy=(9.7, 2.0),
    xytext=(9.7, 2.3),
    arrowprops=dict(arrowstyle="<->", color="#006400", lw=1.5, ls="--"),
)

# Save to ReadTheDocs static assets folder
plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight")
plt.close()

print(f"[Success] Diagram generated and saved to: {output_path}")