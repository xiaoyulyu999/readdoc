import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure docs/_static directory exists
output_dir = "docs/_static"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "framework_overview.png")

# Initialize high-resolution canvas
fig, ax = plt.subplots(figsize=(18, 9), dpi=300)
ax.set_xlim(0, 18)
ax.set_ylim(0, 9)
ax.axis("off")

# Global Framework Title
plt.title(
    "Methodological Framework: Parameter-Efficient Adaptation, XAI Diagnostics, and Foldability Validation Pipeline",
    fontsize=14,
    fontweight="bold",
    pad=25,
)

# Helper function to render rounded process blocks
def draw_block(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.5, bold=False):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.18",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="black",
    )


# -------------------------------------------------------------
# Stage 1: Geometric Input & 3D Featurization
# -------------------------------------------------------------
ax.text(0.5, 8.2, "Stage 1: Input Featurization", fontsize=11, fontweight="bold", color="#1F4E79")
draw_block(ax, 0.5, 6.2, 3.2, 1.6, "Input 3D Backbone\nRaw PDB (Post-2021)\nNon-redundant (<30% Id)", "#EAF2F8", "#1F4E79")
draw_block(ax, 0.5, 3.8, 3.2, 1.8, "ProteinFeatures Engine\n• Pseudo-Cβ Generation\n• 25 Inter-atomic RBF Pairs\n• k-NN Spatial Graph (k=48)", "#D4E6F1", "#1F4E79")

ax.annotate("", xy=(2.1, 5.6), xytext=(2.1, 6.2), arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=2))

# -------------------------------------------------------------
# Stage 2: ProteinMPNN Backbone & LoRA Adapter
# -------------------------------------------------------------
ax.text(4.7, 8.2, "Stage 2: Model Architecture & Adaptation", fontsize=11, fontweight="bold", color="#7D6608")
# Outer Container for MPNN
mpnn_box = patches.FancyBboxPatch(
    (4.7, 1.2),
    4.0,
    6.8,
    boxstyle="round,pad=0.25",
    linewidth=1.5,
    edgecolor="#B7950B",
    facecolor="#FEFDE8",
    linestyle="--",
)
ax.add_patch(mpnn_box)
ax.text(6.7, 7.6, "ProteinMPNN Engine", ha="center", fontsize=9.5, fontweight="bold", color="#7D6608")

draw_block(ax, 5.1, 5.6, 3.2, 1.4, "EncLayer (3D Message Passing)\nSE(3)-Invariant Backbone\n[100% STRICTLY FROZEN]", "#FCF3CF", "#B7950B")
draw_block(ax, 5.1, 3.4, 3.2, 1.4, "DecLayer (Autoregressive)\nPermutation Causal Attention\n[BASE FROZEN]", "#FCF3CF", "#B7950B")

# LoRA Adapter Branch
draw_block(
    ax,
    5.1,
    1.6,
    3.2,
    1.3,
    "Decoder FFN LoRA Adapter\nΔW = B·A (r=4, α=16)\n[TRAINABLE: < 0.2% PARAMS]",
    "#D5F5E3",
    "#1E8449",
    fontsize=8,
    bold=True,
)

# Connect Stage 1 to Stage 2
ax.annotate("", xy=(4.7, 4.7), xytext=(3.7, 4.7), arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=2))

# Internal Stage 2 Connections
ax.annotate("", xy=(6.7, 4.8), xytext=(6.7, 5.6), arrowprops=dict(arrowstyle="->", color="#B7950B", lw=2))
ax.annotate("", xy=(6.7, 2.9), xytext=(6.7, 3.4), arrowprops=dict(arrowstyle="<->", color="#1E8449", lw=1.8, ls="--"))

# -------------------------------------------------------------
# Stage 3: Dual-Scale XAI Diagnostic Profiling
# -------------------------------------------------------------
ax.text(9.7, 8.2, "Stage 3: XAI Diagnostics Suite", fontsize=11, fontweight="bold", color="#78281F")
draw_block(
    ax,
    9.7,
    5.4,
    3.4,
    2.2,
    "Macro-Level Alignment\nCentered Kernel Alignment (CKA)\n\n• Layer-wise Gram HSIC\n• Quantifies Representation Drift\n• Target: CKA > 0.85 (Frozen vs LoRA)",
    "#FADBD8",
    "#78281F",
    fontsize=8,
)

draw_block(
    ax,
    9.7,
    2.0,
    3.4,
    2.5,
    "Micro-Level Feature Attribution\nCaptum (Integrated Gradients)\n\n• Attribution to 3D Coordinates X\n• Validates Geometric Sensitive Loci\n• Verifies Contact Map Saliency",
    "#FADBD8",
    "#78281F",
    fontsize=8,
)

# Connect Decoder/LoRA to XAI
ax.annotate("", xy=(9.7, 6.5), xytext=(8.7, 6.3), arrowprops=dict(arrowstyle="->", color="#78281F", lw=1.8))
ax.annotate("", xy=(9.7, 3.25), xytext=(8.7, 2.25), arrowprops=dict(arrowstyle="->", color="#78281F", lw=1.8))

# -------------------------------------------------------------
# Stage 4: ESMFold In Silico Foldability Loop
# -------------------------------------------------------------
ax.text(14.1, 8.2, "Stage 4: Structural Validation", fontsize=11, fontweight="bold", color="#145A32")
draw_block(
    ax,
    14.1,
    5.7,
    3.4,
    1.7,
    "Designed Sequences\nMultinomial Sampling (T=0.1)\nCandidate Sequence Pool (N=10)",
    "#D4EFDF",
    "#145A32",
)

draw_block(
    ax,
    14.1,
    3.4,
    3.4,
    1.6,
    "ESMFold Prediction\nSingle-Sequence Forward Folding\nPredicted 3D Coordinates (PDB)",
    "#D4EFDF",
    "#145A32",
)

draw_block(
    ax,
    14.1,
    1.2,
    3.4,
    1.6,
    "Geometric Self-Consistency\n• pLDDT > 80 (High Confidence)\n• Backbone RMSD < 2.0 Å",
    "#A9DFBF",
    "#145A32",
    fontsize=8.5,
    bold=True,
)

# Connect Model Output to Sampling
ax.annotate("", xy=(14.1, 6.55), xytext=(8.7, 4.1), arrowprops=dict(arrowstyle="->", color="#145A32", lw=2))

# Stage 4 Internal Flow
ax.annotate("", xy=(15.8, 5.0), xytext=(15.8, 5.7), arrowprops=dict(arrowstyle="->", color="#145A32", lw=2))
ax.annotate("", xy=(15.8, 2.8), xytext=(15.8, 3.4), arrowprops=dict(arrowstyle="->", color="#145A32", lw=2))

# Save image
plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight")
plt.close()

print(f"[Success] Overview diagram generated at: {output_path}")