"""
Generate all architectural, design, and flow diagrams for the SynCura Mini Project Report.
Outputs high-resolution PNGs to PROJ/REPORT/assets/ matching the VTU report format.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

ASSETS_DIR = r"C:\Users\fizan\Downloads\Techfusion\PROJ\REPORT\assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Common styling constants
NAVY = "#1B365D"
TEAL = "#028090"
ACCENT_BLUE = "#3B82F6"
LIGHT_BG = "#F1F5F9"
BORDER_COL = "#64748B"
TEXT_DARK = "#0F172A"
BOX_WHITE = "#FFFFFF"
RED_ALERT = "#EF4444"
GREEN_OK = "#10B981"
AMBER_WARN = "#F59E0B"

def save_fig(fig, filename):
    path = os.path.join(ASSETS_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated {filename}")

# ==============================================================================
# Diagram 1: Figure 1.1 - Overview of User & Admin/Clinician Modules
# ==============================================================================
def draw_fig1_1():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)

    # Core System Box
    box_core = FancyBboxPatch((3.2, 1.2), 3.6, 3.2, boxstyle="round,pad=0.2",
                              facecolor="#EFF6FF", edgecolor=NAVY, linewidth=2)
    ax.add_patch(box_core)
    ax.text(5.0, 4.0, "SynCura Core Engine", ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAVY)
    ax.text(5.0, 3.4, "• Sliding Window Buffer (90-min)\n• Imputation & Normalization\n• Ensemble Attention-LSTM\n• SHAP Explainability Engine\n• Two-Stage Persistence Filter",
            ha="center", va="center", fontsize=9.5, color=TEXT_DARK, linespacing=1.4)

    # Left Box: Clinical User Module
    box_user = FancyBboxPatch((0.4, 1.2), 2.2, 3.2, boxstyle="round,pad=0.15",
                              facecolor="#F0FDF4", edgecolor=GREEN_OK, linewidth=1.8)
    ax.add_patch(box_user)
    ax.text(1.5, 4.0, "User Module\n(ICU Clinicians)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#065F46")
    ax.text(1.5, 2.6, "• Live Bed Board\n• Real-Time Risk Score\n• Vital Sign Sparklines\n• SHAP Waterfall Plot\n• Escalating Siren Alert\n• Alarm Acknowledgment",
            ha="center", va="center", fontsize=8.5, color=TEXT_DARK, linespacing=1.3)

    # Right Box: Administrative / Engineering Module
    box_admin = FancyBboxPatch((7.4, 1.2), 2.2, 3.2, boxstyle="round,pad=0.15",
                               facecolor="#FEF2F2", edgecolor=RED_ALERT, linewidth=1.8)
    ax.add_patch(box_admin)
    ax.text(8.5, 4.0, "Admin / Research\nModule", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#991B1B")
    ax.text(8.5, 2.6, "• Telemetry Simulation\n• Scenario Injection\n• Model Threshold Config\n• Benchmark Logging\n• Database Audit Logs\n• System Diagnostics",
            ha="center", va="center", fontsize=8.5, color=TEXT_DARK, linespacing=1.3)

    # Connecting Arrows
    ax.annotate("", xy=(3.2, 3.0), xytext=(2.6, 3.0),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2))
    ax.annotate("", xy=(7.4, 3.0), xytext=(6.8, 3.0),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2))

    save_fig(fig, "fig1_1_overview_modules.png")

# ==============================================================================
# Diagram 2: Figure 3.1 - System Architecture Diagram
# ==============================================================================
def draw_fig3_1():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)

    layers = [
        ("Layer 1: Telemetry Data Ingestion", 6.2, "#EFF6FF", NAVY,
         "Bedside Monitors (HR, RR, Temp, SysBP, DiasBP, SpO2)  +  Laboratory Results (GCS, BUN, Creatinine, WBC, Platelets, Glucose)"),
        ("Layer 2: Preprocessing & Ring Buffer", 4.9, "#F8FAFC", "#475569",
         "90-Minute Rolling Window (18 timesteps @ 5-min intervals) | Forward-Fill & Median Imputation | Z-Score Normalization"),
        ("Layer 3: Ensemble Deep Learning Engine", 3.6, "#F0FDF4", "#065F46",
         "Triple Model Ensemble: Model s48 (Hidden 96) + Model c93 (Hidden 96) + Model s45 (PhysioNet 2019 Warm-start)\nLogit-Averaged Ensemble Inference (AUROC = 0.840)"),
        ("Layer 4: Explainability & Alarm Verification", 2.3, "#FFFBEB", "#92400E",
         "Temporal Self-Attention Weights (Timestep Salience) + KernelSHAP (Feature Contribution)\nTwo-Stage Persistence Filter (2 Consecutive Triggers -> 76.4% False Alarm Reduction)"),
        ("Layer 5: Presentation & Clinical Dashboard", 1.0, "#FAF5FF", "#581C87",
         "FastAPI Asynchronous Gateway (REST + WebSockets) -> React 19 Client Dashboard\nLive Multi-Bed Ward Grid | Dual-Tier Audio Siren | Interactive SHAP Waterfall Visualizer")
    ]

    for title, y, bg, border, content in layers:
        b = FancyBboxPatch((0.6, y - 0.5), 9.8, 1.0, boxstyle="round,pad=0.15",
                           facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(b)
        ax.text(5.5, y + 0.22, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=border)
        ax.text(5.5, y - 0.16, content, ha="center", va="center",
                fontsize=8.5, color=TEXT_DARK, linespacing=1.2)

    # Downward connectors between layers
    for y in [5.7, 4.4, 3.1, 1.8]:
        ax.annotate("", xy=(5.5, y - 0.4), xytext=(5.5, y),
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))

    save_fig(fig, "fig3_1_system_architecture.png")

# ==============================================================================
# Diagram 3: Figure 3.2 - Context Flow Diagram (DFD Level 0 & Level 1)
# ==============================================================================
def draw_fig3_2():
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6)

    # External Entities
    # Patient / Sensors
    p_box = FancyBboxPatch((0.5, 3.6), 2.2, 1.4, boxstyle="square,pad=0.1",
                           facecolor="#F1F5F9", edgecolor=NAVY, linewidth=2)
    ax.add_patch(p_box)
    ax.text(1.6, 4.3, "ICU Bedside\nSensors & Labs", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=NAVY)

    # Clinician / Nurse
    c_box = FancyBboxPatch((7.8, 3.6), 2.2, 1.4, boxstyle="square,pad=0.1",
                           facecolor="#F1F5F9", edgecolor=NAVY, linewidth=2)
    ax.add_patch(c_box)
    ax.text(8.9, 4.3, "ICU Clinician /\nIntensivist", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=NAVY)

    # Central SynCura Process (Bubble)
    circle = patches.Circle((5.25, 4.3), 1.25, facecolor="#EFF6FF", edgecolor=TEAL, linewidth=2.5)
    ax.add_patch(circle)
    ax.text(5.25, 4.45, "0.0\nSynCura System", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(5.25, 3.75, "(Real-Time ICU\nRisk Scoring)", ha="center", va="center",
            fontsize=8.5, color=TEXT_DARK)

    # Database Store
    db_box = FancyBboxPatch((3.8, 0.8), 2.9, 1.1, boxstyle="round,pad=0.1",
                            facecolor="#FFFBEB", edgecolor="#B45309", linewidth=1.8)
    ax.add_patch(db_box)
    ax.text(5.25, 1.35, "D1: Vitals & Prediction Store\n(vitals.db / Cache)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#92400E")

    # Arrows
    # Sensors -> SynCura
    ax.annotate("Raw Vitals & Labs\n(Every 5 min)", xy=(4.0, 4.5), xytext=(2.7, 4.5),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8),
                fontsize=8.5, ha="center", va="bottom")

    # SynCura -> Clinician
    ax.annotate("Risk Score (0-100),\nSHAP Alerts & Audio", xy=(7.8, 4.5), xytext=(6.5, 4.5),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8),
                fontsize=8.5, ha="center", va="bottom")

    # Clinician -> SynCura
    ax.annotate("Alarm Acknowledge /\nThreshold Controls", xy=(6.5, 3.9), xytext=(7.8, 3.9),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8),
                fontsize=8.5, ha="center", va="top")

    # SynCura <-> DB
    ax.annotate("", xy=(4.8, 1.9), xytext=(4.8, 3.1),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax.annotate("", xy=(5.7, 3.1), xytext=(5.7, 1.9),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax.text(5.25, 2.5, "Read Window /\nLog Prediction", fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))

    save_fig(fig, "fig3_2_context_flow_diagram.png")

# ==============================================================================
# Diagram 4: Figure 3.3 - UML Use Case Diagram
# ==============================================================================
def draw_fig3_3():
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.5)

    # System boundary box
    sys_rect = FancyBboxPatch((2.6, 0.4), 5.3, 5.7, boxstyle="round,pad=0.2",
                              facecolor="#FAFAFA", edgecolor=NAVY, linewidth=2)
    ax.add_patch(sys_rect)
    ax.text(5.25, 5.85, "SynCura ICU Monitoring System", ha="center", va="center",
            fontsize=12, fontweight="bold", color=NAVY)

    use_cases = [
        ("UC1: Stream Real-Time ICU Vitals", 5.1),
        ("UC2: View Multi-Bed Risk Grid", 4.2),
        ("UC3: Inspect Attention & SHAP Breakdown", 3.3),
        ("UC4: Acknowledge / Silence Critical Alarm", 2.4),
        ("UC5: Simulate Clinical Deterioration Scenarios", 1.5),
        ("UC6: Configure Persistence Thresholds", 0.6)
    ]

    for title, y in use_cases:
        uc = patches.Ellipse((5.25, y), 4.2, 0.65, facecolor="#EFF6FF",
                             edgecolor=TEAL, linewidth=1.5)
        ax.add_patch(uc)
        ax.text(5.25, y, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=TEXT_DARK)

    # Actor 1: ICU Clinician (Left)
    ax.text(1.2, 3.8, "[Actor]\nICU Clinician /\nNurse", ha="center", va="center",
            fontsize=11, fontweight="bold", color=NAVY)

    # Actor 2: System Admin / Researcher (Right)
    ax.text(9.3, 1.8, "[Actor]\nSystem Admin /\nResearcher", ha="center", va="center",
            fontsize=11, fontweight="bold", color=NAVY)

    # Connect Clinician to UC1, UC2, UC3, UC4
    for y in [5.1, 4.2, 3.3, 2.4]:
        ax.plot([1.9, 3.15], [3.8, y], color=NAVY, lw=1.5, linestyle="-")

    # Connect Admin to UC5, UC6
    for y in [1.5, 0.6]:
        ax.plot([8.5, 7.35], [1.8, y], color=NAVY, lw=1.5, linestyle="-")

    save_fig(fig, "fig3_3_use_case_diagram.png")

# ==============================================================================
# Diagram 5: Figure 3.4 - UML Sequence Diagram
# ==============================================================================
def draw_fig3_4():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)

    lifelines = [
        ("Bedside Monitor", 1.0),
        ("FastAPI Streamer", 3.0),
        ("Window Preprocessor", 5.0),
        ("Ensemble Model", 7.0),
        ("SHAP Explainer", 8.8),
        ("React Dashboard", 10.2)
    ]

    # Draw lifeline headers and vertical lines
    for name, x in lifelines:
        box = FancyBboxPatch((x - 0.8, 6.4), 1.6, 0.6, boxstyle="square,pad=0.1",
                             facecolor="#EFF6FF", edgecolor=NAVY, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 6.7, name, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=NAVY)
        ax.plot([x, x], [0.5, 6.4], color="#94A3B8", linestyle="--", lw=1.2)

    messages = [
        (6.0, 1.0, 3.0, "1: Push 5-min Vitals Packet (POST)"),
        (5.3, 3.0, 5.0, "2: Append & Extract 90-min Window"),
        (4.6, 5.0, 5.0, "3: Forward-Fill & Normalize (Z-Score)", True),
        (3.9, 5.0, 7.0, "4: Pass Tensor (Batch, 18, 12)"),
        (3.2, 7.0, 7.0, "5: BiLSTM + Attention + Ensemble Logits", True),
        (2.5, 7.0, 8.8, "6: Trigger SHAP Kernel on Top Activations"),
        (1.8, 8.8, 3.0, "7: Return Risk Score + Attributions"),
        (1.1, 3.0, 10.2, "8: Broadcast WebSocket (Risk, SHAP, Cues)")
    ]

    for y, x1, x2, text, *self_call in messages:
        if self_call and self_call[0]:
            # Self call loop
            ax.annotate("", xy=(x1, y - 0.3), xytext=(x1 + 0.5, y - 0.15),
                        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
            ax.plot([x1, x1 + 0.5, x1 + 0.5], [y, y, y - 0.15], color=TEAL, lw=1.5)
            ax.text(x1 + 0.55, y - 0.1, text, fontsize=7.5, color=TEXT_DARK, va="center")
        else:
            ax.annotate("", xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, y + 0.12, text, fontsize=7.5, color=TEXT_DARK, ha="center")

    save_fig(fig, "fig3_4_sequence_diagram.png")

# ==============================================================================
# Diagram 6: Figure 3.5 - System Flow Chart
# ==============================================================================
def draw_fig3_5():
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axis("off")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 8)

    steps = [
        ("Start: Continuous ICU Stream", 7.4, "round", "#EFF6FF", NAVY),
        ("Receive Vitals & Lab Telemetry", 6.5, "square", "#F8FAFC", BORDER_COL),
        ("Append to 90-min Ring Buffer (18 steps)", 5.6, "square", "#F8FAFC", BORDER_COL),
        ("Impute Missing Data & Apply Normalization", 4.7, "square", "#F8FAFC", BORDER_COL),
        ("Execute Ensemble Inference (s48, c93, s45)", 3.8, "square", "#F0FDF4", "#065F46"),
        ("Compute Temporal Attention & SHAP Values", 2.9, "square", "#FFFBEB", "#92400E"),
        ("Check Persistence Threshold (>= 2 hits)", 1.9, "diamond", "#FEF2F2", RED_ALERT),
        ("Update Dashboard & Dispatch Siren Alert", 0.8, "round", "#FEE2E2", RED_ALERT)
    ]

    for text, y, shape, bg, border in steps:
        if shape == "round":
            b = FancyBboxPatch((2.2, y - 0.3), 4.6, 0.6, boxstyle="round,pad=0.2",
                               facecolor=bg, edgecolor=border, linewidth=1.8)
        elif shape == "diamond":
            b = FancyBboxPatch((2.0, y - 0.35), 5.0, 0.7, boxstyle="square,pad=0.15",
                               facecolor=bg, edgecolor=border, linewidth=2)
        else:
            b = FancyBboxPatch((2.0, y - 0.3), 5.0, 0.6, boxstyle="square,pad=0.1",
                               facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(b)
        ax.text(4.5, y, text, ha="center", va="center", fontsize=9,
                fontweight="bold", color=border if shape in ["round", "diamond"] else TEXT_DARK)

    # Vertical flow arrows
    for y in [7.1, 6.2, 5.3, 4.4, 3.5, 2.6]:
        ax.annotate("", xy=(4.5, y - 0.3), xytext=(4.5, y),
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))

    # Diamond condition arrows
    ax.annotate("Yes (High Risk)", xy=(4.5, 1.1), xytext=(4.5, 1.55),
                arrowprops=dict(arrowstyle="->", color=RED_ALERT, lw=2),
                fontsize=8, color=RED_ALERT, ha="left")

    ax.annotate("No (Transient/Normal)", xy=(7.0, 1.9), xytext=(7.8, 1.9),
                arrowprops=dict(arrowstyle="->", color=GREEN_OK, lw=1.5),
                fontsize=8, color=GREEN_OK, va="bottom")
    ax.plot([7.8, 7.8, 4.5], [1.9, 0.4, 0.4], color=GREEN_OK, lw=1.5)
    ax.annotate("", xy=(4.5, 0.5), xytext=(4.5, 0.4),
                arrowprops=dict(arrowstyle="->", color=GREEN_OK, lw=1.5))

    save_fig(fig, "fig3_5_system_flowchart.png")

# ==============================================================================
# Diagram 7: Figure 4.1 - Structured Chart / Decomposition Diagram
# ==============================================================================
def draw_fig4_1():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)

    # Root Box
    root = FancyBboxPatch((4.0, 5.0), 3.0, 0.8, boxstyle="round,pad=0.15",
                          facecolor=NAVY, edgecolor="black", linewidth=1.5)
    ax.add_patch(root)
    ax.text(5.5, 5.4, "1.0 SynCura Predictive\nMonitoring System", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")

    # Level 1 Modules
    l1_mods = [
        ("1.1 Data Ingestion\n& Buffering", 1.4),
        ("1.2 Preprocessing\n& Imputation", 4.1),
        ("1.3 Ensemble Deep\nLearning Engine", 6.8),
        ("1.4 Explainability\n& Clinical UI", 9.5)
    ]

    for title, x in l1_mods:
        b = FancyBboxPatch((x - 1.2, 3.4), 2.4, 0.8, boxstyle="square,pad=0.1",
                           facecolor="#EFF6FF", edgecolor=NAVY, linewidth=1.5)
        ax.add_patch(b)
        ax.text(x, 3.8, title, ha="center", va="center", fontsize=9,
                fontweight="bold", color=NAVY)
        # Line from root to level 1
        ax.plot([5.5, x], [5.0, 4.2], color=NAVY, lw=1.5)

    # Level 2 Sub-modules
    l2_subs = [
        ("1.1.1 Vitals Receiver\n1.1.2 90-min Ring Queue", 1.4),
        ("1.2.1 Forward-Fill Imp.\n1.2.2 Normalization", 4.1),
        ("1.3.1 AttentionLSTM\n1.3.2 Logit Ensemble", 6.8),
        ("1.3.1 SHAP Attributor\n1.4.2 React Live Ward", 9.5)
    ]

    for text, x in l2_subs:
        b = FancyBboxPatch((x - 1.2, 1.6), 2.4, 0.9, boxstyle="square,pad=0.1",
                           facecolor="#F8FAFC", edgecolor=BORDER_COL, linewidth=1.2)
        ax.add_patch(b)
        ax.text(x, 2.05, text, ha="center", va="center", fontsize=8,
                color=TEXT_DARK, linespacing=1.3)
        ax.plot([x, x], [3.4, 2.5], color=BORDER_COL, lw=1.2)

    save_fig(fig, "fig4_1_structured_chart.png")

if __name__ == "__main__":
    print("Generating all 7 diagrams...")
    draw_fig1_1()
    draw_fig3_1()
    draw_fig3_2()
    draw_fig3_3()
    draw_fig3_4()
    draw_fig3_5()
    draw_fig4_1()
    print("All diagrams generated successfully!")
