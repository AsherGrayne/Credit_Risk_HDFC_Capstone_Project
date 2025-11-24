"""
Workflow Diagram Generator
Creates a visual representation of the Early Risk Signal System workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

def create_workflow_diagram(save_path='workflow_diagram.png'):
    """Create a visual workflow diagram of the Early Risk Signal System"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'data': '#E3F2FD',
        'process': '#BBDEFB',
        'output': '#90CAF9',
        'decision': '#FFE082',
        'action': '#C8E6C9',
        'arrow': '#424242'
    }
    
    # Title
    ax.text(5, 9.5, 'Early Risk Signal System - Workflow', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Step 1: Data Input
    box1 = FancyBboxPatch((0.5, 7.5), 1.8, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['data'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.4, 8, 'Data Sources', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.4, 7.7, '• Transactions\n• Payments\n• Account Data', 
            fontsize=9, ha='center', va='top')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.3, 8), (3.2, 8), 
                            arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow1)
    
    # Step 2: Feature Engineering
    box2 = FancyBboxPatch((3.2, 7.5), 1.8, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['process'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(4.1, 8, 'Feature Engineering', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.1, 7.7, 'Early Signals\nCreation', fontsize=9, ha='center', va='top')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5, 8), (5.9, 8), 
                            arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow2)
    
    # Step 3: Risk Scoring
    box3 = FancyBboxPatch((5.9, 7.5), 1.8, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['process'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(6.8, 8, 'Risk Scoring', fontsize=12, fontweight='bold', ha='center')
    ax.text(6.8, 7.7, 'Calculate\nRisk Score', fontsize=9, ha='center', va='top')
    
    # Arrow 3
    arrow3 = FancyArrowPatch((7.7, 8), (7.7, 6.5), 
                            arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow3)
    
    # Step 4: Flag Generation
    box4 = FancyBboxPatch((6.2, 5.5), 3, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['decision'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(7.7, 6, 'Risk Flag Generation', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.7, 5.7, 'CRITICAL | HIGH | MEDIUM | LOW', fontsize=9, ha='center', va='top')
    
    # Arrows from flag generation
    arrow4a = FancyArrowPatch((6.2, 6), (4.5, 6), 
                              arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow4a)
    
    arrow4b = FancyArrowPatch((6.2, 5.8), (4.5, 4.5), 
                              arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow4b)
    
    arrow4c = FancyArrowPatch((6.2, 5.6), (4.5, 3.2), 
                              arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow4c)
    
    arrow4d = FancyArrowPatch((9.2, 6), (9.2, 4.5), 
                              arrowstyle='->', lw=2, color=colors['arrow'])
    ax.add_patch(arrow4d)
    
    # Step 5: Predictive Model (parallel)
    box5 = FancyBboxPatch((0.5, 4.5), 1.8, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['process'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(1.4, 5, 'Predictive Model', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.4, 4.7, 'ML Validation', fontsize=9, ha='center', va='top')
    
    # Step 6: Outreach Strategies
    box6 = FancyBboxPatch((0.5, 3.2), 1.8, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['action'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box6)
    ax.text(1.4, 3.7, 'Outreach', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.4, 3.4, 'Strategies', fontsize=9, ha='center', va='top')
    
    # Step 7: Action Execution
    box7a = FancyBboxPatch((3.2, 5.5), 1.2, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#FFCDD2', 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box7a)
    ax.text(3.8, 5.9, 'CRITICAL', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.8, 5.6, 'Phone Call\n<24hrs', fontsize=8, ha='center', va='top')
    
    box7b = FancyBboxPatch((3.2, 4.5), 1.2, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#FFE0B2', 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box7b)
    ax.text(3.8, 4.9, 'HIGH', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.8, 4.6, 'Phone/Email\n<48hrs', fontsize=8, ha='center', va='top')
    
    box7c = FancyBboxPatch((3.2, 3.2), 1.2, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='#FFF9C4', 
                          edgecolor='black', linewidth=2)
    ax.add_patch(box7c)
    ax.text(3.8, 3.6, 'MEDIUM', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.8, 3.3, 'Email/SMS\n<1 week', fontsize=8, ha='center', va='top')
    
    # Step 8: Output & Monitoring
    box8 = FancyBboxPatch((5.9, 3.2), 3, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=colors['output'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box8)
    ax.text(7.4, 3.7, 'Output & Monitoring', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.4, 3.4, '• Risk Flags\n• Strategies\n• Dashboards', fontsize=9, ha='center', va='top')
    
    # Step 9: Feedback Loop
    arrow_feedback = FancyArrowPatch((7.4, 3.2), (7.4, 1.5), 
                                    arrowstyle='->', lw=2, 
                                    color='#4CAF50', linestyle='--')
    ax.add_patch(arrow_feedback)
    
    box9 = FancyBboxPatch((5.9, 0.5), 3, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor='#C8E6C9', 
                         edgecolor='black', linewidth=2)
    ax.add_patch(box9)
    ax.text(7.4, 1, 'Feedback Loop', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.4, 0.7, 'Track Outcomes\nOptimize Thresholds', fontsize=9, ha='center', va='top')
    
    # Arrow back to feature engineering
    arrow_back = FancyArrowPatch((5.9, 1), (3.2, 7.5), 
                                arrowstyle='->', lw=2, 
                                color='#4CAF50', linestyle='--', 
                                connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow_back)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor='black', label='Data Input'),
        mpatches.Patch(facecolor=colors['process'], edgecolor='black', label='Processing'),
        mpatches.Patch(facecolor=colors['decision'], edgecolor='black', label='Decision'),
        mpatches.Patch(facecolor=colors['action'], edgecolor='black', label='Action'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved workflow diagram: {save_path}")
    plt.close()


if __name__ == "__main__":
    create_workflow_diagram()

