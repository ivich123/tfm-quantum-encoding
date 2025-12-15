"""
Generate comprehensive visualizations for TFM:
1. Comparative table (heatmap)
2. Gradient norms evolution
3. Complexity vs Performance scatter
4. Dataset gap analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style configuration
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Color palette (colorblind-friendly)
COLORS = {
    'baseline': '#000000',      # Black
    'angle': '#0173B2',         # Blue
    'feature_map': '#029E73',   # Green
    'basis': '#DE8F05',         # Orange
    'hybrid_angle_zz': '#CC78BC', # Purple
}

LABELS = {
    'baseline': 'Baseline',
    'angle': 'Angle',
    'feature_map': 'Feature Map',
    'basis': 'Basis',
    'hybrid_angle_zz': 'Hybrid Angle+ZZ',
}

def load_experiment_data(dataset, encoding):
    """Load CSV for a specific experiment."""
    pattern = f"{encoding}_{dataset}_*.csv"
    files = list(METRICS_DIR.glob(pattern))
    
    # Fallback: search for any file with encoding and dataset in name
    if not files:
        pattern_alt = f"*{encoding}*{dataset}*.csv"
        files = list(METRICS_DIR.glob(pattern_alt))
    
    if not files:
        print(f"Warning: No file found for {encoding} on {dataset}")
        return None
    
    # Read CSV, handle malformed test row
    df = pd.read_csv(files[0], on_bad_lines='skip', engine='python')
    
    # Filter out test row if it exists
    if 'epoch' in df.columns:
        df = df[df['epoch'] != 'test'].copy()
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df = df.dropna(subset=['epoch'])
        df['epoch'] = df['epoch'].astype(int)
    
    # Convert numeric columns
    numeric_cols = ['val_acc', 'train_acc', 'val_precision', 'quantum_grad_norm', 
                    'epoch_time_s', 'quantum_circuit_time_ms', 'circuit_depth', 'total_gates', 'cnot_gates']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def load_all_experiments():
    """Load all experiment results into a summary dataframe."""
    datasets = ['mnist', 'cifar10']
    encodings = ['baseline', 'angle', 'feature_map', 'basis', 'hybrid_angle_zz']
    
    results = []
    
    for dataset in datasets:
        for encoding in encodings:
            df = load_experiment_data(dataset, encoding)
            
            if df is not None and len(df) > 0:
                # Get final epoch metrics
                final = df.iloc[-1]
                
                # Get average gradient norm (last 5 epochs)
                avg_grad = df['quantum_grad_norm'].tail(5).mean() if 'quantum_grad_norm' in df.columns else 0
                
                # Get average quantum circuit time (last 5 epochs)
                avg_quantum_time = df['quantum_circuit_time_ms'].tail(5).mean() if 'quantum_circuit_time_ms' in df.columns else 0
                
                results.append({
                    'dataset': dataset.upper(),
                    'encoding': LABELS[encoding],
                    'encoding_key': encoding,
                    'val_acc': final.get('val_acc', 0) * 100,
                    'train_acc': final.get('train_acc', 0) * 100,
                    'val_precision': final.get('val_precision', 0) * 100,
                    'grad_norm': avg_grad,
                    'circuit_depth': final.get('circuit_depth', 0),
                    'total_gates': final.get('total_gates', 0),
                    'cnot_gates': final.get('cnot_gates', 0),
                    'quantum_time_ms': avg_quantum_time,
                })
    
    return pd.DataFrame(results)

# ============================================================================
# 1. COMPARATIVE TABLE WITH HEATMAP
# ============================================================================

def plot_comparative_table(df_summary):
    """Create a comprehensive comparison table with color-coded cells."""
    
    # Pivot for better display
    metrics = ['val_acc', 'grad_norm', 'total_gates', 'circuit_depth', 'quantum_time_ms']
    metric_labels = ['Val Acc (%)', 'Grad Norm', 'Gates', 'Depth', 'Circ. Time (ms)']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    
    for dataset in ['MNIST', 'CIFAR10']:
        df_dataset = df_summary[df_summary['dataset'] == dataset]
        
        # Sort by validation accuracy (descending)
        df_dataset = df_dataset.sort_values('val_acc', ascending=False)
        
        for _, row in df_dataset.iterrows():
            table_data.append([
                dataset,
                row['encoding'],
                f"{row['val_acc']:.1f}",
                f"{row['grad_norm']:.4f}",
                f"{int(row['total_gates'])}",
                f"{int(row['circuit_depth'])}",
                f"{row['quantum_time_ms']:.2f}",
            ])
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Dataset', 'Encoding'] + metric_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(metric_labels) + 2):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Color cells by performance (green=good, red=bad)
    for i, row in enumerate(table_data, start=1):
        # Accuracy: higher is better (green)
        val_acc = float(row[2])
        acc_color = plt.cm.RdYlGn(val_acc / 100)
        table[(i, 2)].set_facecolor(acc_color)
        
        # Gradient norm: higher is better for learning
        grad = float(row[3])
        if grad > 0.01:
            grad_color = '#90EE90'  # Light green
        elif grad > 0.001:
            grad_color = '#FFFFE0'  # Light yellow
        else:
            grad_color = '#FFB6C1'  # Light red
        table[(i, 3)].set_facecolor(grad_color)
        
        # Gates/Depth: lower is better (inverse)
        gates = int(row[4])
        gate_color = plt.cm.RdYlGn_r(gates / 60)
        table[(i, 4)].set_facecolor(gate_color)
        
        depth = int(row[5])
        depth_color = plt.cm.RdYlGn_r(depth / 30)
        table[(i, 5)].set_facecolor(depth_color)
        
        # Quantum time: lower is better
        qtime = float(row[5])
        if qtime == 0:  # Baseline
            qtime_color = '#D3D3D3'  # Gray for N/A
        else:
            qtime_color = plt.cm.RdYlGn_r(qtime / 10)
        table[(i, 6)].set_facecolor(qtime_color)
    
    plt.title('Comparative Analysis of Quantum Encodings', 
              fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "comparative_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    output_path_pdf = FIGURES_DIR / "comparative_table.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")
    
    plt.close()

# ============================================================================
# 2. GRADIENT NORMS EVOLUTION
# ============================================================================

def plot_gradient_norms():
    """Plot gradient norm evolution separated by dataset."""
    
    datasets = ['mnist', 'cifar10']
    # Only quantum encodings (baseline has no gradients)
    encodings = ['angle', 'feature_map', 'basis', 'hybrid_angle_zz']
    
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for encoding in encodings:
            df = load_experiment_data(dataset, encoding)
            
            if df is not None and 'quantum_grad_norm' in df.columns:
                # Filter out zero gradients for log scale
                grad_norms = df['quantum_grad_norm'].replace(0, np.nan)
                
                ax.plot(
                    df['epoch'],
                    grad_norms,
                    color=COLORS[encoding],
                    linewidth=2.5,
                    label=LABELS[encoding],
                    marker='o' if encoding == 'basis' else None,
                    markersize=4,
                    alpha=0.85
                )
        
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
        ax.set_ylabel('Gradient Norm (log scale)', fontweight='bold', fontsize=13)
        ax.set_title(f'{dataset.upper()} - Quantum Gradient Norm Evolution', 
                     fontweight='bold', fontsize=15)
        ax.legend(loc='best', framealpha=0.95, fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_yscale('log')
        
        # Add annotation for dead gradients threshold
        ax.axhline(y=0.0001, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.text(15, 0.00012, 'Dead gradient threshold', 
                color='red', alpha=0.7, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual figures
        output_path = FIGURES_DIR / f"gradient_norms_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        output_path_pdf = FIGURES_DIR / f"gradient_norms_{dataset}.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"✓ Saved: {output_path_pdf}")
        
        plt.close()

# ============================================================================
# 3. COMPLEXITY VS PERFORMANCE
# ============================================================================

def plot_complexity_vs_performance(df_summary):
    """Scatter plot: circuit complexity vs accuracy."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    datasets = ['MNIST', 'CIFAR10']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        df_dataset = df_summary[df_summary['dataset'] == dataset]
        
        for encoding_key in ['baseline', 'angle', 'feature_map', 'basis', 'hybrid_angle_zz']:
            row = df_dataset[df_dataset['encoding_key'] == encoding_key]
            
            if len(row) > 0:
                row = row.iloc[0]
                
                ax.scatter(
                    row['circuit_depth'],
                    row['val_acc'],
                    s=row['total_gates'] * 10,  # Size proportional to gates
                    color=COLORS[encoding_key],
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=1.5
                )
                
                # Add text annotation
                ax.annotate(
                    LABELS[encoding_key],
                    (row['circuit_depth'], row['val_acc']),
                    xytext=(10, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    alpha=0.9
                )
        
        ax.set_xlabel('Circuit Depth', fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
        ax.set_title(f'{dataset} - Complexity vs Performance', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "complexity_vs_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    output_path_pdf = FIGURES_DIR / "complexity_vs_performance.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")
    
    plt.close()

# ============================================================================
# 4. DATASET GAP ANALYSIS
# ============================================================================

def plot_dataset_gap(df_summary):
    """Bar plot showing performance gap between baseline and quantum encodings."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = ['MNIST', 'CIFAR10']
    encodings = ['angle', 'feature_map', 'basis', 'hybrid_angle_zz']
    
    # Calculate gaps
    gaps = []
    labels = []
    colors_list = []
    
    for dataset in datasets:
        df_dataset = df_summary[df_summary['dataset'] == dataset]
        baseline_acc = df_dataset[df_dataset['encoding_key'] == 'baseline']['val_acc'].values[0]
        
        for encoding in encodings:
            quantum_acc = df_dataset[df_dataset['encoding_key'] == encoding]['val_acc'].values
            
            if len(quantum_acc) > 0:
                gap = baseline_acc - quantum_acc[0]
                gaps.append(gap)
                labels.append(f"{dataset}\n{LABELS[encoding]}")
                colors_list.append(COLORS[encoding])
    
    # Create bar plot
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, gaps, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Accuracy Gap (Baseline - Quantum) [%]', fontweight='bold')
    ax.set_title('Performance Gap: Classical vs Quantum Encodings', fontweight='bold', fontsize=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Add reference lines
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(len(labels)-0.5, 21, 'Moderate gap', color='orange', alpha=0.7, fontsize=9)
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(len(labels)-0.5, 52, 'Large gap', color='red', alpha=0.7, fontsize=9)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "dataset_gap_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    output_path_pdf = FIGURES_DIR / "dataset_gap_analysis.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")
    
    plt.close()

# ============================================================================
# 5. ACCURACY TIMELINE (separated by dataset)
# ============================================================================

def plot_accuracy_timeline():
    """Create separate accuracy timeline plots for MNIST and CIFAR-10."""
    
    datasets = ['mnist', 'cifar10']
    encodings = ['baseline', 'angle', 'feature_map', 'basis', 'hybrid_angle_zz']
    
    for dataset in datasets:
        # Create individual figure for each dataset
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for encoding in encodings:
            df = load_experiment_data(dataset, encoding)
            
            if df is not None:
                # Plot validation accuracy
                ax.plot(
                    df['epoch'], 
                    df['val_acc'] * 100,  # Convert to percentage
                    color=COLORS[encoding],
                    linewidth=2.5 if encoding == 'baseline' else 2,
                    linestyle='-',
                    label=LABELS[encoding],
                    alpha=0.9
                )
        
        # Styling
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
        ax.set_ylabel('Validation Accuracy (%)', fontweight='bold', fontsize=13)
        ax.set_title(f'Dataset {dataset.upper()} - Validation Accuracy Evolution', 
                     fontweight='bold', fontsize=15)
        ax.legend(loc='best', framealpha=0.95, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Add horizontal line at 50% for reference
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        
        # Save individual figures
        output_path_png = FIGURES_DIR / f"accuracy_timeline_{dataset}.png"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path_png}")
        
        output_path_pdf = FIGURES_DIR / f"accuracy_timeline_{dataset}.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"✓ Saved: {output_path_pdf}")
        
        plt.close()

# ============================================================================
# 6. LOSS TIMELINE (separated by dataset)
# ============================================================================

def plot_loss_timeline():
    """Create separate loss timeline plots for MNIST and CIFAR-10."""
    
    datasets = ['mnist', 'cifar10']
    encodings = ['baseline', 'angle', 'feature_map', 'basis', 'hybrid_angle_zz']
    
    for dataset in datasets:
        # Create individual figure for each dataset
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for encoding in encodings:
            df = load_experiment_data(dataset, encoding)
            
            if df is not None and 'val_loss' in df.columns:
                # Plot validation loss
                ax.plot(
                    df['epoch'], 
                    df['val_loss'],
                    color=COLORS[encoding],
                    linewidth=2.5 if encoding == 'baseline' else 2,
                    linestyle='-',
                    label=LABELS[encoding],
                    alpha=0.9
                )
        
        # Styling
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=13)
        ax.set_ylabel('Validation Loss', fontweight='bold', fontsize=13)
        ax.set_title(f'Dataset {dataset.upper()} - Validation Loss Evolution', 
                     fontweight='bold', fontsize=15)
        ax.legend(loc='best', framealpha=0.95, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual figures
        output_path_png = FIGURES_DIR / f"loss_timeline_{dataset}.png"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path_png}")
        
        output_path_pdf = FIGURES_DIR / f"loss_timeline_{dataset}.pdf"
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"✓ Saved: {output_path_pdf}")
        
        plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Loading all experiment data...")
    df_summary = load_all_experiments()
    
    print(f"\n{len(df_summary)} experiments loaded")
    print("\nGenerating visualizations...\n")
    
    # Generate all plots
    plot_comparative_table(df_summary)
    plot_gradient_norms()
    plot_complexity_vs_performance(df_summary)
    plot_dataset_gap(df_summary)
    plot_accuracy_timeline()
    plot_loss_timeline()
    
    print("\n✅ All visualizations generated successfully!")
    print(f"📁 Saved to: {FIGURES_DIR.absolute()}")
