"""
Visualize collected body mask data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
import os


def visualize_body_masks(data_path, num_samples=20):
    """
    Visualize body mask samples from the collected data.
    
    Args:
        data_path: Path to the .npz file
        num_samples: Number of samples to visualize
    """
    print(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    body_masks = data['body_masks']
    gait = data['gait']
    
    print(f"Body masks shape: {body_masks.shape}")
    print(f"Gait array shape: {gait.shape}")
    print(f"Depth range: [{body_masks.min():.3f}, {body_masks.max():.3f}]")
    print(f"Unique gait types: {np.unique(gait)}")
    
    # Randomly select samples to visualize
    num_samples = min(num_samples, len(body_masks))
    indices = np.random.choice(len(body_masks), num_samples, replace=False)
    
    # Create figure
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, ax_idx in enumerate(indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get the mask
        mask = body_masks[ax_idx]
        gait_type = gait[ax_idx]
        
        # Display with FIXED range: [-0.5, 1.0]
        # This ensures: 
        #   -0.5 (body, nearest) -> black
        #    0.0 (body, mid) -> gray
        #    0.5 (body, farthest) -> light gray
        #    1.0 (non-body/background) -> white
        im = ax.imshow(mask, cmap='gray', vmin=-0.5, vmax=1.0)
        ax.set_title(f'Sample {ax_idx}\nGait: {gait_type}', fontsize=9)
        ax.axis('off')
        
        # Add colorbar with fixed range
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    
    # Hide empty subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = data_path.replace('.npz', '_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    # Don't show in non-interactive mode, just close
    plt.close()


def analyze_statistics(data_path):
    """
    Analyze and print statistics of the collected data.
    """
    print("\n" + "=" * 80)
    print("Data Statistics Analysis")
    print("=" * 80)
    
    data = np.load(data_path, allow_pickle=True)
    body_masks = data['body_masks']
    gait = data['gait']
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(body_masks)}")
    print(f"  Image shape: {body_masks.shape[1:]} (H x W)")
    
    print(f"\nDepth Statistics:")
    print(f"  Min depth: {body_masks.min():.4f}")
    print(f"  Max depth: {body_masks.max():.4f}")
    print(f"  Mean depth: {body_masks.mean():.4f}")
    print(f"  Std depth: {body_masks.std():.4f}")
    
    print(f"\nGait Distribution:")
    unique_gaits, counts = np.unique(gait, return_counts=True)
    for g, c in zip(unique_gaits, counts):
        print(f"  Gait {g}: {c} samples ({c/len(gait)*100:.2f}%)")
    
    # Check for non-body regions (typically filled with 1.0 or max value)
    threshold = body_masks.max() * 0.95
    non_body_ratio = (body_masks > threshold).mean()
    print(f"\nNon-body region ratio: {non_body_ratio*100:.2f}%")
    
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize body mask data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the .npz file containing body mask data')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--stats_only', action='store_true',
                        help='Only print statistics without visualization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: File not found: {args.data_path}")
        exit(1)
    
    # Analyze statistics
    analyze_statistics(args.data_path)
    
    # Visualize samples
    if not args.stats_only:
        visualize_body_masks(args.data_path, args.num_samples)
