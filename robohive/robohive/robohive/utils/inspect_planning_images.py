"""
Inspect planning images to verify they're meaningful.
Usage: python inspect_planning_images.py --planning_dir ./planning_images_ep0
"""

import click
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

@click.command()
@click.option('--planning_dir', type=str, required=True, help='Directory containing planning images')
@click.option('--show_differences', is_flag=True, help='Show difference between consecutive frames')
def main(planning_dir, show_differences):
    # Find all current step images
    image_files = sorted([f for f in os.listdir(planning_dir) if f.endswith('_current.png')])
    
    if not image_files:
        print(f"No planning images found in {planning_dir}")
        return
    
    print(f"Found {len(image_files)} planning step images")
    
    # Load goal image
    goal_path = os.path.join(planning_dir, 'goal.png')
    if os.path.exists(goal_path):
        goal_img = np.array(Image.open(goal_path))
        print(f"Goal image shape: {goal_img.shape}")
    else:
        print("Warning: Goal image not found")
        goal_img = None
    
    # Create figure with subplots
    n_images = len(image_files)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.array(axes).flatten() if n_images > 1 else [axes]
    
    images = []
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(planning_dir, img_file)
        img = np.array(Image.open(img_path))
        images.append(img)
        
        step_num = int(img_file.split('step')[1].split('_')[0])
        axes[idx].imshow(img)
        axes[idx].set_title(f'Step {step_num}')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(planning_dir, 'planning_sequence.png'), dpi=150)
    print(f"Saved planning sequence to: {os.path.join(planning_dir, 'planning_sequence.png')}")
    
    # Show differences between consecutive frames
    if show_differences and len(images) > 1:
        fig2, axes2 = plt.subplots(1, len(images)-1, figsize=(4*(len(images)-1), 4))
        axes2 = [axes2] if len(images) == 2 else axes2
        
        for idx in range(len(images) - 1):
            diff = np.abs(images[idx+1].astype(float) - images[idx].astype(float))
            diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            
            axes2[idx].imshow(diff_norm)
            axes2[idx].set_title(f'Diff {idx}→{idx+1}')
            axes2[idx].axis('off')
            
            # Print statistics
            print(f"Diff step {idx}→{idx+1}: mean={diff.mean():.2f}, max={diff.max():.2f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(planning_dir, 'frame_differences.png'), dpi=150)
        print(f"Saved frame differences to: {os.path.join(planning_dir, 'frame_differences.png')}")
    
    plt.show()

if __name__ == '__main__':
    main()
