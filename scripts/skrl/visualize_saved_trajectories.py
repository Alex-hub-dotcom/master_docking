#!/usr/bin/env python3
"""
Visualize saved trajectories from training.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_trajectories(npz_path):
    """Load trajectories from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    num_traj = int(data['num_trajectories'])
    
    trajectories = []
    for i in range(num_traj):
        traj = {
            'stage': int(data[f'traj_{i}_stage']),
            'positions': data[f'traj_{i}_positions'],
            'yaws': data[f'traj_{i}_yaws'],
            'actions': data[f'traj_{i}_actions'],
            'rewards': data[f'traj_{i}_rewards'],
            'lateral_offsets': data[f'traj_{i}_lateral_offsets'],
            'yaw_errors': data[f'traj_{i}_yaw_errors'],
            'goal_pos': data[f'traj_{i}_goal_pos'],
            'success': bool(data[f'traj_{i}_success']),
            'length': int(data[f'traj_{i}_length']),
        }
        trajectories.append(traj)
    
    return trajectories

def plot_trajectories_by_stage(trajectories, stage, save_dir):
    """Plot all trajectories for a specific stage."""
    stage_trajs = [t for t in trajectories if t['stage'] == stage]
    
    if not stage_trajs:
        print(f"No trajectories for stage {stage}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: All spatial trajectories
    ax = axes[0, 0]
    for i, traj in enumerate(stage_trajs):
        color = 'green' if traj['success'] else 'red'
        alpha = 0.6 if traj['success'] else 0.3
        ax.plot(traj['positions'][:, 0], traj['positions'][:, 1], 
               color=color, alpha=alpha, linewidth=1.5)
        
        # Mark start
        ax.scatter(traj['positions'][0, 0], traj['positions'][0, 1], 
                  c='blue', s=30, marker='o', zorder=5)
        
        # Mark goal
        if i == 0:  # Only once
            ax.scatter(traj['goal_pos'][0], traj['goal_pos'][1], 
                      c='gold', s=300, marker='*', label='Goal', zorder=10)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Stage {stage}: Spatial Trajectories\n'
                f'Green=Success ({sum(t["success"] for t in stage_trajs)}/{len(stage_trajs)}), '
                f'Red=Failure', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Action distributions
    ax = axes[0, 1]
    all_v = np.concatenate([t['actions'][:, 0] for t in stage_trajs])
    all_w = np.concatenate([t['actions'][:, 1] for t in stage_trajs])
    
    ax.hist2d(all_v, all_w, bins=50, cmap='viridis')
    ax.set_xlabel('v_cmd (forward/back)', fontsize=12)
    ax.set_ylabel('w_cmd (turn)', fontsize=12)
    ax.set_title(f'Stage {stage}: Action Distribution', fontsize=13)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, color='white')
    
    # Plot 3: Yaw error over time
    ax = axes[1, 0]
    for traj in stage_trajs:
        color = 'green' if traj['success'] else 'red'
        alpha = 0.6 if traj['success'] else 0.2
        ax.plot(np.rad2deg(np.abs(traj['yaw_errors'])), 
               color=color, alpha=alpha, linewidth=1)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Absolute Yaw Error (degrees)', fontsize=12)
    ax.set_title(f'Stage {stage}: Alignment Progress', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Lateral offset over time
    ax = axes[1, 1]
    for traj in stage_trajs:
        color = 'green' if traj['success'] else 'red'
        alpha = 0.6 if traj['success'] else 0.2
        ax.plot(traj['lateral_offsets'], color=color, alpha=alpha, linewidth=1)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Lateral Offset (m)', fontsize=12)
    ax.set_title(f'Stage {stage}: Lateral Correction', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'trajectories_stage{stage:02d}.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def create_summary_stats(trajectories, save_dir):
    """Create summary statistics across all stages."""
    stages = sorted(set(t['stage'] for t in trajectories))
    
    stats = {}
    for stage in stages:
        stage_trajs = [t for t in trajectories if t['stage'] == stage]
        
        stats[stage] = {
            'num_trajectories': len(stage_trajs),
            'success_rate': sum(t['success'] for t in stage_trajs) / len(stage_trajs),
            'mean_length': np.mean([t['length'] for t in stage_trajs]),
            'mean_final_yaw_error': np.mean([np.abs(t['yaw_errors'][-1]) for t in stage_trajs]),
            'mean_lateral_offset': np.mean([np.mean(t['lateral_offsets']) for t in stage_trajs]),
            'v_cmd_mean': np.mean([t['actions'][:, 0].mean() for t in stage_trajs]),
            'w_cmd_mean': np.mean([np.abs(t['actions'][:, 1]).mean() for t in stage_trajs]),
        }
    
    # Plot summary
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    x = list(stats.keys())
    
    # Success rate
    axes[0, 0].bar(x, [stats[s]['success_rate'] * 100 for s in x])
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_xlabel('Stage')
    axes[0, 0].set_title('Success Rate by Stage')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean episode length
    axes[0, 1].bar(x, [stats[s]['mean_length'] for s in x])
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_xlabel('Stage')
    axes[0, 1].set_title('Mean Episode Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final yaw error
    axes[0, 2].bar(x, [np.rad2deg(stats[s]['mean_final_yaw_error']) for s in x])
    axes[0, 2].set_ylabel('Degrees')
    axes[0, 2].set_xlabel('Stage')
    axes[0, 2].set_title('Mean Final Yaw Error')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Lateral offset
    axes[1, 0].bar(x, [stats[s]['mean_lateral_offset'] * 100 for s in x])
    axes[1, 0].set_ylabel('cm')
    axes[1, 0].set_xlabel('Stage')
    axes[1, 0].set_title('Mean Lateral Offset')
    axes[1, 0].grid(True, alpha=0.3)
    
    # v_cmd usage
    axes[1, 1].bar(x, [stats[s]['v_cmd_mean'] for s in x])
    axes[1, 1].set_ylabel('Mean v_cmd')
    axes[1, 1].set_xlabel('Stage')
    axes[1, 1].set_title('Forward/Backward Command')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # w_cmd usage (absolute)
    axes[1, 2].bar(x, [stats[s]['w_cmd_mean'] for s in x])
    axes[1, 2].set_ylabel('Mean |w_cmd|')
    axes[1, 2].set_xlabel('Stage')
    axes[1, 2].set_title('Turning Command (Absolute)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'summary_stats.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traj_file', type=str, help='Path to trajectories .npz file')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (default: same as input)')
    args = parser.parse_args()
    
    traj_path = Path(args.traj_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = traj_path.parent / 'visualizations'
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading trajectories from {traj_path}...")
    trajectories = load_trajectories(traj_path)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Plot by stage
    stages = sorted(set(t['stage'] for t in trajectories))
    for stage in stages:
        print(f"Plotting stage {stage}...")
        plot_trajectories_by_stage(trajectories, stage, output_dir)
    
    # Summary stats
    print("Creating summary statistics...")
    create_summary_stats(trajectories, output_dir)
    
    print(f"\n✅ Done! Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()