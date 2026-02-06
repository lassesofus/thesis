#!/usr/bin/env python3
"""
Visualize Energy Landscape Flattening

Generates 2D heatmaps of the energy landscape for different training checkpoints,
showing how the landscape flattens over training.
"""
import argparse, copy, glob, os, random, sys
from pathlib import Path
import h5py, imageio, numpy as np, torch, torch.nn.functional as F, yaml

sys.path.insert(0, "/home/s185927/thesis/vjepa2")
sys.path.insert(0, "/home/s185927/thesis")
from app.vjepa_droid.utils import init_video_model
from app.vjepa_droid.transforms import make_transforms
from plot_config import PLOT_PARAMS, configure_axis
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path, config, device='cuda:0'):
    d, m, meta = config.get('data', {}), config.get('model', {}), config.get('meta', {})
    crop, patch = d.get('crop_size', 256), d.get('patch_size', 16)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
        torch.cuda.set_device(device)

    encoder, predictor = init_video_model(
        uniform_power=m.get('uniform_power', True), device=device, patch_size=patch,
        max_num_frames=512, tubelet_size=d.get('tubelet_size', 2),
        model_name=m.get('model_name', 'vit_giant_xformers'), crop_size=crop,
        pred_depth=m.get('pred_depth', 24), pred_num_heads=m.get('pred_num_heads', 16),
        pred_embed_dim=m.get('pred_embed_dim', 1024), action_embed_dim=7,
        pred_is_frame_causal=m.get('pred_is_frame_causal', True),
        use_extrinsics=m.get('use_extrinsics', False), use_sdpa=meta.get('use_sdpa', True),
        use_silu=m.get('use_silu', False), use_pred_silu=m.get('use_pred_silu', False),
        wide_silu=m.get('wide_silu', True), use_rope=m.get('use_rope', True),
        use_activation_checkpointing=False)

    ckpt = torch.load(ckpt_path, map_location=device)
    strip = lambda sd: {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}

    encoder.load_state_dict(strip(ckpt['encoder']), strict=False)
    if 'predictor' in ckpt:
        predictor.load_state_dict(strip(ckpt['predictor']), strict=False)
    encoder.eval()
    predictor.eval()
    return encoder, predictor, device


def create_transform(crop_size=256):
    return make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.), reprob=0., auto_augment=False,
        motion_shift=False, crop_size=crop_size)


def encode_frames(encoder, curr, goal, transform, device, tpf):
    clips = transform(np.stack([curr, goal])).unsqueeze(0).to(device)
    B, C, T, H, W = clips.size()
    with torch.no_grad():
        c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        h = F.layer_norm(h, (h.size(-1),))
    return h[:, :tpf].contiguous(), h[:, -tpf:].contiguous()


def load_traj(ep_dir, frame_idx=10):
    """Load only the specific frames we need (start and goal) to save memory."""
    ep = Path(ep_dir)
    with h5py.File(ep / "trajectory.h5", 'r') as f:
        pos = f['observation/robot_state/cartesian_position'][:][:, :3]
    vids = glob.glob(str(ep / "recordings" / "MP4" / "*.mp4"))
    reader = imageio.get_reader(vids[0])

    # Get video metadata
    n_video_frames = reader.count_frames()
    n_traj = len(pos)

    # Compute frame indices mapping
    if n_video_frames > n_traj:
        indices = np.linspace(0, n_video_frames - 1, n_traj, dtype=int)
    else:
        indices = np.arange(n_video_frames)
        pos = pos[np.linspace(0, n_traj - 1, n_video_frames, dtype=int)]

    # Only load the frames we need
    start_vid_idx = indices[frame_idx]
    goal_vid_idx = indices[-1]

    start_frame = reader.get_data(start_vid_idx)
    goal_frame = reader.get_data(goal_vid_idx)
    reader.close()

    return start_frame, goal_frame, pos[frame_idx], pos[-1]


def compute_grid(pred, tpf, z, state, g, res, size, dev, bs=64):
    x = np.linspace(-size, size, res)
    xx, yy = np.meshgrid(x, x)
    acts = np.array([[xx[i, j], yy[i, j], 0, 0, 0, 0, 0]
                     for i in range(res) for j in range(res)])
    at = torch.tensor(acts, device=dev, dtype=z.dtype).unsqueeze(1)
    Es = []
    with torch.no_grad():
        for i in range(0, len(acts), bs):
            ba = at[i:i + bs]
            n = ba.shape[0]
            zp = pred(z.repeat(n, 1, 1), ba, state.repeat(n, 1, 1))[:, -tpf:]
            zp = F.layer_norm(zp, (zp.size(-1),))
            Es.append(torch.mean(torch.abs(zp - g[:, :tpf].repeat(n, 1, 1)), dim=[1, 2]).cpu())
    return torch.cat(Es).numpy().reshape(res, res), x


def plot_heatmaps(grids, epochs, x, out, goal_dir=None):
    n = len(epochs)
    # Create figure with extra space on right for colorbar
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1.5, 4.5))
    if n == 1:
        axes = [axes]
    vmin, vmax = np.percentile(np.concatenate([g.flatten() for g in grids]), [2, 98])

    for idx, (ax, g, ep) in enumerate(zip(axes, grids, epochs)):
        im = ax.imshow(g, extent=[x[0], x[-1], x[0], x[-1]], origin='lower',
                       cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
        ax.contour(x, x, g, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        ax.plot(0, 0, 'w+', ms=10, mew=2)
        if goal_dir is not None:
            ax.arrow(0, 0, goal_dir[0] * 0.04, goal_dir[1] * 0.04,
                     head_width=0.008, head_length=0.005, fc='red', ec='red', lw=2)

        # Set title
        ax.set_title(f'Epoch {ep}', fontsize=14)
        ax.set_xlabel(r'$\Delta x$ (m)', fontsize=12)

        # Only show y-label on leftmost plot
        if idx == 0:
            ax.set_ylabel(r'$\Delta y$ (m)', fontsize=12)
        else:
            ax.set_ylabel('')

        # Reduce number of ticks to avoid overlap
        ax.set_xticks([-0.05, 0, 0.05])
        ax.set_yticks([-0.05, 0, 0.05])
        ax.tick_params(labelsize=11)

        ax.text(0.02, 0.98, f'$\\sigma(\\mathcal{{E}})$={np.std(g):.4f}\nRange={np.ptp(g):.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # Add more space between subplots
    fig.subplots_adjust(wspace=0.3, right=0.85)

    # Add colorbar on the right side, spanning the full height
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Energy $\mathcal{E}$ (latent distance)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--epochs', type=int, nargs='+', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--data_dir', default='/data/s185927/droid_sim/axis_aligned/x_axis')
    p.add_argument('--probe_id', type=int, default=0)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--grid_resolution', type=int, default=50)
    p.add_argument('--grid_size', type=float, default=0.075)
    p.add_argument('--frame_idx', type=int, default=10)
    a = p.parse_args()

    set_seed(42)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    tpf = (cfg['data']['crop_size'] // cfg['data']['patch_size']) ** 2

    sf, gf, sp, gp = load_traj(os.path.join(a.data_dir, f"episode_{a.probe_id:04d}"), a.frame_idx)
    gdir = (gp - sp)[:2]
    gdir = gdir / (np.linalg.norm(gdir) + 1e-8)

    grids, x = [], None
    for ckpt, ep in zip(a.checkpoints, a.epochs):
        print(f"Epoch {ep}: {ckpt}")
        enc, pred, dev = load_model(ckpt, cfg, a.device)
        tr = create_transform(cfg['data']['crop_size'])
        z, g = encode_frames(enc, sf, gf, tr, dev, tpf)
        st = torch.tensor([[list(sp) + [0, 0, 0, 0]]], device=dev, dtype=z.dtype)
        grid, x = compute_grid(pred, tpf, z, st, g, a.grid_resolution, a.grid_size, dev)
        grids.append(grid)
        print(f"  std={np.std(grid):.4f} range={np.ptp(grid):.4f}")
        del enc, pred
        torch.cuda.empty_cache()

    plot_heatmaps(grids, a.epochs, x, out / 'energy_landscape_heatmaps.png', gdir)
    for g, ep in zip(grids, a.epochs):
        plot_heatmaps([g], [ep], x, out / f'energy_landscape_epoch_{ep}.png', gdir)


if __name__ == '__main__':
    main()
