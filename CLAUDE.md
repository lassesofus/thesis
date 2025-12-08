# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis research project integrating **V-JEPA 2** (Video Joint-Embedding Predictive Architecture) with **RoboHive** (robotics simulation framework) to evaluate visual world models for robotic manipulation. The project explores whether V-JEPA 2's self-supervised video representations can enable effective goal-conditioned reaching without task-specific training.

**Key Technologies:**
- PyTorch 2.4.0 with CUDA 12.6
- MuJoCo 3.1.3 for physics simulation
- Vision Transformers (ViT-L/16, ViT-H/16, ViT-g/16)
- Action-conditioned predictors for robot control

## Repository Structure

```
/home/s185927/thesis/
├── vjepa2/                          # V-JEPA 2 core library
│   ├── app/                         # Training applications
│   │   ├── vjepa/                  # Standard V-JEPA pre-training
│   │   └── vjepa_droid/            # Action-conditioned training on DROID data
│   ├── configs/                     # YAML configuration files
│   │   ├── train/                  # Pre-training, cooldown, action-conditioned
│   │   ├── eval/                   # Frozen evaluation configs
│   │   └── inference/              # Inference configs
│   ├── src/                        # Core library code
│   │   ├── models/                 # Vision Transformers, Predictors
│   │   ├── datasets/               # Video dataset loaders
│   │   └── utils/                  # Distributed training utilities
│   ├── evals/                      # Evaluation/probing modules
│   └── notebooks/                  # Examples and utilities
│       └── utils/                  # WorldModel wrapper, MPC utilities
├── robohive/                       # RoboHive robotics framework
│   └── robohive/robohive/
│       ├── envs/                   # Environment suites (arms, hands, etc.)
│       ├── utils/                  # Core utilities
│       │   ├── robo_samples.py    # Main experiment script (1106 lines)
│       │   ├── inverse_kinematics.py
│       │   ├── min_jerk.py        # Trajectory generation
│       │   └── plot_distance_analysis.py
│       ├── tutorials/              # Example scripts
│       └── experiments/            # Experiment outputs
│           ├── reach_along_x/
│           ├── reach_along_y/
│           └── reach_along_z/
└── environment.yml                 # Conda environment (300+ dependencies)
```

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate vjepa2-312

# Install V-JEPA 2 (from local directory)
pip install -e vjepa2/

# Install RoboHive (from local directory)
pip install -e robohive/

# Initialize RoboHive (required for proper setup)
robohive_init
```

**Note:** Conda environment at `/opt/conda` is available on the system.

## Core Workflow: Three-Phase Experiment Protocol

The main integration script `robohive/robohive/robohive/utils/robo_samples.py` bridges V-JEPA 2 and RoboHive:

### Phase 1: IK Baseline
- Uses inverse kinematics to reach target position
- Generates min-jerk trajectories for smooth motion
- Records RGB video and distance metrics
- Serves as baseline for comparison

### Phase 2: Return Home
- Returns robot to start position
- Captures "start" state image for planning

### Phase 3: V-JEPA Planning
- Loads V-JEPA 2 action-conditioned model
- Uses CEM (Cross-Entropy Method) planning in latent space
- Iteratively plans actions to reach goal image
- Tracks both representation distances and physical distances

## Common Commands

### V-JEPA 2 Training

```bash
# Local training (single GPU)
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml --devices cuda:0

# Distributed training (SLURM)
python -m app.main_distributed \
  --fname configs/train/vitg16/droid-256px-8f.yaml \
  --time 6000 \
  --account my_account --qos=my_qos
```

### V-JEPA 2 Evaluation

```bash
# Local evaluation (frozen probe)
python -m evals.main --fname configs/eval/vitl/ssv2.yaml --devices cuda:0

# Distributed evaluation
python -m evals.main_distributed \
  --fname configs/eval/vitl/ssv2.yaml \
  --time 8600 \
  --account my_account --qos=my_qos
```

### RoboHive Environment Testing

```bash
# Examine a specific environment
python -m robohive.utils.examine_env -e FrankaReachRandom-v0

# Run full test suite
python -m robohive.tests.test_all
python -m robohive.tests.test_arms  # Test arm environments only
```

### Running Reach Experiments

```bash
# Single experiment (manual)
cd /home/s185927/thesis/robohive/robohive/robohive
python utils/robo_samples.py \
  --render offscreen \
  --out_dir experiments \
  --fixed_target \
  --experiment_type z \
  --planning_steps 5 \
  --enable_vjepa_planning \
  --save_distance_data \
  --max_episodes 10 \
  --action_transform swap_xy_negate_x

# Run all experiments (x, y, z axes)
cd /home/s185927/thesis
bash single_goal_reach_experiments.sh

# Generate plots from experiment data
bash single_goal_reach_experiments_plots.sh
```

### Testing and Linting

```bash
# V-JEPA 2 tests
cd vjepa2
pytest tests/

# V-JEPA 2 linting
black app evals src tests
isort app evals src tests
flake8 --config .flake8 app evals src tests
```

## Architecture Details

### V-JEPA 2 Models

**Available Models:**
- **ViT-L/16**: 300M parameters, 256px resolution
- **ViT-H/16**: 600M parameters, 256px resolution
- **ViT-g/16**: 1B parameters, 256px or 384px resolution

**Loading Models (PyTorch Hub):**
```python
import torch

# Load action-conditioned model (used for robot control)
vjepa2_encoder, vjepa2_ac_predictor = torch.hub.load(
    'facebookresearch/vjepa2',
    'vjepa2_ac_vit_giant'
)

# Load standard encoder only
vjepa2_vit_giant = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
```

**Loading via WorldModel Wrapper (for planning):**
```python
from notebooks.utils.world_model_wrapper import WorldModel

world_model = WorldModel(
    enc_checkpoint_path="path/to/vjepa2-ac-vitg.pt",
    device="cuda"
)
```

### RoboHive Key Concepts

**Primary Environment:** `FrankaReachRandom-v0`
- 7-DOF Franka robot arm
- End-effector site: "end_effector"
- Offscreen rendering for data collection

**Inverse Kinematics:**
- Function: `qpos_from_site_pose(sim, site_name, target_pos, target_quat)`
- Converts Cartesian positions to joint angles
- Used for both IK baseline and executing planned actions

**Min-Jerk Trajectories:**
- Function: `generate_joint_space_min_jerk(qpos_start, qpos_end, duration, hz)`
- Generates smooth joint-space trajectories
- Ensures natural-looking robot motion

### Integration Points

**Action Transformation:**
- V-JEPA 2 outputs actions in camera frame
- Must transform to robot frame: `--action_transform swap_xy_negate_x`
- Handles coordinate system differences between vision and control

**Distance Metrics:**
- **Euclidean Distance**: Physical distance to goal in 3D space
- **Representation Distance**: Cosine distance in V-JEPA 2 latent space
- Both tracked throughout planning to evaluate world model quality

**Planning Algorithm:**
- CEM (Cross-Entropy Method) in latent space
- Plans sequences of actions to minimize representation distance
- Actions converted to waypoints via IK, executed with min-jerk

## Configuration System

All V-JEPA 2 configurations use YAML files with the following structure:

**Key Configuration Fields:**
- `app`: Application type (`vjepa`, `vjepa_droid`)
- `nodes`, `tasks_per_node`: Distributed training setup
- `data`: Dataset paths, batch size, augmentation settings
- `model`: Architecture (e.g., `vit_giant_xformers`)
- `optimization`: Learning rate, epochs, warmup schedule
- `meta`: Checkpoints, seeds, evaluation frequency

**Example:** `configs/train/vitg16/droid-256px-8f.yaml`
- 4 nodes × 8 GPUs = 32 total GPUs
- Batch size: 8 per GPU
- Training on DROID dataset (robot trajectories)
- Action-conditioned predictor training
- Initializes from pre-trained ViT-g checkpoint

## Experiment Types

The project focuses on single-goal reaching tasks along different axes:

- **reach_along_x**: Movement primarily along x-axis
- **reach_along_y**: Movement primarily along y-axis
- **reach_along_z**: Movement primarily along z-axis

Each experiment compares:
1. IK baseline performance (Phase 1)
2. V-JEPA planning performance (Phase 3)

Output includes videos, distance metrics (JSON/NPZ), and planning visualizations.

## Data Flow

```
DROID Dataset (Robot Videos + Actions)
    ↓
V-JEPA 2 Action-Conditioned Training
    ↓
Trained WorldModel Checkpoint (.pt)
    ↓
WorldModel Wrapper (planning interface)
    ↓
RoboHive Simulation (Franka environment)
    ↓
robo_samples.py (3-phase experiment)
    ↓
Experimental Results (videos, metrics, plots)
```

## Important Notes

- **Device Selection**: V-JEPA 2 models are large (1B parameters). Use GPU with sufficient memory.
- **Offscreen Rendering**: Required for data collection. Use `--render offscreen`.
- **Fixed vs Random Targets**: Use `--fixed_target` for reproducibility or omit for random sampling.
- **Action Transform**: Always specify `--action_transform swap_xy_negate_x` for proper coordinate alignment.
- **SLURM Usage**: Distributed training configs assume SLURM cluster with specified account/QOS.
- **Checkpoint Paths**: Update paths in configs to match local storage (e.g., `/data/s185927/vjepa2/weights/`).
- **V-JEPA Root**: The script `robo_samples.py` assumes V-JEPA 2 is at `/home/s185927/thesis/vjepa2`.

## CI/CD

GitHub Actions workflows (in `.github/workflows/`):
- `base_tests.yaml`: Runs pytest on push
- `linters.yaml`: Runs black, isort, flake8 on PRs

These only apply to the V-JEPA 2 component.
