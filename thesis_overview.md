# Thesis Overview: From Words to Actions

## Language-Guided Robotic Action Planning with V-JEPA 2

**Institution:** DTU Compute, Technical University of Denmark  
**Degree:** Master of Science in Engineering  
**Author:** s185927

---

## Executive Summary

This thesis investigates how predictive world models can be extended to interpret and act upon goals expressed in natural language. The work centers on **V-JEPA 2-AC** (Visual Joint-Embedding Predictive Architecture with Action Conditioning), a self-supervised latent world model that predicts future representations from visual observations and action sequences. The core challenge is bridging the gap between symbolic linguistic intent and embodied predictive control.

---

## Research Questions

1. **Language-to-Latent Grounding:** How can text-based goals be grounded into the latent predictive space of V-JEPA-2-AC such that language-conditioned goal representations are compatible with action-conditioned latent rollouts?

2. **Planning Effectiveness:** To what extent does text-based goal specification support effective latent-space planning for goal-directed robotic control, compared to visual goal conditioning?

3. **Failure Mode Analysis:** What failure modes arise when using language as a goal-specification medium for embodied planning, particularly due to ambiguity, underspecification, and perceptual misalignment?

---

## Planned Contributions

1. **Language-conditioned latent goal grounding framework** for V-JEPA 2-AC that maps textual instructions to target latent representations compatible with action-conditioned latent rollouts

2. **Empirical evaluation** of text-conditioned latent planning in a simulated robotic manipulation environment

3. **Systematic analysis of failure modes** arising from language-based goal specification in embodied planning

---

## Technical Background

### Core Architecture: V-JEPA 2-AC

- **Encoder (Eθ):** ViT-g/16 visual encoder that produces latent state representations from observations
- **Predictor (Pϕ):** Action-conditioned predictor that forecasts future latent states given current state and actions
- **Training:** Self-supervised on DROID dataset (real-world robotic manipulation data)
- **Planning:** Uses Cross-Entropy Method (CEM) to optimize action sequences in latent space

### Self-Supervised Learning Paradigms

| Paradigm | Objective | Key Property |
|----------|-----------|--------------|
| Invariance-based | Align augmented views (e.g., SimCLR, DINO) | Semantic representations |
| Generative/Reconstructive | Reconstruct inputs (e.g., MAE) | Preserves sensory detail |
| Predictive Latent-Space | Predict latent representations (e.g., JEPA) | Task-relevant abstraction |

V-JEPA 2 uses the **predictive latent-space** approach, which avoids pixel-level prediction and captures causal/temporal structure.

### Planning Algorithm: Cross-Entropy Method (CEM)

```
Key Parameters:
- Planning horizon (H): 5 steps (default), 10 steps (extended)
- Number of samples (J): Candidate action sequences per iteration
- Elite size (K): Top-performing sequences for distribution update
- Action bound (amax): 7.5 cm maximum action magnitude
- Momentum coefficients (αμ, ασ): Smoothing for distribution updates
```

**Objective:** Minimize ℓ₁ distance between predicted final latent state and goal latent state:
```
d = ||z_H - z_g||₁
```

---

## Experimental Setup

### Simulation Environment

- **Platform:** RoboHive (MuJoCo-based)
- **Robot:** Franka Panda 7-DOF arm
- **Camera:** Single monocular viewpoint (fixed)

### Robot State Representation

```python
s_k = (p_k, r_k, g_k) ∈ R^7

where:
- p_k = (p_x, p_y, p_z) ∈ R^3  # Cartesian end-effector position
- r_k = (φ, θ, ψ) ∈ R^3        # Euler angles (roll, pitch, yaw)
- g_k ∈ [0, 1]                  # Gripper openness (0=open, 1=closed)
```

**Note:** V-JEPA action planning operates only over positions, not rotations.

### Inverse Kinematics

- **Method:** Jacobian-based iterative IK solver (damped least-squares)
- **Precision floor:** ~4-5 cm absolute positioning error (see Appendix A)
- **Trajectory execution:** Min-jerk smoothing over 3-4.5 seconds

---

## Current Experimental Results

### Experiment 1: Zero-Shot Reaching Tasks (Section 4.1)

**Setup:**
- Single-axis reaching along x, y, z axes
- Target displacement: 20 cm
- 5-step receding-horizon planning
- N=10 episodes per axis
- Success threshold: 5 cm final error

**Results Summary:**

| Axis | Initial Error | Final Error (5 steps) | Success Rate |
|------|---------------|----------------------|--------------|
| X | ~23 cm | ~19 cm | 0% |
| Y | ~20-23 cm | ~10 cm | 0% |
| Z | ~20-23 cm | ~10 cm | 0% |

**Key Findings:**
- **X-axis performance severely degraded** compared to y/z axes
- X-axis: Latent distance decreases but Cartesian error doesn't improve proportionally
- This indicates **representation-geometry misalignment** for x-axis motion
- Extended horizon (10 steps) shows plateauing—problem is not horizon length
- Hypothesis: Camera viewpoint causes lateral (x-axis) motion to induce larger background/parallax effects

### Experiment 2: Predictor Fine-Tuning on X-Axis Trajectories (Section 4.2)

**Setup:**
- Dataset: 1000 simulated x-axis trajectories (∆p_x ~ U(0.05m, 0.3m))
- Splits: 640 train / 160 validation / 200 test
- Training fractions: 25%, 50%, 75%, 100%
- **Only predictor fine-tuned; encoder frozen**
- No data augmentation during training

**Training Configuration:**
```python
optimizer: AdamW
learning_rate: 1e-4  # ~4x lower than pretraining
weight_decay: 0.04
schedule: cosine annealing with 5 epochs linear warmup
early_stopping: patience=10, min_delta=0.001
enc_lr_scale: 0  # encoder frozen
```

**Results:**
- Validation loss decreases consistently
- **Planning performance does NOT improve** over zero-shot baseline
- All training fractions (25%-100%) perform similarly (~17-19 cm median final error)
- 0% success rate across all conditions

**Open Questions (from thesis):**
1. Is gripper state varying between training and validation?
2. Are there differences in how training vs. validation data is observed?
3. Have we overfit to training samples? (Is planning better on training data?)

---

## Current Thesis Status

### Completed Sections

| Section | Status | Notes |
|---------|--------|-------|
| 1. Introduction | ✅ Complete | Motivation, framing, state of art, research questions |
| 2.1 Self-Supervised Learning | ✅ Complete | SSL paradigms overview |
| 2.2 JEPA | ✅ Complete | Conceptual foundations |
| 2.4 Model-Predictive Control | ✅ Complete | CEM algorithm detailed |
| 3.1 Experimental Setup | 🔶 Partial | Robot state, IK described; some TODOs |
| 4.1 Zero-shot Results | ✅ Complete | Full results and discussion |
| 4.2 Fine-tuning Results | ✅ Complete | Full results and discussion |
| Appendix A (IK Error Analysis) | ✅ Complete | Establishes 4-5cm precision floor |

### Incomplete/Placeholder Sections

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | ❌ Empty | Not written |
| Acknowledgements | ❌ Template | Placeholder text |
| 2.3 V-JEPA 2 | ❌ Empty | Section header only |
| 2.5 Neuroscientific Foundations | ❌ Empty | Section header only |
| 2.6 World Models in AI | ❌ Empty | Section header only |
| 5. Discussion/Conclusion | ❌ Missing | Not yet created |

### Key TODOs Mentioned in Draft

1. Section 3.1: "MENTION THAT V-JEPA ACTION-PLANNING IS NOT OVER ROTATIONS"
2. Section 3.1.2: "Let's include a table on the DROID dataset here"
3. Section 4.2.3: Open questions about fine-tuning failure modes

---

## Key Technical Details for Implementation

### File/Data Conventions

- **Video frame rate:** 30 fps recording, 4 fps for training (temporal downsampling)
- **Action space:** 3D Cartesian displacements, clipped to [-7.5cm, 7.5cm]
- **Latent distance metric:** ℓ₁ norm
- **Cartesian distance metric:** ℓ₂ (Euclidean) norm

### Model Checkpoints

- Base model: Official V-JEPA 2-AC checkpoint from Meta
- Encoder: ViT-g/16 (frozen during fine-tuning)
- Predictor: Fine-tunable action-conditioned module

### Known Limitations

1. **IK precision floor:** ~4-5 cm absolute error regardless of displacement magnitude
2. **X-axis representation-geometry misalignment:** Latent space doesn't capture x-axis motion well from current camera viewpoint
3. **Predictor-only fine-tuning insufficient:** May need encoder adaptation or camera calibration

---

## Relevant Codebases and References

### Primary References

- **V-JEPA 2 Paper:** Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (June 2025)
- **RoboHive:** Simulation framework for robot learning
- **DROID Dataset:** Real-world robotic manipulation dataset used for V-JEPA 2-AC pretraining

### Key Equations

**Predictive Latent Loss (Eq. 2.3):**
```
L_pred = ||p(f(c)) - sg(f(y))||²₂
```
where f(·) is encoder, p(·) is predictor, c is context, y is target, sg(·) is stop-gradient.

**CEM Distribution Update (Eq. 2.6-2.7):**
```
μ ← (1 - α_μ) μ̄_K + α_μ μ
σ ← (1 - α_σ) σ̄_K + α_σ σ
```

---

## Next Steps (Inferred)

1. **Investigate fine-tuning failure:** Debug why predictor fine-tuning doesn't improve planning
2. **Language grounding:** Implement text-to-latent mapping (core contribution not yet shown)
3. **Complete theory sections:** V-JEPA 2 details, neuroscience foundations, world models
4. **Multi-axis experiments:** Extend fine-tuning to y/z axes
5. **Camera calibration:** Apply V-JEPA 2's proposed viewpoint calibration procedure
6. **Write abstract and conclusion**

---

## Contact

This document serves as context for Claude Code to assist with thesis-related coding tasks. The thesis is a work in progress at DTU Compute.
