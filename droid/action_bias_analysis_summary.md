# DROID Action Distribution Analysis

## Summary

Analysis of 1.24M action samples from 5000 DROID trajectories to investigate whether learned action biases explain off-axis movements in zero-shot planning.

## Key Findings

### 1. Correlation Matrix (Position Axes)

|      | dx     | dy     | dz     |
|------|--------|--------|--------|
| dx   | +1.000 | +0.012 | **-0.265** |
| dy   | +0.012 | +1.000 | -0.002 |
| dz   | -0.265 | -0.002 | +1.000 |

**Notable**: dx-dz correlation of **-0.265** indicates forward x movements are coupled with downward z movements.

### 2. Conditional Analysis

When dx > 0.1 (forward x movement, n=423,470):
- Mean dz = **-0.062** (downward bias)
- 53.5% have dz < -0.05 (downward)
- 23.3% have dz > 0.05 (upward)
- Mean dy = -0.005 (negligible)

This reflects typical manipulation patterns: reaching forward and down to grasp objects.

### 3. Off-Axis Movement Ratios

When one axis dominates (|primary| > 0.1 and > other axes):

| Primary Axis | Off-axis / Primary Ratio |
|--------------|-------------------------|
| X-axis       | 0.56                    |
| Y-axis       | 0.68                    |
| Z-axis       | 0.59                    |

Off-axis movement is common across all axes (56-68% of primary magnitude), not specific to x-axis.

### 4. Comparison with Planned Actions

In zero-shot x-axis reaching experiments, CEM selects:
- Step 0: [+0.075, -0.000] (pure x)
- Step 1: [+0.075, -0.035] (x with -y)
- Step 2+: [+0.06, **-0.075**] (large -y component)

The planner introduces large **y-axis** components, not z-axis components as the DROID dx-dz correlation would suggest.

## Interpretation

The dx-dz correlation in DROID (-0.265) does not directly explain the off-axis y-movements seen in x-axis planning. Two possibilities:

1. **Predictor internalized task-specific correlations**: The world model may have learned that certain visual changes (associated with x-axis movement in the target domain) require coupled y/z movements based on the robot's configuration in DROID data.

2. **Energy landscape geometry**: The latent-space objective may have local minima at coupled action directions, independent of the raw action correlations in training data.

The key insight is that **off-axis movement is ubiquitous in DROID** (56-68% of primary axis magnitude), and the predictor may have learned to expect such coupled movements, even when the zero-shot task requires pure single-axis motion.

## Files Generated

- `action_statistics.npz`: Raw velocity data and correlations
- `action_distribution_analysis.png`: Comprehensive visualization
- `dx_dz_correlation_analysis.png`: Focused dx-dz analysis
