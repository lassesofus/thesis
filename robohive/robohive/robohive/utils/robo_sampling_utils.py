
import numpy as np
import math

# Config
ARM_nJnt = 7
EE_SITE = "end_effector"  # from the Franka chain include

# Default initial joint configuration for Franka
ARM_JNT0 = np.array([
        -0.0321842,  # Joint 1
        -0.394346,   # Joint 2
        0.00932319,  # Joint 3
        -2.77917,    # Joint 4
        -0.011826,   # Joint 5
        0.713889,    # Joint 6
        1.53183      # Joint 7
    ])

def sample_point_in_sphere(center, R):
    """Uniform-in-volume sample inside a sphere centered at center with radius R."""
    u = np.random.rand()
    r = R * (u ** (1 / 3))
    phi = 2 * np.pi * np.random.rand()
    cost = np.random.uniform(-1, 1)
    sint = math.sqrt(1 - cost * cost)
    dir_vec = np.array([
        sint * math.cos(phi),
        sint * math.sin(phi),
        cost
    ])
    return center + r * dir_vec

def compute_new_pose(current_pos, delta_pos, delta_rpy):
    """
    Compute new end-effector pose given current position and deltas.

    Args:
        current_pos: Current position [x, y, z]
        delta_pos: Position delta [dx, dy, dz]
        delta_rpy: Orientation delta [droll, dpitch, dyaw]

    Returns:
        new_pos: New position [x, y, z]
        new_rpy: New orientation [roll, pitch, yaw]
    """
    new_pos = current_pos + delta_pos
    # For simplicity, just return the delta as the new orientation
    # In practice, you might want to integrate this properly with current orientation
    new_rpy = delta_rpy
    return new_pos, new_rpy

def transform_action(action, transform_type='none'):
        """
        Transform actions from camera frame to robot frame.

        Args:
            action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            transform_type: Type of transformation to apply
        Returns:
            Transformed action in robot frame
        """
        transformed = action.copy()

        if transform_type == 'swap_xy':
            # Swap x and y axes
            transformed[0], transformed[1] = action[1], action[0]
            transformed[3], transformed[4] = action[4], action[3]  # Also swap roll/pitch
        elif transform_type == 'negate_x':
            # Negate x axis
            transformed[0] = -action[0]
        elif transform_type == 'negate_y':
            # Negate y axis
            transformed[1] = -action[1]
        elif transform_type == 'swap_xy_negate_x':
            # Swap xy and negate new x
            transformed[0], transformed[1] = -action[1], action[0]
            transformed[3], transformed[4] = action[4], action[3]
        elif transform_type == 'swap_xy_negate_y':
            # Swap xy and negate new y
            transformed[0], transformed[1] = action[1], -action[0]
            transformed[3], transformed[4] = action[4], action[3]
        elif transform_type == 'rotate_90_cw':
            # Rotate 90 degrees clockwise in xy plane
            transformed[0], transformed[1] = action[1], -action[0]
            transformed[3], transformed[4] = action[4], action[3]
        elif transform_type == 'rotate_90_ccw':
            # Rotate 90 degrees counter-clockwise in xy plane
            transformed[0], transformed[1] = -action[1], action[0]
            transformed[3], transformed[4] = action[4], action[3]

        return transformed