"""
Camera configuration utilities for MuJoCo experiments.

Understanding MuJoCo Camera Orientation:
=========================================

There are 3 ways to specify camera orientation in MuJoCo:

1. `xyaxes` (6 values): x1 x2 x3 y1 y2 y3
   - First 3: camera's local X-axis direction (points RIGHT in image)
   - Last 3: camera's local Y-axis direction (points UP in image)
   - Z-axis (viewing direction, INTO scene) = X cross Y

2. `quat` (4 values): w x y z (quaternion)
   - Standard quaternion rotation from world frame

3. `euler` (3 values): depends on compiler euler sequence (default 'xyz')
   - Euler angles in radians

The `xyaxes` format is most intuitive for camera setup because you directly
specify what directions appear as "right" and "up" in the rendered image.

Coordinate Frame Reference (RoboHive/MuJoCo):
=============================================
- X: Positive = forward (toward workspace from robot base)
- Y: Positive = left (when facing the workspace)
- Z: Positive = up

For a camera looking AT the robot workspace:
- Camera X-axis: what direction appears as "right" in the image
- Camera Y-axis: what direction appears as "up" in the image
- Camera Z-axis: opposite of viewing direction (camera looks along -Z)
"""

import numpy as np
from scipy.spatial.transform import Rotation


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def xyaxes_from_look_at(camera_pos, target_pos, up_hint=np.array([0, 0, 1])):
    """
    Generate xyaxes for a camera looking at a target point.

    Args:
        camera_pos: [x, y, z] camera position in world coordinates
        target_pos: [x, y, z] point the camera looks at
        up_hint: [x, y, z] approximate "up" direction (default: world Z-up)

    Returns:
        xyaxes: 6-element array [x1, x2, x3, y1, y2, y3]
    """
    camera_pos = np.array(camera_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)
    up_hint = np.array(up_hint, dtype=float)

    # Camera looks along -Z, so forward = target - camera
    forward = normalize(target_pos - camera_pos)

    # Camera Z-axis points backward (opposite of viewing direction)
    cam_z = -forward

    # Camera X-axis: perpendicular to up_hint and cam_z
    cam_x = normalize(np.cross(up_hint, cam_z))

    # Camera Y-axis: perpendicular to cam_z and cam_x
    cam_y = normalize(np.cross(cam_z, cam_x))

    return np.concatenate([cam_x, cam_y])


def xyaxes_from_spherical(camera_pos, target_pos, roll_deg=0):
    """
    Generate xyaxes with optional roll (rotation around viewing axis).

    Args:
        camera_pos: [x, y, z] camera position
        target_pos: [x, y, z] look-at point
        roll_deg: rotation in degrees around the viewing axis (0 = level horizon)

    Returns:
        xyaxes: 6-element array
    """
    xyaxes = xyaxes_from_look_at(camera_pos, target_pos)

    if abs(roll_deg) > 1e-6:
        cam_x = xyaxes[:3]
        cam_y = xyaxes[3:]
        cam_z = np.cross(cam_x, cam_y)  # viewing direction (into scene)

        # Rotate X and Y around Z by roll angle
        roll_rad = np.radians(roll_deg)
        c, s = np.cos(roll_rad), np.sin(roll_rad)

        new_cam_x = c * cam_x + s * cam_y
        new_cam_y = -s * cam_x + c * cam_y

        xyaxes = np.concatenate([new_cam_x, new_cam_y])

    return xyaxes


def spherical_camera_position(target_pos, distance, azimuth_deg, elevation_deg):
    """
    Calculate camera position using spherical coordinates relative to target.

    Args:
        target_pos: [x, y, z] point the camera looks at
        distance: distance from target to camera
        azimuth_deg: angle in XY plane, measured from +X toward +Y (0=front, 90=left, 180=back, 270=right)
        elevation_deg: angle above XY plane (0=level, 90=directly above)

    Returns:
        camera_pos: [x, y, z] camera position
    """
    target_pos = np.array(target_pos, dtype=float)

    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)

    # Spherical to Cartesian offset
    # At azimuth=0, elevation=0: camera is at +X from target (looking in -X direction)
    dx = distance * np.cos(el_rad) * np.cos(az_rad)
    dy = distance * np.cos(el_rad) * np.sin(az_rad)
    dz = distance * np.sin(el_rad)

    return target_pos + np.array([dx, dy, dz])


def generate_camera_xml(name, camera_pos, target_pos, roll_deg=0):
    """
    Generate MuJoCo XML camera element string.

    Args:
        name: camera name
        camera_pos: [x, y, z] camera position
        target_pos: [x, y, z] look-at point
        roll_deg: roll angle in degrees

    Returns:
        XML string for the camera element
    """
    xyaxes = xyaxes_from_spherical(camera_pos, target_pos, roll_deg)

    pos_str = f"{camera_pos[0]:.3f} {camera_pos[1]:.3f} {camera_pos[2]:.3f}"
    xyaxes_str = f"{xyaxes[0]:.3f} {xyaxes[1]:.3f} {xyaxes[2]:.3f} {xyaxes[3]:.3f} {xyaxes[4]:.3f} {xyaxes[5]:.3f}"

    return f"<camera name='{name}' pos='{pos_str}' xyaxes='{xyaxes_str}'/>"


# ============================================================================
# Pre-defined camera configurations for experiments
# ============================================================================

# Target position (approximate workspace center for Franka reach task)
WORKSPACE_CENTER = np.array([0.0, 0.475, 1.0])

# Distance from workspace center
CAMERA_DISTANCE = 0.9

# Pre-defined camera angles (azimuth, elevation)
CAMERA_CONFIGS = {
    # Original left camera (approximately)
    'left_cam': {
        'azimuth': 110,      # From front-left
        'elevation': 15,     # Slightly above
        'distance': 0.9,
        'roll': 0,
    },
    # Front view - looking from +X direction
    'front_cam': {
        'azimuth': 0,        # Directly in front
        'elevation': 15,     # Slightly above
        'distance': 0.9,
        'roll': 0,
    },
    # Right side view
    'right_cam': {
        'azimuth': 270,      # From the right (-Y direction)
        'elevation': 15,
        'distance': 0.9,
        'roll': 0,
    },
    # Back view
    'back_cam': {
        'azimuth': 180,      # From behind
        'elevation': 15,
        'distance': 0.9,
        'roll': 0,
    },
    # Top-down view
    'top_cam': {
        'azimuth': 90,       # Doesn't matter much for top-down
        'elevation': 80,     # Almost directly above
        'distance': 0.8,
        'roll': 0,
    },
    # 45-degree view from front-right
    'front_right_cam': {
        'azimuth': 315,      # Front-right corner
        'elevation': 25,
        'distance': 0.9,
        'roll': 0,
    },
    # 45-degree view from back-left
    'back_left_cam': {
        'azimuth': 135,
        'elevation': 25,
        'distance': 0.9,
        'roll': 0,
    },
    # Higher elevation left view
    'left_high_cam': {
        'azimuth': 110,
        'elevation': 45,     # Much higher angle
        'distance': 0.9,
        'roll': 0,
    },
    # Lower elevation left view
    'left_low_cam': {
        'azimuth': 110,
        'elevation': 0,      # Level with workspace
        'distance': 0.9,
        'roll': 0,
    },
}


def generate_all_cameras_xml(target_pos=WORKSPACE_CENTER):
    """Generate XML for all pre-defined cameras."""
    lines = ["<!-- Auto-generated camera configurations -->"]
    for name, config in CAMERA_CONFIGS.items():
        pos = spherical_camera_position(
            target_pos,
            config['distance'],
            config['azimuth'],
            config['elevation']
        )
        xml = generate_camera_xml(name, pos, target_pos, config.get('roll', 0))
        lines.append(f"        {xml}")
    return "\n".join(lines)


def print_camera_info():
    """Print all camera configurations with their positions and xyaxes."""
    print("=" * 80)
    print("Camera Configurations for Franka Reach Experiments")
    print("=" * 80)
    print(f"\nTarget (workspace center): {WORKSPACE_CENTER}")
    print("\nSpherical coordinate system:")
    print("  - azimuth: angle in XY plane (0=+X/front, 90=+Y/left, 180=-X/back, 270=-Y/right)")
    print("  - elevation: angle above XY plane (0=level, 90=top)")
    print()

    for name, config in CAMERA_CONFIGS.items():
        pos = spherical_camera_position(
            WORKSPACE_CENTER,
            config['distance'],
            config['azimuth'],
            config['elevation']
        )
        xyaxes = xyaxes_from_spherical(pos, WORKSPACE_CENTER, config.get('roll', 0))

        print(f"{name}:")
        print(f"  Azimuth: {config['azimuth']}deg, Elevation: {config['elevation']}deg, Distance: {config['distance']}m")
        print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"  xyaxes: [{xyaxes[0]:.3f}, {xyaxes[1]:.3f}, {xyaxes[2]:.3f}, {xyaxes[3]:.3f}, {xyaxes[4]:.3f}, {xyaxes[5]:.3f}]")
        print()

    print("\n" + "=" * 80)
    print("XML to add to franka_reach_v0.xml:")
    print("=" * 80)
    print(generate_all_cameras_xml())


if __name__ == "__main__":
    print_camera_info()
