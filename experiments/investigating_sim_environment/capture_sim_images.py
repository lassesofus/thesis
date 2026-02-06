"""
Capture images of the Franka reach simulation with DROID coordinate frame visualization.

Usage:
    python capture_sim_images.py [--width 640] [--height 480] [--output_dir .]
    python capture_sim_images.py --gripper robotiq  # Use RobotiQ gripper
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Add robohive to path
robohive_path = "/home/s185927/thesis/robohive/robohive/robohive"
if robohive_path not in sys.path:
    sys.path.insert(0, robohive_path)

from robohive.physics.sim_scene import SimScene
from robohive.utils.xml_utils import reassign_parent

# Robot arm configuration
ARM_nJnt = 7

# Joint configurations for different grippers
ARM_JNT0_FRANKA = np.array([
    -0.0321842,  # Joint 1
    -0.394346,   # Joint 2
    0.00932319,  # Joint 3
    -2.77917,    # Joint 4
    -0.011826,   # Joint 5
    0.713889,    # Joint 6
    1.53183      # Joint 7
])

# Joint 7 rotated -45 degrees to match real robot gripper orientation
ARM_JNT0_ROBOTIQ = np.array([
    -0.0321842,  # Joint 1
    -0.394346,   # Joint 2
    0.00932319,  # Joint 3
    -2.77917,    # Joint 4
    -0.011826,   # Joint 5
    0.713889,    # Joint 6
    0.74663      # Joint 7 (rotated for RobotiQ)
])


def capture_images(sim_path, cameras, width, height, output_dir, device_id=0, gripper='franka'):
    """
    Load simulation and capture images from specified cameras.

    Args:
        sim_path: Path to MuJoCo XML file
        cameras: List of camera names to capture from
        width: Image width
        height: Image height
        output_dir: Directory to save images
        device_id: GPU device ID for rendering
        gripper: Gripper type ('franka' or 'robotiq')

    Returns:
        List of (camera_name, image_path) tuples
    """
    print(f"Loading simulation from: {sim_path}")
    sim = SimScene.get_sim(model_handle=sim_path)

    # For RobotiQ model, reparent ee_mount to panda0_link7
    if gripper == 'robotiq':
        raw_xml = sim.model.get_xml()
        processed_xml = reassign_parent(xml_str=raw_xml, receiver_node="panda0_link7", donor_node="ee_mount")
        processed_path = os.path.join(os.path.dirname(os.path.abspath(sim_path)), '_robotiq_processed.xml')
        with open(processed_path, 'w') as f:
            f.write(processed_xml)
        sim = SimScene.get_sim(model_handle=processed_path)
        os.remove(processed_path)
        print("RobotiQ gripper attached to Franka arm (panda0_link7)")

    # Select joint configuration based on gripper type
    ARM_JNT0 = ARM_JNT0_ROBOTIQ if gripper == 'robotiq' else ARM_JNT0_FRANKA

    # Set robot to start position (same as experiments)
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt, dtype=float)
    sim.data.ctrl[:ARM_nJnt] = sim.data.qpos[:ARM_nJnt].copy()
    sim.forward()
    print(f"Robot arm set to experiment start position (gripper: {gripper})")

    os.makedirs(output_dir, exist_ok=True)

    captured_images = []
    for camera_name in cameras:
        print(f"Capturing from camera: {camera_name}")

        try:
            rgb = sim.renderer.render_offscreen(
                width=width,
                height=height,
                camera_id=camera_name,
                device_id=device_id
            )

            if rgb is not None:
                img = Image.fromarray(rgb)

                output_path = os.path.join(output_dir, f"sim_{camera_name}.png")
                img.save(output_path)
                print(f"  Saved: {output_path}")
                captured_images.append((camera_name, output_path))
            else:
                print(f"  Warning: No image returned for {camera_name}")

        except Exception as e:
            print(f"  Error capturing {camera_name}: {e}")

    return captured_images


def create_combined_figure(image_paths, output_path, camera_names=None):
    """
    Combine multiple images into a single figure (2x2 grid).

    Args:
        image_paths: List of image file paths
        output_path: Output path for combined figure
        camera_names: Optional list of camera names for labels
    """
    if not image_paths:
        print("No images to combine")
        return

    images = [Image.open(p) for p in image_paths]

    # Calculate combined image size (2x2 grid layout)
    img_width = images[0].width
    img_height = images[0].height

    total_width = img_width * 2
    total_height = img_height * 2

    # Create combined image
    combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Paste images in 2x2 grid
    positions = [(0, 0), (img_width, 0), (0, img_height), (img_width, img_height)]
    for i, img in enumerate(images):
        if i < len(positions):
            combined.paste(img, positions[i])

    combined.save(output_path)
    print(f"Saved combined figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Capture simulation images with DROID frame arrows")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--output_dir", type=str,
                       default="/home/s185927/thesis/experiments/investigating_sim_environment",
                       help="Output directory for images")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--cameras", type=str, nargs="+",
                       default=["left_cam", "front_cam", "left_high_cam", "back_cam"],
                       help="Camera names to capture from")
    parser.add_argument("--gripper", type=str, choices=['franka', 'robotiq'],
                       default='franka',
                       help="Gripper type: franka (default) or robotiq")
    args = parser.parse_args()

    # Path to the XML with DROID axes - select based on gripper type
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.gripper == 'robotiq':
        sim_path = os.path.join(script_dir, "franka_reach_robotiq_droid_axes.xml")
    else:
        sim_path = os.path.join(script_dir, "franka_reach_droid_axes.xml")

    if not os.path.exists(sim_path):
        print(f"Error: XML file not found: {sim_path}")
        sys.exit(1)

    captured = capture_images(
        sim_path=sim_path,
        cameras=args.cameras,
        width=args.width,
        height=args.height,
        output_dir=args.output_dir,
        device_id=args.device_id,
        gripper=args.gripper
    )

    # Create combined figure
    if captured:
        image_paths = [path for _, path in captured]
        combined_path = os.path.join(args.output_dir, "sim_combined.png")
        create_combined_figure(image_paths, combined_path)

    print("Done!")


if __name__ == "__main__":
    main()
