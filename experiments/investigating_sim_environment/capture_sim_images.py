"""
Capture images of the Franka reach simulation with DROID coordinate frame visualization.

Usage:
    python capture_sim_images.py [--width 640] [--height 480] [--output_dir .]
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add robohive to path
robohive_path = "/home/s185927/thesis/robohive/robohive/robohive"
if robohive_path not in sys.path:
    sys.path.insert(0, robohive_path)

from robohive.physics.sim_scene import SimScene

# Robot arm configuration
ARM_nJnt = 7
ARM_JNT0 = np.array([
    -0.0321842,  # Joint 1
    -0.394346,   # Joint 2
    0.00932319,  # Joint 3
    -2.77917,    # Joint 4
    -0.011826,   # Joint 5
    0.713889,    # Joint 6
    1.53183      # Joint 7
])


def capture_images(sim_path, cameras, width, height, output_dir, device_id=0):
    """
    Load simulation and capture images from specified cameras.

    Args:
        sim_path: Path to MuJoCo XML file
        cameras: List of camera names to capture from
        width: Image width
        height: Image height
        output_dir: Directory to save images
        device_id: GPU device ID for rendering

    Returns:
        List of (camera_name, image_path) tuples
    """
    print(f"Loading simulation from: {sim_path}")
    sim = SimScene.get_sim(model_handle=sim_path)

    # Set robot to start position (same as experiments)
    sim.data.qpos[:ARM_nJnt] = ARM_JNT0
    sim.data.qvel[:ARM_nJnt] = np.zeros(ARM_nJnt, dtype=float)
    sim.data.ctrl[:ARM_nJnt] = sim.data.qpos[:ARM_nJnt].copy()
    sim.forward()
    print("Robot arm set to experiment start position")

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


def add_axis_labels(img, camera_name):
    """Add axis legend to the image."""
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

    # Add legend in top-left corner
    legend_x, legend_y = 10, 10
    line_height = 20

    # Semi-transparent background for legend
    legend_width = 150
    legend_height = 85
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [legend_x - 5, legend_y - 5, legend_x + legend_width, legend_y + legend_height],
        fill=(0, 0, 0, 180)
    )
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((legend_x, legend_y), "DROID Frame:", fill=(255, 255, 255), font=font)
    legend_y += line_height + 5

    # Axis labels
    draw.text((legend_x, legend_y), "X-axis (Red)", fill=(255, 100, 100), font=font_small)
    legend_y += line_height
    draw.text((legend_x, legend_y), "Y-axis (Green)", fill=(100, 255, 100), font=font_small)
    legend_y += line_height
    draw.text((legend_x, legend_y), "Z-axis (Blue)", fill=(100, 100, 255), font=font_small)

    # Camera name in bottom-left
    draw.text((10, img.height - 25), f"Camera: {camera_name}", fill=(255, 255, 255), font=font_small)

    return img


def create_combined_figure(image_paths, output_path, camera_names=None):
    """
    Combine multiple images into a single figure with legend.

    Args:
        image_paths: List of image file paths
        output_path: Output path for combined figure
        camera_names: Optional list of camera names for labels
    """
    if not image_paths:
        print("No images to combine")
        return

    images = [Image.open(p) for p in image_paths]

    # Load font - larger size for better visibility
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except Exception:
            font = ImageFont.load_default()

    # Load smaller font for origin text
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        try:
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except Exception:
            font_small = font

    # Calculate combined image size (2x2 grid layout with legend below)
    img_width = images[0].width
    img_height = images[0].height
    legend_height = 50  # Space for legend

    total_width = img_width * 2
    total_height = img_height * 2 + legend_height

    # Create combined image with space for legend
    combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(combined)

    # Paste images in 2x2 grid
    positions = [(0, 0), (img_width, 0), (0, img_height), (img_width, img_height)]
    for i, img in enumerate(images):
        if i < len(positions):
            combined.paste(img, positions[i])

    # Draw axis legend at bottom center (first line)
    legend_y = img_height * 2 + 10

    # Calculate legend text positions
    x_text = draw.textlength("X (Red)    ", font=font)
    y_text = draw.textlength("Y (Green)    ", font=font)
    z_text = draw.textlength("Z (Blue)", font=font)

    total_legend_width = x_text + y_text + z_text
    legend_x = (total_width - total_legend_width) // 2

    # Draw axis legend
    draw.text((legend_x, legend_y), "X (Red)", fill=(200, 0, 0), font=font)
    legend_x += x_text

    draw.text((legend_x, legend_y), "Y (Green)", fill=(0, 150, 0), font=font)
    legend_x += y_text

    draw.text((legend_x, legend_y), "Z (Blue)", fill=(0, 0, 200), font=font)

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
    args = parser.parse_args()

    # Path to the XML with DROID axes
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
        device_id=args.device_id
    )

    # Create combined figure
    if captured:
        image_paths = [path for _, path in captured]
        combined_path = os.path.join(args.output_dir, "sim_combined.png")
        create_combined_figure(image_paths, combined_path)

    print("Done!")


if __name__ == "__main__":
    main()
