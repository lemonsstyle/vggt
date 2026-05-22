# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def depth_to_rgb_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth_image = np.asarray(depth_image, dtype=np.float32)
    valid_mask = np.isfinite(depth_image) & (depth_image > 0)
    color_image = np.zeros((*depth_image.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return color_image

    valid_depth = depth_image[valid_mask]
    depth_min = float(valid_depth.min())
    depth_max = float(valid_depth.max())
    if depth_max <= depth_min:
        color_image[valid_mask] = np.array([0, 255, 0], dtype=np.uint8)
        return color_image

    normalized = (depth_image - depth_min) / (depth_max - depth_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color_image[valid_mask] = colored[valid_mask]
    return color_image


def export_point_cloud_ply(points, colors, filename="point_cloud.ply"):
    """
    Export point cloud to PLY format.
    
    Args:
        points: numpy array of shape (N, 3) containing XYZ coordinates
        colors: numpy array of shape (N, 3) containing RGB values (0-255)
        filename: output filename
    """
    # Ensure colors are in 0-255 range and uint8
    if colors.dtype != np.uint8:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    
    # PLY header
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(points)):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} ")
            f.write(f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
    
    print(f"Point cloud saved to {filename} ({len(points)} points)")


def export_point_cloud_xyz(points, colors, filename="point_cloud.xyz"):
    """
    Export point cloud to XYZ format with RGB colors.
    
    Args:
        points: numpy array of shape (N, 3) containing XYZ coordinates
        colors: numpy array of shape (N, 3) containing RGB values (0-255)
        filename: output filename
    """
    # Ensure colors are in 0-1 range for XYZ format
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0
    elif colors.max() > 1.0:
        colors = colors.astype(np.float32) / 255.0
    
    # Combine points and colors
    data = np.hstack([points, colors])
    
    # Save to file
    np.savetxt(filename, data, fmt='%.6f %.6f %.6f %.3f %.3f %.3f', 
               header='x y z r g b', comments='')
    
    print(f"Point cloud saved to {filename} ({len(points)} points)")


def export_depth_maps(
    depth_map: np.ndarray,
    confidence_map: np.ndarray,
    image_paths: List[str],
    output_dir: str,
    confidence_percent: float,
    depth_visualization: str,
) -> None:
    """
    Export predicted depth maps as confidence-filtered raw 16-bit PNGs.

    Args:
        depth_map: Array of shape (S, H, W, 1) or (S, H, W).
        confidence_map: Array of shape (S, H, W) used for filtering.
        image_paths: Input image paths used for naming outputs.
        output_dir: Directory for exported files.
        confidence_percent: Percentile used to filter out low-confidence pixels.
        depth_visualization: "Gray" or "RGB" output format.
    """
    os.makedirs(output_dir, exist_ok=True)

    depth_map = np.asarray(depth_map)
    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]

    confidence_map = np.asarray(confidence_map)
    confidence_threshold = np.percentile(confidence_map.reshape(-1), confidence_percent)

    export_count = min(len(image_paths), depth_map.shape[0])
    for idx in range(export_count):
        image_path = image_paths[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cur_depth = np.asarray(depth_map[idx], dtype=np.float32)
        cur_conf = np.asarray(confidence_map[idx], dtype=np.float32)
        cur_depth = np.nan_to_num(cur_depth, nan=0.0, posinf=0.0, neginf=0.0)
        cur_depth[cur_depth < 0] = 0.0
        valid_mask = (cur_depth > 0) & np.isfinite(cur_depth) & (cur_conf >= confidence_threshold) & (cur_conf > 1e-5)
        cur_depth = np.where(valid_mask, cur_depth, 0.0)

        depth_png_path = os.path.join(output_dir, f"{base_name}_depth.png")
        if depth_visualization == "RGB":
            depth_vis = depth_to_rgb_colormap(cur_depth)
            cv2.imwrite(depth_png_path, depth_vis)
        else:
            raw_depth_png = cur_depth.astype(np.float16).view(np.uint16)
            cv2.imwrite(depth_png_path, raw_depth_png)

    print(f"Depth maps saved to {output_dir}")


def viser_wrapper(
    pred_dict: dict,
    port: int = 6006,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
    auto_export: bool = False,  # New parameter for automatic export
    image_paths: Optional[List[str]] = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
        auto_export (bool): Whether to automatically export point cloud on startup.
        image_paths (List[str] | None): Original image file paths for naming exports.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    depth_conf_export = depth_conf.copy()

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)
        depth_conf_export = apply_sky_segmentation(depth_conf_export, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Store current filtered points and colors for export
    current_points = None
    current_colors = None

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Add export section
    server.gui.add_markdown("## Export Point Cloud")
    export_format = server.gui.add_dropdown(
        "Export Format", options=["PLY", "XYZ", "Depth"], initial_value="PLY"
    )
    depth_visualization = server.gui.add_dropdown(
        "Depth Color", options=["Gray", "RGB"], initial_value="Gray"
    )
    depth_visualization.visible = False
    export_button = server.gui.add_button("Export", color="green")
    export_status = server.gui.add_markdown("")

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # Update current points and colors
    current_points = points_centered[init_conf_mask]
    current_colors = colors_flat[init_conf_mask]

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        nonlocal current_points, current_colors
        
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        current_points = points_centered[combined_mask]
        current_colors = colors_flat[combined_mask]
        
        point_cloud.points = current_points
        point_cloud.colors = current_colors
        
        # Update export status with current point count
        export_status.content = f"Current points: {len(current_points):,}"

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    @export_format.on_update
    def _(_) -> None:
        depth_visualization.visible = export_format.value == "Depth"

    @export_button.on_click
    def _(_) -> None:
        """Export the current point cloud."""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"vggt_pointcloud_{timestamp}"
        
        try:
            if export_format.value == "Depth":
                if image_paths is None or len(image_paths) == 0:
                    export_status.content = "❌ No image paths available for depth export!"
                    return
                depth_dir = f"depth_{timestamp}"
                export_depth_maps(
                    depth_map,
                    depth_conf_export,
                    image_paths[: depth_map.shape[0]],
                    depth_dir,
                    gui_points_conf.value,
                    depth_visualization.value,
                )
                export_status.content = f"✅ Exported depth maps to {depth_dir}"
            else:
                if current_points is None or len(current_points) == 0:
                    export_status.content = "❌ No points to export!"
                    return
                if export_format.value == "PLY":
                    filename = f"{base_name}.ply"
                    export_point_cloud_ply(current_points, current_colors, filename)
                else:  # XYZ
                    filename = f"{base_name}.xyz"
                    export_point_cloud_xyz(current_points, current_colors, filename)
                
                export_status.content = f"✅ Exported {len(current_points):,} points to {filename}"
        except Exception as e:
            export_status.content = f"❌ Export failed: {str(e)}"

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)
    
    # Update initial status
    update_point_cloud()

    # Auto-export if requested
    if auto_export:
        print("Auto-exporting initial point cloud...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vggt_pointcloud_auto_{timestamp}.ply"
        export_point_cloud_ply(current_points, current_colors, filename)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=6006, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--auto_export", action="store_true", help="Automatically export point cloud on startup")


def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser
    6. Provides options to export the point cloud

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    --auto_export: Automatically export point cloud on startup
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")
    
    if args.auto_export:
        print("Auto-export enabled - will save point cloud automatically")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        auto_export=args.auto_export,
        image_paths=image_names,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
