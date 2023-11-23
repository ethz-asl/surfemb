import argparse
import cv2
import numpy as np
import os
import sys

_parent_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_headless_o3d_installation_dir = os.path.abspath(
    os.path.join(_parent_dir, 'Open3D/headless_installation'))
sys.path.insert(1, _headless_o3d_installation_dir)
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--W_NeuS_T_lock', nargs="+", required=True)
parser.add_argument('--H', type=int, required=True)
parser.add_argument('--W', type=int, required=True)
parser.add_argument('--H_crop', type=int, required=True)
parser.add_argument('--W_crop', type=int, required=True)
parser.add_argument('--AABB_crop', nargs='+', required=True)
parser.add_argument('--K', nargs='+', required=True)
parser.add_argument('--C_T_W_m', nargs='+', required=True)
parser.add_argument('--output_file_path', type=str, required=True)

args = parser.parse_args()

W_NeuS_T_lock = np.array([float(v) for v in args.W_NeuS_T_lock]).reshape(4, 4)
K = np.array([float(v) for v in args.K]).reshape(3, 3)
C_T_W_m = np.array([float(v) for v in args.C_T_W_m]).reshape(4, 4)
AABB_crop = [int(v) for v in args.AABB_crop]
assert (len(AABB_crop) == 4)

W_NeuS_t_lock = W_NeuS_T_lock[:3, 3]
W_NeuS_R_lock = W_NeuS_T_lock[:3, :3]
lock_coord_frame = (o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.15, origin=W_NeuS_t_lock))
lock_coord_frame.rotate(W_NeuS_R_lock, center=W_NeuS_t_lock)
lock_visualizer = o3d.visualization.Visualizer()
lock_visualizer.create_window(width=args.W, height=args.H)
lock_visualizer.clear_geometries()
lock_visualizer.add_geometry(lock_coord_frame)
lock_visualizer.update_renderer()
lock_visualizer.update_geometry(lock_coord_frame)

view_control = lock_visualizer.get_view_control()
camera_view_control = (view_control.convert_to_pinhole_camera_parameters())
camera_view_control.intrinsic.height = args.H
camera_view_control.intrinsic.width = args.W

camera_view_control.intrinsic.intrinsic_matrix = K
camera_view_control.extrinsic = C_T_W_m
view_control.convert_from_pinhole_camera_parameters(camera_view_control,
                                                    allow_arbitrary=True)
lock_visualizer.poll_events()
estimated_lock_frame = np.asarray(lock_visualizer.capture_screen_float_buffer())
# - Retrieve the actual crop used, which also corresponds to the
#   rendered pose image.
left, top, right, bottom = AABB_crop
# - Crop and resize the rendered lock image to the size of the
#   rendered pose image.
estimated_lock_frame = estimated_lock_frame[top:bottom + 1, left:right + 1]
estimated_lock_frame = cv2.resize(estimated_lock_frame,
                                  (args.W_crop, args.H_crop),
                                  interpolation=cv2.INTER_NEAREST)
# Save to file.
cv2.imwrite(args.output_file_path, estimated_lock_frame[..., ::-1])
