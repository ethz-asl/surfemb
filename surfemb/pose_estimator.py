import cv2
import numpy as np
import os
from pathlib import Path
import torch
from typing import List, Optional
import yaml

from .data.obj import load_objs
from .data.renderer import _INFER_RENDERERS
from .data.std_auxs import RandomRotatedMaskCrop
from .pose_est import estimate_pose
from .pose_refine import refine_pose
from .surface_embedding import SurfaceEmbeddingModel
from .utils import load_surface_samples


class PoseEstimator:

    @classmethod
    def from_flags(cls, flags_yaml_path):
        # Read flags from file.
        with open(flags_yaml_path, "r") as f:
            flags = yaml.load(f, Loader=yaml.SafeLoader)

        # Use the folder of the config file as the parent folder.
        parent_folder = os.path.dirname(flags_yaml_path)

        intrinsics_path = os.path.join(parent_folder,
                                       flags.pop('intrinsics_path'))
        flags["K"] = np.loadtxt(intrinsics_path)
        for flag in [
                "model_path", "object_model_folder",
                "orig_frame_T_lock_center_path"
        ]:
            try:
                flags[flag] = os.path.join(parent_folder, flags[flag])
            except KeyError:
                pass
        if ("neus2_checkpoint_folders" in flags):
            assert (isinstance(flags["neus2_checkpoint_folders"], list))
            for checkpoint_folder_idx in range(
                    len(flags["neus2_checkpoint_folders"])):
                flags["neus2_checkpoint_folders"][
                    checkpoint_folder_idx] = os.path.join(
                        parent_folder, flags["neus2_checkpoint_folders"]
                        [checkpoint_folder_idx])

        return cls(**flags)

    def __init__(self,
                 model_path: str,
                 object_model_folder: str,
                 K: np.ndarray,
                 use_normals_criterion: str,
                 renderer_type: str,
                 neus2_checkpoint_folders: Optional[List] = None,
                 orig_frame_T_lock_center_path: Optional[str] = None,
                 device: str = "cuda:0",
                 res_crop: int = 224,
                 no_rotation_ensemble: bool = False):
        r"""Pose estimation class.

        Args:
            model_path (str): Path to the model checkpoint to use.
            object_model_folder (str): Path to the folder containing the 3D
                object model information. In particular, it is assumed that:
                - The subfolder `models` contains the mesh file of the object.
                  This is used to rescale the reference 3D point cloud and the
                  rendered coordinates in all the cases, also when the renderer
                  does not require a mesh model (i.e., when `renderer_type` is
                  `'neus2_online'`).
                - The subfolder `surface_samples` contains the surface samples
                  that define the 3D points in the reference (key) point cloud.
                - The subfolder `surface_samples_normals` contains the normal
                  vectors at the 3D points in the reference (key) point cloud.
            K (np.ndarray): Camera intrinsic matrix.
            use_normals_criterion (bool): Whether to use criterion based on
                normals to filter out pose hypotheses ("true" or "false").
            renderer_type (str): Type of renderer to use. Valid options are:
                `'moderngl'`, `'neus2_online'`.
            neus2_checkpoint_folders (str): List of path of the folders
                containing the NeuS2 model of the objects to evaluate on. Only
                required when `renderer_type` is `'neus2_online'`. NOTE: Each
                folder is supposed to contain a "checkpoints/neus" subfolder
                with the actual checkpoint.
            orig_frame_T_lock_center_path (str): If not None, path to the
                `orig_frame_T_lock_center.txt` defining the transformation
                matrix from the lock-center coordinate frame to the world
                coordinate frame of the object model, in meters.
            device (str): Device to use.
            res_crop (int): Resolution of the crop to use.
            no_rotation_ensemble (bool): If False, no rotation ensemble is used
                at test time.
        """
        self._res_crop = res_crop
        self._device = torch.device(device)
        self._model_path = Path(model_path)
        assert (self._model_path.is_file())
        self._model_name = self._model_path.name.rsplit('.', maxsplit=1)[0]
        assert (isinstance(K, np.ndarray) and K.shape == (3, 3))
        self._K = K
        self._renderer_type = renderer_type
        self._neus2_checkpoint_folders = neus2_checkpoint_folders
        self._rotation_ensemble = not no_rotation_ensemble
        self._use_normals_criterion = use_normals_criterion

        if (not renderer_type in _INFER_RENDERERS):
            raise ValueError(
                f"Invalid value '{renderer_type}' for `renderer_type`. Valid "
                f"values are: {sorted(_INFER_RENDERERS.keys())}.")

        kwargs_renderer = {}
        if (renderer_type == "neus2_online"):
            kwargs_renderer["checkpoint_folders"] = neus2_checkpoint_folders

        # Load model.
        self.model = SurfaceEmbeddingModel.load_from_checkpoint(
            str(self._model_path)).eval().to(
                device)  # type: SurfaceEmbeddingModel
        self.model.freeze()

        # Load data.
        # - Load 3D mesh model.
        self._objs, self._obj_ids = load_objs(
            Path(object_model_folder) / "models")
        assert (len(self._obj_ids) == 1)
        self._objs = [self._objs[0]]
        self._obj_ids = [self._obj_ids[0]]
        # - Load 3D surface samples.
        (self._surface_samples,
         self._surface_samples_normals) = load_surface_samples(
             "", self._obj_ids, root=Path(object_model_folder))
        # - If available, load the file `orig_frame_T_lock_center.txt`.
        if (orig_frame_T_lock_center_path is None):
            self._W_NeuS_T_lock = None
        else:
            # - First, import the headless Open3D installation.
            import sys
            _parent_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            _headless_o3d_installation_dir = os.path.abspath(
                os.path.join(_parent_dir, 'Open3D/headless_installation'))
            sys.path.insert(1, _headless_o3d_installation_dir)
            import open3d as o3d
            self.__o3d = o3d

            self._W_NeuS_T_lock = np.loadtxt(orig_frame_T_lock_center_path)
            assert (self._W_NeuS_T_lock.shape == (4, 4))

        # Set up the renderer.
        self._renderer = _INFER_RENDERERS[renderer_type](objs=self._objs,
                                                         w=res_crop,
                                                         h=res_crop,
                                                         **kwargs_renderer)
        self._renderer_type = renderer_type

        # Set up infer-time augmentations.
        self._auxs = [
            RandomRotatedMaskCrop(res_crop,
                                  max_angle=0,
                                  offset_scale=0,
                                  use_bbox=True,
                                  rgb_interpolation=(cv2.INTER_LINEAR,),
                                  crop_keys=['rgb'])
        ]
        for aux in self._auxs:
            aux.init(self)

        # Pre-compute key features.
        self._obj = self._objs[0]
        self._verts = self._surface_samples[0]
        verts_norm = (self._verts - self._obj.offset) / self._obj.scale
        self._obj_keys = self.model.infer_mlp(
            torch.from_numpy(verts_norm).float().to(device), obj_idx=0)
        self._verts = torch.from_numpy(self._verts).float().to(device)

    def estimate_pose(self,
                      image,
                      bbox,
                      depth_image=None,
                      depth_scale=1.0,
                      visualize_estimated_pose=False):
        instance = dict(
            rgb=image.copy(),
            K=self._K,
            bbox=bbox,
        )
        # - Apply augmentations to the image.
        for aux in self._auxs:
            instance = aux(instance, self)

        img = instance['rgb_crop']
        K_crop = instance['K_crop']

        # Forward pass.
        mask_lgts, query_img = self.model.infer_cnn(
            img, obj_idx=0, rotation_ensemble=self._rotation_ensemble)
        mask_lgts[0, 0].item()

        # PnP-RANSAC.
        R_est, t_est, scores, *_ = estimate_pose(
            mask_lgts=mask_lgts,
            query_img=query_img,
            obj_pts=self._verts,
            obj_normals=self._surface_samples_normals[0],
            obj_keys=self._obj_keys,
            obj_diameter=self._obj.diameter,
            K=K_crop,
            use_normals_criterion=self._use_normals_criterion)
        success = len(scores) > 0
        if (success):
            best_idx = torch.argmax(scores).item()
            R_est, t_est = R_est[best_idx].cpu().numpy(), t_est[best_idx].cpu(
            ).numpy()[:, None]
        else:
            R_est, t_est = np.eye(3), np.zeros((3, 1))

        # Refinement.
        if (success):
            R_est, t_est, score_r = refine_pose(
                R=R_est,
                t=t_est,
                query_img=query_img,
                K_crop=K_crop,
                renderer=self._renderer,
                obj_idx=0,
                obj_=self._obj,
                model=self.model,
                keys_verts=self._obj_keys,
            )

        # Potentially apply depth-based refinement.
        if (depth_image is not None):
            depth_image = depth_image * depth_scale

            depth_sensor_mask = (depth_image > 0).astype(np.float32)
            M = (K_crop @ np.linalg.inv(self._K))[:2]
            depth_sensor_mask_crop = cv2.warpAffine(
                depth_sensor_mask,
                M, (self._res_crop, self._res_crop),
                flags=cv2.INTER_LINEAR) == 1.
            depth_sensor_crop = cv2.warpAffine(depth_image,
                                               M,
                                               (self._res_crop, self._res_crop),
                                               flags=cv2.INTER_LINEAR)
            if (self._renderer_type == "moderngl"):
                depth_render = self._renderer.render(0,
                                                     K_crop,
                                                     R_est,
                                                     t_est,
                                                     read_depth=True)
            else:
                depth_render = self._renderer.render_depth(
                    0, K_crop, R_est, t_est)
            render_mask = depth_render > 0

            query_img_norm = torch.norm(query_img,
                                        dim=-1) * torch.sigmoid(mask_lgts)
            query_img_norm = query_img_norm.cpu().numpy(
            ) * render_mask * depth_sensor_mask_crop
            norm_sum = query_img_norm.sum()
            if (norm_sum != 0):
                query_img_norm /= norm_sum
                norm_mask = query_img_norm > (query_img_norm.max() * 0.8)
                yy, xx = np.argwhere(norm_mask).T  # 2 x (N,)
                depth_diff = depth_sensor_crop[yy, xx] - depth_render[yy, xx]
                depth_adjustment = np.median(depth_diff)

                yx_coords = np.meshgrid(np.arange(self._res_crop),
                                        np.arange(self._res_crop))
                yx_coords = np.stack(
                    yx_coords[::-1],
                    axis=-1)  # (self._res_crop, self._res_crop, 2yx)
                yx_ray_2d = (yx_coords * query_img_norm[..., None]).sum(
                    axis=(0, 1))  # y, x
                ray_3d = np.linalg.inv(K_crop) @ (*yx_ray_2d[::-1], 1)
                ray_3d /= ray_3d[2]

                t_est = t_est + ray_3d[:, None] * depth_adjustment

        # Convert the estimated pose to meters.
        C_T_W_mm = np.eye(4)
        C_T_W_mm[:3, :3] = R_est
        C_T_W_mm[:3, 3] = t_est[..., 0]

        C_T_W_m = C_T_W_mm.copy()
        C_T_W_m[:3, 3] = C_T_W_m[:3, 3] / 1000.

        # Render estimated pose.
        render = self._renderer.render(0, K_crop, R_est, t_est)
        render_mask = render[..., 3] == 1.
        pose_img = img.copy()
        pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[
            ..., :3][render_mask] * 0.25 + 0.25

        # - If available, render and overlay the lock frame.
        if (self._W_NeuS_T_lock is not None):
            H, W = image.shape[:2]
            H_crop, W_crop = img.shape[:2]
            # - Call the script to render the lock frame to a temporary file. It
            #   is required to perform this in a separate thread, because the
            #   Open3D renderer for the lock frame conflicts with the `moderngl`
            #   object renderer.
            curr_file_path = os.path.dirname(os.path.realpath(__file__))
            script_path = os.path.abspath(
                os.path.join(curr_file_path,
                             'scripts/misc/render_lock_frame.py'))
            tmp_file_path = os.path.abspath(
                os.path.join(curr_file_path, "tmp_lock_frame.png"))
            return_code = os.system(
                f"python {script_path} --W_NeuS_T_lock " +
                ''.join([str(v) + ' '
                         for v in self._W_NeuS_T_lock.reshape(-1)]) +
                f"--H {H} --W {W}  --H_crop {H_crop} --W_crop {W_crop} "
                "--AABB_crop " +
                ''.join(str(v) + ' ' for v in instance['AABB_crop']) + "--K " +
                ''.join([str(v) + ' ' for v in self._K.reshape(-1)]) +
                "--C_T_W_m " +
                ''.join([str(v) + ' ' for v in C_T_W_m.reshape(-1)]) +
                f"--output_file_path {tmp_file_path}")
            assert (return_code == 0
                   ), "Error in the subprocess to render the lock frame."
            # - Read from file the temporary lock-frame image and overlay it to
            #   the rendered pose image.
            estimated_lock_frame = cv2.imread(tmp_file_path)
            pose_img[
                estimated_lock_frame != [1., 1., 1.]] = estimated_lock_frame[
                    estimated_lock_frame != [1., 1., 1.]]
            # - Delete the temporary lock-frame image.
            os.remove(tmp_file_path)

        if (visualize_estimated_pose):
            # Optionally visualize estimated pose.
            cv2.imshow("Estimated pose", pose_img[..., ::-1])
            cv2.waitKey(0)

        # If a lock coordinate frame is available, return the transform from
        # this frame rather than the object world frame.
        if (self._W_NeuS_T_lock is not None):
            C_T_W_m = C_T_W_m @ self._W_NeuS_T_lock

        return C_T_W_m, pose_img
