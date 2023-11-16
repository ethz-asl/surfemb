import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from typing import List, Optional

from .data.obj import load_objs
from .data.renderer import _INFER_RENDERERS
from .data.std_auxs import RandomRotatedMaskCrop
from .pose_est import estimate_pose
from .pose_refine import refine_pose
from .surface_embedding import SurfaceEmbeddingModel
from .utils import load_surface_samples


class PoseEstimator:

    def __init__(self,
                 model_path: str,
                 object_model_folder: str,
                 K: np.ndarray,
                 use_normals_criterion: str,
                 renderer_type: str,
                 neus2_checkpoint_folders: Optional[List] = None,
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
            kwargs_renderer[
                "checkpoint_folders"] = args.neus2_checkpoint_folders

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

        # Set up the renderer.
        self._renderer = _INFER_RENDERERS[renderer_type](objs=self._objs,
                                                         w=res_crop,
                                                         h=res_crop,
                                                         **kwargs_renderer)

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

    def estimate_pose(self, image, bbox):
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
            R_est_r, t_est_r, score_r = refine_pose(
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
        else:
            R_est_r, t_est_r = R_est, t_est

        # Show rendered pose.
        render = self._renderer.render(0, K_crop, R_est_r, t_est_r)
        render_mask = render[..., 3] == 1.
        pose_img = img.copy()
        pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[
            ..., :3][render_mask] * 0.25 + 0.25

        plt.imshow(pose_img)
        plt.show()
