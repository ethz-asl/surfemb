"""
Depth refinement:
* run image crop through model
* mask-multiply query with both estimated mask probs and the pose est mask
* use query norm threshold to estimate a conservative 2d mask
  (estimating the visible mask in the encoder-decoder would make more sense.)
* find median of depth difference within the 2d mask (ignoring invalid depth)
* find com with respect to the 2d mask and use that as the ray to adjust the
  depth along
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import re
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..utils import add_timing_to_list
from ..data.config import config
from ..data.obj import load_objs
from ..data.renderer import _INFER_RENDERERS
from ..data.detector_crops import DetectorCropDataset
from ..surface_embedding import SurfaceEmbeddingModel

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--device', required=True)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--renderer-type', type=str, required=True)
parser.add_argument('--neus2-checkpoint-folders', nargs='+')
parser.add_argument('--dataset',
                    help="Dataset containing the images on which to evaluate.",
                    required=True)
parser.add_argument(
    '--surface-samples-dataset',
    help=("Dataset from which to extract the surface samples. This can be "
          "useful when using the objects reconstructed by NeuS2 instead of the "
          "CAD models."),
    required=True)

args = parser.parse_args()
res_crop = args.res_crop
device = torch.device(args.device)
model_path = Path(args.model_path)
assert model_path.is_file()
model_name = model_path.name.rsplit('.', maxsplit=1)[0]
dataset = args.dataset
renderer_type = args.renderer_type
neus2_checkpoint_folders = args.neus2_checkpoint_folders

if (not renderer_type in _INFER_RENDERERS):
    raise ValueError(f"Invalid value '{renderer_type}' for `renderer_type`. "
                     f"Valid values are: {sorted(_INFER_RENDERERS.keys())}.")

kwargs_renderer = {}
if (renderer_type == "neus2_online"):
    kwargs_renderer["checkpoint_folders"] = args.neus2_checkpoint_folders

cfg = config[dataset]
root = Path('data/bop') / dataset
test_folder = root / cfg.test_folder
assert root.exists()

poses = np.load(f'data/results/{model_name}-poses.npy')
poses_timings = np.load(f'data/results/{model_name}-poses-timings.npy')
poses_depth = poses.copy()
poses_depth_fp = Path(f'data/results/{model_name}-depth-poses.npy')
poses_depth_timings_fp = Path(
    f'data/results/{model_name}-depth-poses-timings.npy')
for fp in poses_depth_fp, poses_depth_timings_fp:
    assert not fp.exists()

model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path)).to(device)
model.eval()
model.freeze()

objs, obj_ids = load_objs(
    Path('data/bop') / args.surface_samples_dataset / cfg.model_folder)
assert len(obj_ids) > 0
# If the model was only trained for one object, only evaluate on that.
obj_id_in_name = re.findall(r"0\d{5}[-_]", (model_path).parts[-1])
if (len(obj_id_in_name) > 0):
    assert (len(obj_id_in_name) == 1)
    assert (len(model.cnn.decoders) == 1)
    obj_id_in_name = int(obj_id_in_name[0][:-1])
    objs = [objs[obj_ids.index(obj_id_in_name)]]
    obj_ids = [obj_ids[obj_ids.index(obj_id_in_name)]]
dataset = DetectorCropDataset(
    dataset_root=root,
    obj_ids=obj_ids,
    cfg=cfg,
    detection_folder=Path(f'data/detection_results/{dataset}'),
    auxs=model.get_infer_auxs(objs=objs,
                              crop_res=res_crop,
                              from_detections=True))
assert poses.shape[1] == len(dataset)

crop_renderer = _INFER_RENDERERS[renderer_type](objs=objs,
                                                w=res_crop,
                                                h=res_crop,
                                                **kwargs_renderer)
n_failed = 0
all_depth_timings = [[], []]
for j in range(2):
    depth_timings = all_depth_timings[j]
    for i, d in enumerate(tqdm(dataset)):
        if (d is None):
            continue
        pose = poses[j, i]  # (3, 4)
        R = pose[:3, :3]
        t = pose[:3, 3:]

        obj_idx, K_crop, K = d['obj_idx'], d['K_crop'], d['K']

        depth_sensor = cv2.imread(
            str(test_folder /
                f'{d["scene_id"]:06d}/depth/{d["img_id"]:06d}.{cfg.depth_ext}'),
            cv2.IMREAD_UNCHANGED)
        scene_camera_fp = test_folder / f'{d["scene_id"]:06d}/scene_camera.json'
        depth_scale = json.load(scene_camera_fp.open())[str(
            d['img_id'])]['depth_scale']
        depth_sensor = depth_sensor * depth_scale
        h, w = depth_sensor.shape

        mask_lgts, query_img = [
            v.cpu() for v in model.infer_cnn(d['rgb_crop'], obj_idx)
        ]
        # the above either doesn't count towards the time (loading images),
        # or has already been done in the initial pose estimate (cnn forward
        # pass) so timing starts here:
        with add_timing_to_list(depth_timings):
            depth_sensor_mask = (depth_sensor > 0).astype(np.float32)
            M = (K_crop @ np.linalg.inv(K))[:2]
            depth_sensor_mask_crop = cv2.warpAffine(
                depth_sensor_mask,
                M, (res_crop, res_crop),
                flags=cv2.INTER_LINEAR) == 1.
            depth_sensor_crop = cv2.warpAffine(depth_sensor,
                                               M, (res_crop, res_crop),
                                               flags=cv2.INTER_LINEAR)
            if (renderer_type == "moderngl"):
                depth_render = crop_renderer.render(obj_idx,
                                                    K_crop,
                                                    R,
                                                    t,
                                                    read_depth=True)
            else:
                depth_render = crop_renderer.render_depth(obj_idx, K_crop, R, t)
            render_mask = depth_render > 0

            query_img_norm = torch.norm(query_img,
                                        dim=-1) * torch.sigmoid(mask_lgts)
            query_img_norm = query_img_norm.numpy(
            ) * render_mask * depth_sensor_mask_crop
            norm_sum = query_img_norm.sum()
            if norm_sum == 0:
                n_failed += 1
                continue
            query_img_norm /= norm_sum
            norm_mask = query_img_norm > (query_img_norm.max() * 0.8)
            yy, xx = np.argwhere(norm_mask).T  # 2 x (N,)
            depth_diff = depth_sensor_crop[yy, xx] - depth_render[yy, xx]
            depth_adjustment = np.median(depth_diff)

            yx_coords = np.meshgrid(np.arange(res_crop), np.arange(res_crop))
            yx_coords = np.stack(yx_coords[::-1],
                                 axis=-1)  # (res_crop, res_crop, 2yx)
            yx_ray_2d = (yx_coords * query_img_norm[..., None]).sum(
                axis=(0, 1))  # y, x
            ray_3d = np.linalg.inv(K_crop) @ (*yx_ray_2d[::-1], 1)
            ray_3d /= ray_3d[2]

            t_depth_refined = t + ray_3d[:, None] * depth_adjustment
            poses_depth[j, i, :3, 3:] = t_depth_refined

        if args.debug:
            axs = plt.subplots(3, 4, figsize=(12, 9))[1]
            axs[0, 0].imshow(d['rgb'])
            axs[0, 0].set_title('rgb')
            axs[0, 1].imshow(depth_sensor)
            axs[0, 1].set_title('depth')
            axs[0, 2].imshow(depth_render)
            axs[0, 2].set_title('depth render')

            axs[1, 0].imshow(d['rgb_crop'])
            axs[1, 0].set_title('rgb crop')
            axs[1, 0].scatter(xx, yy)
            axs[1, 1].imshow(depth_sensor_crop)
            axs[1, 1].set_title('depth crop')
            axs[1, 2].imshow(query_img_norm)
            axs[1, 2].scatter(*yx_ray_2d[None, ::-1].T, c='r')
            axs[1, 2].set_title('query norm')
            axs[1, 3].imshow(norm_mask)
            axs[1, 3].set_title('norm mask')

            axs[2, 0].imshow(d['rgb_crop'])
            axs[2, 0].imshow(render_mask, alpha=0.5)
            axs[2, 0].set_title('initial pose')
            if (renderer_type == "moderngl"):
                render_mask_after = crop_renderer.render(obj_idx,
                                                         K_crop,
                                                         R,
                                                         t_depth_refined,
                                                         read_depth=True) > 0
            else:
                render_mask_after = crop_renderer.render_depth(
                    obj_idx, K_crop, R, t_depth_refined) > 0
            axs[2, 1].imshow(d['rgb_crop'])
            axs[2, 1].imshow(render_mask_after, alpha=0.5)
            axs[2, 1].set_title('pose after depth refine')
            axs[2, 3].hist(depth_diff)
            axs[2, 3].plot([depth_adjustment, depth_adjustment],
                           [0, max(axs[2, 3].get_ylim())])
            axs[2, 3].set_title('depth diff. hist.')
            for ax in axs.reshape(-1)[:-1]:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

poses_depth_timings = poses_timings + np.array(all_depth_timings)
print(f'{n_failed / (len(dataset) * 2):.3f} failed')

if not args.debug:
    np.save(str(poses_depth_fp), poses_depth)
    np.save(str(poses_depth_timings_fp), poses_depth_timings)
