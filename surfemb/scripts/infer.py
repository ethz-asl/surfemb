import argparse
from pathlib import Path

import numpy as np
import re
import torch
from tqdm import tqdm

from .. import utils
from ..data import detector_crops
from ..data.config import config
from ..data.obj import load_objs
from ..data.renderer import _INFER_RENDERERS
from ..surface_embedding import SurfaceEmbeddingModel
from ..pose_est import estimate_pose
from ..pose_refine import refine_pose

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--res-data', type=int, default=256)
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument(
    '--use-normals-criterion',
    type=str,
    required=True,
    help=("Whether to use criterion based on normals to filter out pose "
          "hypotheses."))
parser.add_argument('--renderer-type', type=str, required=True)
parser.add_argument('--neus2-checkpoint-folders', nargs='+')
# Unused.
# parser.add_argument('--max-poses', type=int, default=10000)
# parser.add_argument('--max-pose-evaluations', type=int, default=1000)
parser.add_argument('--no-rotation-ensemble',
                    dest='rotation_ensemble',
                    action='store_false')
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

results_dir = Path('data/results')
results_dir.mkdir(exist_ok=True)
poses_fp = results_dir / f'{model_name}-poses.npy'
poses_scores_fp = results_dir / f'{model_name}-poses-scores.npy'
poses_timings_fp = results_dir / f'{model_name}-poses-timings.npy'
for fp in poses_fp, poses_scores_fp, poses_timings_fp:
    assert not fp.exists()

# load model
model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path)).eval().to(
    device)  # type: SurfaceEmbeddingModel
model.freeze()

# load data
cfg = config[dataset]
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

surface_samples, surface_sample_normals = utils.load_surface_samples(
    args.surface_samples_dataset, obj_ids)
data = detector_crops.DetectorCropDataset(
    dataset_root=Path('data/bop') / dataset,
    cfg=cfg,
    obj_ids=obj_ids,
    detection_folder=Path(f'data/detection_results/{dataset}'),
    auxs=model.get_infer_auxs(objs=objs,
                              crop_res=res_crop,
                              from_detections=True))
renderer = _INFER_RENDERERS[renderer_type](objs=objs,
                                           w=res_crop,
                                           h=res_crop,
                                           **kwargs_renderer)

# infer
all_poses = np.empty((2, len(data), 3, 4))
all_scores = np.ones(len(data)) * -np.inf
time_forward, time_pnpransac, time_refine = [], [], []


def infer(i, d):
    obj_idx = d['obj_idx']
    img = d['rgb_crop']
    K_crop = d['K_crop']

    with utils.add_timing_to_list(time_forward):
        mask_lgts, query_img = model.infer_cnn(
            img, obj_idx, rotation_ensemble=args.rotation_ensemble)
        mask_lgts[0, 0].item()  # synchronize for timing

        # keys are independent of input (could be cached, but it's not the
        # bottleneck)
        obj_ = objs[obj_idx]
        verts = surface_samples[obj_idx]
        verts_norm = (verts - obj_.offset) / obj_.scale
        obj_keys = model.infer_mlp(
            torch.from_numpy(verts_norm).float().to(device), obj_idx)
        verts = torch.from_numpy(verts).float().to(device)

    with utils.add_timing_to_list(time_pnpransac):
        R_est, t_est, scores, *_ = estimate_pose(
            mask_lgts=mask_lgts,
            query_img=query_img,
            obj_pts=verts,
            obj_normals=surface_sample_normals[obj_idx],
            obj_keys=obj_keys,
            obj_diameter=obj_.diameter,
            K=K_crop,
            use_normals_criterion=args.use_normals_criterion)
        success = len(scores) > 0
        if success:
            best_idx = torch.argmax(scores).item()
            all_scores[i] = scores[best_idx].item()
            R_est, t_est = R_est[best_idx].cpu().numpy(), t_est[best_idx].cpu(
            ).numpy()[:, None]
        else:
            R_est, t_est = np.eye(3), np.zeros((3, 1))

    with utils.add_timing_to_list(time_refine):
        if success:
            R_est_r, t_est_r, score_r = refine_pose(
                R=R_est,
                t=t_est,
                query_img=query_img,
                K_crop=K_crop,
                renderer=renderer,
                obj_idx=obj_idx,
                obj_=obj_,
                model=model,
                keys_verts=obj_keys,
            )
        else:
            R_est_r, t_est_r = R_est, t_est

    for j, (R, t) in enumerate([(R_est, t_est), (R_est_r, t_est_r)]):
        all_poses[j, i, :3, :3] = R
        all_poses[j, i, :3, 3:] = t


for i, d in enumerate(tqdm(data, desc='running pose est.', smoothing=0)):
    if (d is not None):
        infer(i, d)

time_forward = np.array(time_forward)
time_pnpransac = np.array(time_pnpransac)
time_refine = np.array(time_refine)

timings = np.stack((time_forward + time_pnpransac,
                    time_forward + time_pnpransac + time_refine))

np.save(str(poses_fp), all_poses)
np.save(str(poses_scores_fp), all_scores)
np.save(str(poses_timings_fp), timings)
