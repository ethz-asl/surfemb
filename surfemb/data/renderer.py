from typing import Sequence

import copy
import cv2
import numpy as np
import moderngl

from .obj import Obj

from neusurfemb.misc_utils.transforms import _W_NEUS_T_W_BOP, _W_NEUS_T_SCENE
from neusurfemb.neus.network import NeuSNetwork


def orthographic_matrix(left, right, bottom, top, near, far):
    return np.array((
        (2 / (right - left), 0, 0, -(right + left) / (right - left)),
        (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
        (0, 0, -2 / (far - near), -(far + near) / (far - near)),
        (0, 0, 0, 1),
    ))


def projection_matrix(K, w, h, near=10., far=10000.):  # 1 cm to 10 m
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[:2, :3] = K[:2, :3]
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y
    # and z)
    persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl
    # NDC, therefore the -.5 below:
    orth = orthographic_matrix(-.5, w - .5, -.5, h - .5, near, far)
    return orth @ persp @ view


class NeuS2OnlineRenderer:

    def __init__(self, checkpoint_folders, objs, w, *args, h=None, **kwargs):
        assert (checkpoint_folders is not None and
                isinstance(checkpoint_folders, list) and
                len(checkpoint_folders) > 0)
        self.objs = objs

        self._W = w
        if (h is None):
            h = w
        self._H = h

        # Load NeuS2 network(s).
        self._models = [NeuSNetwork() for _ in checkpoint_folders]

        # Load pre-trained checkpoint.
        for obj_idx, checkpoint_folder in enumerate(checkpoint_folders):
            self._models[obj_idx].load_checkpoint_from_folder(
                checkpoint_folder=checkpoint_folder)

    def render(self, obj_idx, K, R, t, M_crop=None):
        # By default, render coordinates.
        return self._render(obj_idx=obj_idx,
                            K=K,
                            R=R,
                            t=t,
                            render_mode="coordinate",
                            M_crop=M_crop)

    def render_depth(self, obj_idx, K, R, t, M_crop=None):
        depth_image = self._render(obj_idx=obj_idx,
                                   K=K,
                                   R=R,
                                   t=t,
                                   render_mode="depth",
                                   M_crop=M_crop)

        # Threshold alpha channel based on density.
        depth_image[depth_image[..., 3] <= 0.8] = 0.

        return depth_image[..., 0]

    def _render(self, obj_idx, K, R, t, render_mode, M_crop=None):
        if (not (K[0, 1] == K[1, 0] == K[2, 0] == K[2, 1] == 0. and
                 K[2, 2] == 1.)):
            raise NotImplementedError(
                "Rendering with a non-pinhole camera is currently not "
                "supported with NeuS2-based renderer. Please use the original "
                "K matrix and the `M_crop` argument instead.")

        C_T_W_bop = np.concatenate((
            np.concatenate((R, t), axis=1),
            [[0, 0, 0, 1]],
        ))
        # Undo the transformations in
        # `neusurfemb/dataset_scripts/bop_dataset_to_neus.py`.
        W_bop_T_C = np.linalg.inv(C_T_W_bop.copy())
        W_bop_T_C[0:3, 2] *= -1
        W_bop_T_C[0:3, 1] *= -1
        _W_bop_T_scene = np.linalg.inv(_W_NEUS_T_W_BOP) @ _W_NEUS_T_SCENE
        scene_T_C = np.linalg.inv(_W_bop_T_scene) @ W_bop_T_C
        scene_T_C[:3, 3] = scene_T_C[:3, 3] * 1.e-3

        output = self._models[obj_idx].render_from_given_pose(
            K=K,
            W_nerf_T_C=scene_T_C,
            H=self._H,
            W=self._W,
            render_mode=render_mode,
            # Consider a discontinuity of 0.02 in the NeuS scale when rendering
            # coordinates, to filter them.
            threshold_coordinates_filtering=0.02
            if render_mode in ["coordinate", "depth"] else None)

        # Threshold alpha channel based on density.
        output[..., 3] = output[..., 3] > 0.8

        # Transform the coordinate image to the training scale and to the BOP
        # coordinate frame.
        one_uom_scene_to_m = 1. / self._models[
            obj_idx].neus.nerf.training.dataset.scale
        is_coordinate_valid = output[..., 3] == 1.
        output[..., :3][is_coordinate_valid] = (
            (
                (np.linalg.inv(_W_NEUS_T_W_BOP) @ np.hstack([
                    output[..., :3][is_coordinate_valid],
                    np.ones_like(
                        output[..., :3][is_coordinate_valid][..., 0][..., None])
                ]).T).T[..., :3] * one_uom_scene_to_m * 1000) -
            self.objs[obj_idx].offset) / self.objs[obj_idx].scale

        if (M_crop is not None):
            # Apply affine transformation.
            return cv2.warpAffine(output,
                                  M_crop,
                                  output.shape[1::-1],
                                  flags=cv2.INTER_NEAREST)
        else:
            return output


class NeuS2OfflineRenderer:

    def __init__(self, *args, **kwargs):
        # NOOP.
        pass

    def render(self, *args, **kwargs):
        # NOOP.
        return None


class ObjCoordRenderer:

    def __init__(self,
                 objs: Sequence[Obj],
                 w: int,
                 h: int = None,
                 device_idx=0):
        self.objs = objs
        if h is None:
            h = w
        self.h, self.w = h, w
        self.ctx = moderngl.create_context(standalone=True,
                                           backend='egl',
                                           device_index=device_idx)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = self.ctx.simple_framebuffer((w, h), components=4, dtype='f4')
        self.near, self.far = 10., 10000.,

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform vec3 offset;
                uniform float scale;
                uniform mat4 mvp;
                in vec3 in_vert;
                out vec3 color;
                void main() {
                    gl_Position = mvp * vec4(in_vert, 1.0);
                    color = (in_vert - offset) / scale;
                }
                """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                in vec3 color;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
                """,
        )

        self.vaos = []
        for obj in self.objs:
            vertices = obj.mesh.vertices[obj.mesh.faces].astype('f4')  # (n, 3)
            vao = self.ctx.simple_vertex_array(self.prog,
                                               self.ctx.buffer(vertices),
                                               'in_vert')
            self.vaos.append(vao)

    def read(self):
        return np.frombuffer(self.fbo.read(components=4, dtype='f4'),
                             'f4').reshape((self.h, self.w, 4))

    def read_depth(self):
        depth = np.frombuffer(self.fbo.read(attachment=-1, dtype='f4'),
                              'f4').reshape(self.h, self.w)
        neg_mask = depth == 1
        # TODO: use projection matrix instead of the default values
        near, far = 10., 10000.
        depth = 2 * depth - 1
        depth = 2 * near * far / (far + near - depth * (far - near))
        depth[neg_mask] = 0
        return depth

    def render(self, obj_idx, K, R, t, clear=True, read=True, read_depth=False):
        obj = self.objs[obj_idx]
        mv = np.concatenate((
            np.concatenate((R, t), axis=1),
            [[0, 0, 0, 1]],
        ))
        mvp = projection_matrix(K, self.w, self.h, self.near, self.far) @ mv
        self.prog['mvp'].value = tuple(mvp.T.astype('f4').reshape(-1))
        self.prog['scale'].value = obj.scale
        self.prog['offset'].value = tuple(obj.offset.astype('f4'))

        self.fbo.use()
        if clear:
            self.ctx.clear()
        self.vaos[obj_idx].render(mode=moderngl.TRIANGLES)
        if read_depth:
            return self.read_depth()
        elif read:
            return self.read()
        else:
            return None

    @staticmethod
    def extract_mask(model_coords_img: np.ndarray):
        return model_coords_img[..., 3] == 255

    def denormalize(self, model_coords: np.ndarray, obj_idx: int):
        return model_coords * self.objs[obj_idx].scale + self.objs[
            obj_idx].offset


_INFER_RENDERERS = {
    "moderngl": ObjCoordRenderer,
    "neus2_online": NeuS2OnlineRenderer
}
_RENDERERS = {
    **copy.deepcopy(_INFER_RENDERERS),
    "neus2_offline": NeuS2OfflineRenderer,
}
