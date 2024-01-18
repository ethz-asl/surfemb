"""Adapted from https://github.com/mahmoudnafifi/WB_color_augmenter,
released by Mahmoud Afifi and Michael S. Brown under MIT License.
Reference:
M. Afifi and M. S. Brown, "What Else Can Fool Deep Learning? Addressing Color
Constancy Errors on Deep Neural Network Performance", in ICCV (2019).
"""
from math import ceil
import numpy as np
from numpy.matlib import repmat
import os
import random as rnd

curr_folder = os.path.dirname(__file__)

# source: https://github.com/fatheral/matlab_imresize


def derive_size_from_scale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def derive_scale_from_size(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply(
        (1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(
        left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices -
                1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1),
                                                axis=1))
    aux = np.concatenate(
        (np.arange(in_length), np.arange(in_length - 1, -1,
                                         step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresize_mex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w,
                       i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0),
                                                   w.T),
                                       axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img,
                       i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0),
                                                 w.T),
                                     axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresize_vec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights *
                        ((inimg[indices].squeeze(axis=1)).astype(np.float64)),
                        axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(
            weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)),
            axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resize_along_dim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresize_mex(A, weights, indices, dim)
    else:
        out = imresize_vec(A, weights, indices, dim)
    return out


def imresize(I,
             scalar_scale=None,
             method='bicubic',
             output_shape=None,
             mode="vec"):
    if method is 'bicubic':
        kernel = cubic
    elif method is 'bilinear':
        kernel = triangle
    else:
        print('Error: Unidentified method supplied')

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = derive_size_from_scale(I.shape, scale)
    elif output_shape is not None:
        scale = derive_scale_from_size(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel,
                               kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resize_along_dim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


class WBEmulator:

    def __init__(self):
        # Training encoded features.
        self.features = np.load(os.path.join(curr_folder,
                                             'params/features.npy'))
        # Mapping functions to emulate WB effects.
        self.mappingFuncs = np.load(
            os.path.join(curr_folder, 'params/mappingFuncs.npy'))
        # Weight matrix for histogram encoding.
        self.encoderWeights = np.load(
            os.path.join(curr_folder, 'params/encoderWeights.npy'))
        # Bias vector for histogram encoding.
        self.encoderBias = np.load(
            os.path.join(curr_folder, 'params/encoderBias.npy'))
        self.h = 60  # Histogram bin width.
        self.K = 25  # K value for nearest neighbor searching.
        self.sigma = 0.25  # Fall-off factor for KNN.
        # WB & photo finishing styles.
        self.wb_photo_finishing = [
            '_F_AS', '_F_CS', '_S_AS', '_S_CS', '_T_AS', '_T_CS', '_C_AS',
            '_C_CS', '_D_AS', '_D_CS'
        ]

    def encode(self, hist):
        """Generates a compacted feature of a given RGB-uv histogram tensor."""
        histR_reshaped = np.reshape(np.transpose(hist[:, :, 0]),
                                    (1, int(hist.size / 3)),
                                    order="F")
        histG_reshaped = np.reshape(np.transpose(hist[:, :, 1]),
                                    (1, int(hist.size / 3)),
                                    order="F")
        histB_reshaped = np.reshape(np.transpose(hist[:, :, 2]),
                                    (1, int(hist.size / 3)),
                                    order="F")
        hist_reshaped = np.append(histR_reshaped,
                                  [histG_reshaped, histB_reshaped])
        feature = np.dot(hist_reshaped - self.encoderBias.transpose(),
                         self.encoderWeights)
        return feature

    def rgbuv_hist(self, image):
        """Computes an RGB-uv histogram tensor."""
        sz = np.shape(image)  # Get size of current image.
        if (sz[0] * sz[1] > 202500):  # Resize if it is larger than 450*450.
            factor = np.sqrt(202500 / (sz[0] * sz[1]))  # Rescale factor.
            newH = int(np.floor(sz[0] * factor))
            newW = int(np.floor(sz[1] * factor))
            image = imresize(image, output_shape=(newW, newH))
        I_reshaped = image[(image > 0).all(axis=2)]
        eps = 6.4 / self.h
        A = np.arange(-3.2, 3.19, eps)  # Dummy vector.
        hist = np.zeros((A.size, A.size, 3))  # Histogram will be stored here.
        Iy = np.linalg.norm(I_reshaped, axis=1)  # Intensity vector.
        for i in range(3):  # For each histogram layer, do:
            r = []  # excluded channels will be stored here
            for j in range(3):  # For each color channel, do:
                if j != i:  # if current channel does not match current layer,
                    r.append(j)  # exclude it.
            Iu = np.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
            Iv = np.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
            hist[:, :, i], _, _ = np.histogram2d(
                Iu,
                Iv,
                bins=self.h,
                range=((-3.2 - eps / 2, 3.2 - eps / 2),) * 2,
                weights=Iy)
            norm_ = hist[:, :, i].sum()
            hist[:, :, i] = np.sqrt(hist[:, :, i] / norm_)  # (hist/norm)^(1/2)
        return hist

    def generate_white_balance_augmentation(self, image):
        """Generates a random, augmented version of a given image."""
        image = to_numpy(image)  # Convert to double.
        feature = self.encode(self.rgbuv_hist(image))
        wb_pf = rnd.sample(self.wb_photo_finishing, 1)
        ind = self.wb_photo_finishing.index(wb_pf[0])

        D_sq = np.einsum('ij, ij ->i', self.features,
                         self.features)[:, None] + np.einsum(
                             'ij, ij ->i', feature,
                             feature) - 2 * self.features.dot(feature.T)

        # Get smallest K distances.
        idH = D_sq.argpartition(self.K, axis=0)[:self.K]
        dH = np.sqrt(np.take_along_axis(D_sq, idH, axis=0))
        weightsH = np.exp(-(np.power(dH, 2)) /
                          (2 * np.power(self.sigma, 2)))  # Compute weights.
        weightsH = weightsH / sum(weightsH)  # Normalize blending weights
        # Generate a mapping function
        mf = sum(
            np.reshape(repmat(weightsH, 1, 27), (self.K, 1, 9, 3)) *
            self.mappingFuncs[(idH - 1) * 10 + ind, :])
        mf = mf.reshape(9, 3, order="F")  # Reshape it to be 9 * 3
        synth_WB_image = change_white_balance(image, mf)  # Apply it!
        return synth_WB_image, wb_pf

    def single_image_processing(self, input_image):
        """Applies the WB emulator to a single image `input_image`."""
        # Generate a new image with different WB settings.
        out_img, _ = self.generate_white_balance_augmentation(image=input_image)

        return out_img


def change_white_balance(input, m):
    """Applies a mapping function m to a given input image."""
    sz = np.shape(input)  # Get size of the input image.
    I_reshaped = np.reshape(input, (int(input.size / 3), 3), order="F")
    # Raise input image to a higher-dim space.
    kernel_out = kernelP9(I_reshaped)
    # Apply m to the input image after raising it the selected higher degree
    out = np.dot(kernel_out, m)
    out = outOfGamutClipping(out)  # Clip out-of-gamut pixels.
    # Reshape output image back to the original image shape.
    out = out.reshape(sz[0], sz[1], sz[2], order="F")

    out = (255. * out).astype(np.uint8)

    return out


def kernelP9(image):
    """Kernel function: kernel(r, g, b) -> (r, g, b, r2, g2, b2, rg, rb, gb)"""
    return (np.transpose(
        (image[:, 0], image[:, 1], image[:, 2], image[:, 0] * image[:, 0],
         image[:, 1] * image[:, 1], image[:, 2] * image[:, 2],
         image[:, 0] * image[:, 1], image[:, 0] * image[:, 2],
         image[:, 1] * image[:, 2])))


def outOfGamutClipping(image):
    """Clips out-of-gamut pixels."""
    image[image > 1] = 1  # any pixel is higher than 1, clip it to 1
    image[image < 0] = 0  # any pixel is below 0, clip it to 0
    return image


def to_numpy(im):
    """Returns a double numpy image [0,1] of the uint8 im [0,255]."""
    assert (im.dtype == np.uint8)
    return np.array(im) / 255
