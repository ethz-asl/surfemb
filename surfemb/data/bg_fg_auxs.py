import numpy as np
import skimage
import torch
import torchvision
from torchvision import transforms as tv_transforms

from typing import Tuple


class RandomBackgroundForegroundCreator:

    def __init__(self, H: int, W: int, augmentation_dataset_folder: str,
                 probability_foreground_objects: float):

        # Optionally, overlay an object onto the input image.
        assert (isinstance(probability_foreground_objects, float) and
                0.0 <= probability_foreground_objects <= 1.0)
        self.probability_foreground_objects = probability_foreground_objects

        # On one side, sample images from VOC2012.
        self._dataset = torchvision.datasets.VOCSegmentation(
            root=augmentation_dataset_folder,
            year="2012",
            download=True,
            transform=tv_transforms.Compose(
                [tv_transforms.ToTensor(),
                 tv_transforms.Resize((H, W))]),
            target_transform=tv_transforms.Compose([
                tv_transforms.ToTensor(),
                tv_transforms.Resize(
                    (H, W),
                    interpolation=tv_transforms.InterpolationMode.NEAREST)
            ]))
        self._num_samples_dataset = len(self._dataset)
        # On the other side, use a random-noise-like background.
        self._fake_dataset = torchvision.datasets.FakeData(
            size=10000,
            image_size=(3, H, W),
            transform=tv_transforms.Compose(
                [tv_transforms.ToTensor(),
                 tv_transforms.Resize((H, W))]))
        self._num_samples_fake_dataset = len(self._fake_dataset)

    def transform_image_bg(self, img: torch.Tensor) -> torch.Tensor:
        assert (len(img.shape) == 3 and img.shape[-1] == 4 and
                img[..., 3].max() <= 1.)
        assert (set(torch.unique(img[..., 3]).tolist()).issubset([0., 0.5, 1.]))
        mask = img[..., 3] != 0.
        img = img[..., :3]
        # Select with equal probability a random background from VOC 2012 or
        # random-noise-like background.
        seed = torch.rand(1).item()

        should_select_voc_background = (seed < 1 / 2)
        should_select_random_noise_background = (seed >= 1 / 2)
        assert (sum([
            should_select_voc_background, should_select_random_noise_background
        ]) == 1)

        if (should_select_voc_background):
            output_sample = self._dataset.__getitem__(
                torch.randint(self._num_samples_dataset,
                              (1,)).item())[0].permute(1, 2,
                                                       0).to(device=img.device)
        elif (should_select_random_noise_background):
            output_sample = self._fake_dataset.__getitem__(
                torch.randint(self._num_samples_fake_dataset,
                              (1,)).item())[0].permute(1, 2,
                                                       0).to(device=img.device)
        if (output_sample.shape[0] != img.shape[0] or
                output_sample.shape[1] != img.shape[1]):
            # If the current image is a crop (i.e., it does not have size
            # (H, W)), randomly select a crop of the random background that fits
            # the size of the current image.
            start_H_crop = torch.randint(output_sample.shape[0] - img.shape[0],
                                         (1,)).item()
            start_W_crop = torch.randint(output_sample.shape[1] - img.shape[1],
                                         (1,)).item()
            output_sample = output_sample[start_H_crop:start_H_crop +
                                          img.shape[0],
                                          start_W_crop:start_W_crop +
                                          img.shape[1]]

        output_sample[mask] = img[mask]

        return output_sample

    def transform_image_fg(
            self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """NOTE: The foreground will be given alpha channel 0.5, to distinguish
        it from the object.
        """
        seed = torch.rand(1).item()

        assert (torch.is_tensor(img) and img.ndim == 3 and img.shape[-1] == 4)
        assert (img.dtype == torch.float32 and
                set(torch.unique(img[..., 3]).tolist()).issubset([0., 1.]))
        H, W = img.shape[:2]

        if (seed < 1. - self.probability_foreground_objects):
            # Do not apply any transformation.
            return img

        # Require a minimum size of the foreground object to partially overlay.
        min_size_fg = 50
        H_fg = -1
        W_fg = -1
        while (H_fg < min_size_fg or W_fg < min_size_fg):
            # Sample an image from the VOC dataset.
            output_sample, output_label = self._dataset.__getitem__(
                torch.randint(self._num_samples_dataset, (1,)).item())
            output_sample = output_sample.permute(1, 2, 0).to(device=img.device)
            output_label = output_label.permute(1, 2, 0).to(device=img.device)

            # Select an object label.
            object_labels = torch.unique(output_label).tolist()
            try:
                object_labels.remove(0.0)
            except ValueError:
                pass
            try:
                object_labels.remove(1.0)
            except ValueError:
                pass
            assert (len(object_labels) > 0)
            object_label = object_labels[np.random.randint(len(object_labels))]

            # Find the parts of the image that contains the given object label
            # and extract one as foreground.
            connected_components = skimage.measure.label(
                (output_label == object_label).cpu().numpy(), background=0)
            connected_components_labels = np.unique(
                connected_components).tolist()
            try:
                connected_components_labels.remove(0)
            except ValueError:
                pass
            connected_components_label = connected_components_labels[
                np.random.randint(len(connected_components_labels))]

            # Extract a crop of foreground object and overlay it to the input
            # image, ensuring that at least 30% of the input image is still
            # visible.
            y_where_label, x_where_label = np.where(
                connected_components[..., 0] == connected_components_label)
            x_where_label_max = x_where_label.max()
            x_where_label_min = x_where_label.min()
            y_where_label_max = y_where_label.max()
            y_where_label_min = y_where_label.min()
            y_where_image, x_where_image = torch.where(img[..., 3] == 1.)
            x_where_image_max = x_where_image.max().item()
            x_where_image_min = x_where_image.min().item()
            y_where_image_max = y_where_image.max().item()
            y_where_image_min = y_where_image.min().item()

            H_fg = y_where_label_max - y_where_label_min
            W_fg = x_where_label_max - x_where_label_min

        max_num_pixels_foreground = int(
            0.7 * (x_where_image_max - x_where_image_min) *
            (y_where_image_max - y_where_image_min))
        min_num_pixels_foreground = int(
            0.2 * (x_where_image_max - x_where_image_min) *
            (y_where_image_max - y_where_image_min))

        num_pixels_foreground = np.random.randint(min_num_pixels_foreground,
                                                  max_num_pixels_foreground)
        # - Keep an aspect ratio of at least 1:5 / 5:1 in the cropped
        #   foreground.
        min_rel_ratio_crop = 5.

        fg_image = torch.cat(
            [output_sample,
             torch.ones_like(output_sample[..., 0])[..., None]],
            axis=-1)
        assert (fg_image.dtype == torch.float32)
        fg_image[connected_components[...,
                                      0] != connected_components_label] = 0.
        H_fg_crop = min(
            H_fg, H,
            np.random.randint(
                np.sqrt(num_pixels_foreground / min_rel_ratio_crop),
                np.sqrt(num_pixels_foreground * min_rel_ratio_crop)))
        W_fg_crop = min(
            np.round(num_pixels_foreground / H_fg_crop).astype(int), W, W_fg)

        # - Choose one of eight configurations.
        seed = np.random.rand()
        output_img = img.clone()
        if (seed < 1 / 8.):
            # Top-left overlay.
            x_start_crop = x_where_label_max - W_fg_crop + 1
            x_end_crop = x_where_label_max
            y_start_crop = y_where_label_max - H_fg_crop + 1
            y_end_crop = y_where_label_max
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[:H_fg_crop, :W_fg_crop][
                is_pix_in_object] = cropped_fg_image[is_pix_in_object]
        elif (1 / 8. <= seed < 2 / 8.):
            # Top-right overlay.
            x_start_crop = x_where_label_min
            x_end_crop = x_where_label_min + W_fg_crop - 1
            y_start_crop = y_where_label_max - H_fg_crop + 1
            y_end_crop = y_where_label_max
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[:H_fg_crop, -W_fg_crop:][
                is_pix_in_object] = cropped_fg_image[is_pix_in_object]
        elif (2 / 8. <= seed < 3 / 8.):
            # Bottom-right overlay.
            x_start_crop = x_where_label_min
            x_end_crop = x_where_label_min + W_fg_crop - 1
            y_start_crop = y_where_label_min
            y_end_crop = y_where_label_min + H_fg_crop - 1
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[-H_fg_crop:, -W_fg_crop:][
                is_pix_in_object] = cropped_fg_image[is_pix_in_object]
        elif (3 / 8. <= seed < 4 / 8.):
            # Bottom-left overlay.
            x_start_crop = x_where_label_max - W_fg_crop + 1
            x_end_crop = x_where_label_max
            y_start_crop = y_where_label_min
            y_end_crop = y_where_label_min + H_fg_crop - 1
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[-H_fg_crop:, :W_fg_crop][
                is_pix_in_object] = cropped_fg_image[is_pix_in_object]
        elif (4 / 8. <= seed < 5 / 8.):
            # Top overlay.
            # - Resize foreground image to fit horizontally.
            x_start_crop = x_where_label_min
            x_end_crop = x_where_label_max
            y_start_crop = y_where_label_max - H_fg_crop + 1
            y_end_crop = y_where_label_max
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            cropped_fg_image = tv_transforms.Resize(
                size=(int(cropped_fg_image.shape[0] * W_fg_crop /
                          cropped_fg_image.shape[1]), W_fg_crop),
                interpolation=tv_transforms.InterpolationMode.NEAREST)(
                    cropped_fg_image.permute(2, 0, 1)).permute(1, 2, 0)
            H_fg_crop, W_fg_crop = cropped_fg_image.shape[:2]
            if (W_fg_crop >= W):
                x_start_fg_on_image = W - W_fg_crop
            else:
                # Randomly place the crop horizontally on the image.
                x_start_fg_on_image = np.random.randint(0, W - W_fg_crop + 1)
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[:H_fg_crop, x_start_fg_on_image:x_start_fg_on_image +
                       W_fg_crop][is_pix_in_object] = cropped_fg_image[
                           is_pix_in_object]
        elif (5 / 8. <= seed < 6 / 8.):
            # Right overlay.
            # - Resize foreground image to fit vertically.
            x_start_crop = x_where_label_min
            x_end_crop = x_where_label_min + W_fg_crop - 1
            y_start_crop = y_where_label_min
            y_end_crop = y_where_label_max
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            cropped_fg_image = tv_transforms.Resize(
                size=(H_fg_crop,
                      int(cropped_fg_image.shape[1] * H_fg_crop /
                          cropped_fg_image.shape[0])),
                interpolation=tv_transforms.InterpolationMode.NEAREST)(
                    cropped_fg_image.permute(2, 0, 1)).permute(1, 2, 0)
            H_fg_crop, W_fg_crop = cropped_fg_image.shape[:2]
            if (H_fg_crop >= H):
                y_start_fg_on_image = H - H_fg_crop
            else:
                # Randomly place the crop vertically on the image.
                y_start_fg_on_image = np.random.randint(0, H - H_fg_crop + 1)
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[y_start_fg_on_image:y_start_fg_on_image + H_fg_crop,
                       -W_fg_crop:][is_pix_in_object] = cropped_fg_image[
                           is_pix_in_object]
        elif (6 / 8. <= seed < 7 / 8.):
            # Bottom overlay.
            # - Resize foreground image to fit horizontally.
            x_start_crop = x_where_label_min
            x_end_crop = x_where_label_max
            y_start_crop = y_where_label_min
            y_end_crop = y_where_label_min + H_fg_crop - 1
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            cropped_fg_image = tv_transforms.Resize(
                size=(int(cropped_fg_image.shape[0] * W_fg_crop /
                          cropped_fg_image.shape[1]), W_fg_crop),
                interpolation=tv_transforms.InterpolationMode.NEAREST)(
                    cropped_fg_image.permute(2, 0, 1)).permute(1, 2, 0)
            H_fg_crop, W_fg_crop = cropped_fg_image.shape[:2]
            if (W_fg_crop >= W):
                x_start_fg_on_image = W - W_fg_crop
            else:
                # Randomly place the crop horizontally on the image.
                x_start_fg_on_image = np.random.randint(0, W - W_fg_crop + 1)
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[-H_fg_crop:, x_start_fg_on_image:x_start_fg_on_image +
                       W_fg_crop][is_pix_in_object] = cropped_fg_image[
                           is_pix_in_object]
        else:
            # Left overlay.
            # - Resize foreground image to fit vertically.
            x_start_crop = x_where_label_max - W_fg_crop + 1
            x_end_crop = x_where_label_max
            y_start_crop = y_where_label_min
            y_end_crop = y_where_label_max
            cropped_fg_image = fg_image[y_start_crop:y_end_crop + 1,
                                        x_start_crop:x_end_crop + 1]
            cropped_fg_image = tv_transforms.Resize(
                size=(H_fg_crop,
                      int(cropped_fg_image.shape[1] * H_fg_crop /
                          cropped_fg_image.shape[0])),
                interpolation=tv_transforms.InterpolationMode.NEAREST)(
                    cropped_fg_image.permute(2, 0, 1)).permute(1, 2, 0)
            H_fg_crop, W_fg_crop = cropped_fg_image.shape[:2]
            if (H_fg_crop >= H):
                y_start_fg_on_image = H - H_fg_crop
            else:
                # Randomly place the crop vertically on the image.
                y_start_fg_on_image = np.random.randint(0, H - H_fg_crop + 1)
            assert (set(torch.unique(
                cropped_fg_image[..., 3]).tolist()).issubset([0., 1.]))
            is_pix_in_object = cropped_fg_image[..., 3] == 1.
            y_where_pix_in_obj, x_where_pix_in_obj = torch.where(
                is_pix_in_object)
            cropped_fg_image[y_where_pix_in_obj, x_where_pix_in_obj, 3] = 0.5
            output_img[
                y_start_fg_on_image:y_start_fg_on_image +
                H_fg_crop, :W_fg_crop][is_pix_in_object] = cropped_fg_image[
                    is_pix_in_object]

        return output_img