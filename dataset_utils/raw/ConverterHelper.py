from copy import deepcopy
from typing import Tuple, List

import numpy as np
from PIL import Image
from imageio.plugins.pillow import ndarray_to_pil


class ConverterHelper:

    @staticmethod
    def cut_relevant_area(
            image: Image.Image,
            annotations: dict,
            padding=5
    ) -> Tuple[Image.Image, dict]:
        """
        Removes not relevant content from image with thyroid nodules (black borders with some data from US device).
        @param image: Input image.
        @param annotations: Dict with annotations.
        @param padding: Additional area around image to be removed.
        @return: Processed image.
        """
        width, height = image.size

        img = np.array(image)

        vertical_avg = np.average(img, axis=0)
        horizontal_avg = np.average(img, axis=1)

        def reduce(red_func, indices, default_value):
            if len(indices[0]) == 0:
                return default_value
            return red_func(indices)

        top = reduce(np.max, np.where(horizontal_avg[:height // 2] < 10), 0) + padding
        bottom = reduce(np.min, np.where(horizontal_avg[height // 2:] < 10), height // 2) + height // 2 - padding
        right_boundary = width // 4
        left = reduce(np.max, np.where(vertical_avg[:right_boundary] < 10), 0) + padding
        left_boundary = width // 4 * 3
        right = reduce(np.min, np.where(vertical_avg[left_boundary:] < 10), left_boundary) + left_boundary - padding

        new_image = img[top:bottom, left:right]
        new_annotations = deepcopy(annotations)

        for region in new_annotations['regions']:
            # shift area with respect to image cut
            region[..., 0] -= top
            region[..., 1] -= left

        return ndarray_to_pil(new_image), new_annotations

    @staticmethod
    def divide_doubled_image(
            img: np.ndarray,
            regions: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Splits cases which has doubled image inside.
        If there is only one image inside, returns given input since there is nothing to do.
        @param img: Image to be split up.
        @param regions: Corresponding annotations in region form.
        @return: Tuple with separated images with converted bboxes.
        """
        uint_img = img
        img = np.array(img, dtype=np.int)

        img_width = img.shape[1]
        vertical_diffs = []

        # iterate over all vertical lines and compute diff values between adjacent lines
        for i in range(1, img_width):
            diff = np.average(np.abs(img[:, i - 1] - img[:, i]))
            vertical_diffs.append(diff)
        vertical_diffs = np.array(vertical_diffs)
        # find average diff value
        avg_diff = np.average(vertical_diffs)
        # find maximum diff value (most likely it is a line between two images)
        max_diff_index = np.argmax(vertical_diffs)

        # if line appears in the center of x axis and its diff is greater than (2.5 * average diff) then it is line between images
        # if one of above conditions is not fulfilled then given image does not contain two images and does not have to be divided
        if vertical_diffs[max_diff_index] > 2.4 * avg_diff and img_width / 10 * 4 < max_diff_index < img_width / 10 * 6:
            second_image_start = max_diff_index + 1
            first_image: np.ndarray = uint_img[:, :second_image_start]
            second_image: np.ndarray = uint_img[:, second_image_start:]
            first_regions = []
            second_regions = []

            # divide regions between new images and shift points to left in second image
            for region in regions:
                if region[0, 1] < second_image_start:
                    first_regions.append(region)
                else:
                    region[:, 1] -= second_image_start
                    second_regions.append(region)

            return [(first_image, first_regions), (second_image, second_regions)]

        return [(uint_img, regions)]

    @staticmethod
    def extract_bboxes_single(
            regions: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Extracts bboxes array from single dict with annotations.
        @param regions: Regions assigned to images.
        @return: [n_bboxes][center_y, center_x, height, width]
        """
        bboxes = []
        for region in regions:
            region = np.array(region)
            min_y, min_x = np.min(region, axis=0)
            max_y, max_x = np.max(region, axis=0)
            bbox = [
                (min_y + max_y) // 2, (min_x + max_x) // 2,
                (max_y - min_y), (max_x - min_x)
            ]
            bboxes.append(np.array(bbox))

        return bboxes

    @staticmethod
    def random_crop_nodule(
            image: np.ndarray,
            regions: List[np.ndarray],
            cls: str,
            crops_per_region: int,
            crop_min_size: int
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[str]]:
        """
        Crops nodules from image, marked by regions.
        @param image: Image.
        @param regions: Corresponding annotations in region form.
        @param cls: Class assigned to image.
        @param crops_per_region: Amount of generated images per one region.
        @param crop_min_size: Threshold when cropping small nodules.
        @return: Cropped images with empty regions.
        """

        cropped_images = []

        img_h, img_w = image.shape

        bboxes = ConverterHelper.extract_bboxes_single(regions)
        for bbox in bboxes:
            bbox_yx, bbox_hw = bbox[:2], bbox[2:]

            crop_yx = bbox_yx.copy()
            crop_hw = bbox_hw * 1.3
            min_size = np.array([crop_min_size] * 2)
            crop_hw = np.maximum(crop_hw, min_size)

            base_crop_bbox = np.array((*crop_yx, *crop_hw))
            for _ in range(crops_per_region):
                crop_bbox = base_crop_bbox.copy()
                crop_yx, crop_hw = crop_bbox[:2], crop_bbox[2:]
                # generate randomly scaled and shifted crop boxes, range is +/-0.15 of w and h
                crop_yx += crop_hw * np.random.uniform(-0.10, 0.10, size=2)
                crop_hw += crop_hw * np.random.uniform(-0.05, 0.05, size=2)

                c_top, c_bottom, c_left, c_right = ConverterHelper.convert_bbox(crop_bbox).astype(np.int32)
                c_top, c_bottom, c_left, c_right = (
                    max(0, c_top), min(img_h - 1, c_bottom),
                    max(0, c_left), min(img_w - 1, c_right)
                )
                cropped_images.append(image[c_top:c_bottom, c_left:c_right])

        return cropped_images, [[] for _ in range(len(cropped_images))], [cls] * len(cropped_images)

    @staticmethod
    def random_crop_background(
            image: np.ndarray,
            regions: List[np.ndarray],
            crops: int,
            crop_size: int
    ) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[str]]:
        """

        @param image:
        @param regions:
        @param crops:
        @param crop_size:
        @return:
        """
        found_images = []

        img_h, img_w = image.shape

        gt_bboxes = ConverterHelper.extract_bboxes_single(regions)
        
        half_crop_size = crop_size // 2
        for _ in range(crops):
            for __ in range(100):
                rnd_yx = np.array([np.random.randint(half_crop_size, img_h - 1 - half_crop_size),
                                   np.random.randint(half_crop_size, img_w - 1 - half_crop_size)])
                found_bg_box = [*rnd_yx, crop_size, crop_size]
                top, bottom, left, right = ConverterHelper.convert_bbox(found_bg_box)
                for gt_bbox in gt_bboxes:
                    top_gt, bottom_gt, left_gt, right_gt = ConverterHelper.convert_bbox(gt_bbox)
                    if ConverterHelper.intersection(
                            (left, top, right, bottom), (left_gt, top_gt, right_gt, bottom_gt)
                    ) > 0:
                        break
                else:
                    found_images.append(image[top:bottom, left:right])
                    break

        return found_images, [[] for _ in range(len(found_images))], ['bg'] * len(found_images)

    @staticmethod
    def convert_bbox(bbox) -> np.ndarray:
        bbox_y, bbox_x, bbox_h, bbox_w = bbox
        top, bottom, left, right = (bbox_y - bbox_h // 2, bbox_y + bbox_h // 2,
                                    bbox_x - bbox_w // 2, bbox_x + bbox_w // 2)
        return np.array([top, bottom, left, right])

    @staticmethod
    def intersection(ai, bi):
        """
        @param ai: (x1,y1,x2,y2)
        @param bi: (x1,y1,x2,y2)
        @return:
        """
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
            return 0
        return w * h
