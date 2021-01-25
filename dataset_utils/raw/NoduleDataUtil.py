from typing import List

import numpy as np


class NoduleDataUtil:

    @staticmethod
    def validate_regions(
            image: np.ndarray,
            regions: List[np.ndarray],
            pad: int = 0
    ) -> bool:
        height, width = image.shape

        for region in regions:
            invalid_y_points = np.logical_or(region[..., 0] < 0 - pad, region[..., 0] >= height + pad)
            invalid_x_points = np.logical_or(region[..., 1] < 0 - pad, region[..., 1] >= width + pad)
            if np.count_nonzero(invalid_x_points) > 0 or np.count_nonzero(invalid_y_points) > 0:
                return False

        return True

    @staticmethod
    def cut_regions_to_image(
            image: np.ndarray,
            regions: List[np.ndarray]
    ) -> List[np.ndarray]:

        height, width = image.shape

        new_regions = []
        for region in regions:
            new_region = region.copy()
            min_region = np.zeros(shape=new_region.shape)
            max_region = np.zeros(shape=new_region.shape)
            max_region[..., 0] = height - 1
            max_region[..., 1] = width - 1

            new_region[..., 0] = np.maximum(new_region[..., 0], min_region[..., 0])
            new_region[..., 0] = np.minimum(new_region[..., 0], max_region[..., 0])

            new_region[..., 1] = np.maximum(new_region[..., 1], min_region[..., 1])
            new_region[..., 1] = np.minimum(new_region[..., 1], max_region[..., 1])

            new_regions.append(new_region)

        return new_regions
