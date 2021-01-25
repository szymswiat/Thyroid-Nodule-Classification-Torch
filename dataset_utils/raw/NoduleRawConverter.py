import os
import pickle
from typing import Tuple, List, Callable
from pathlib import Path
import numpy as np
from imageio.plugins.pillow import ndarray_to_pil
from progressbar import *

from dataset_utils.NoduleDataVisualizer import NoduleDataVisualizer
from dataset_utils.generated.DatasetMeta import DatasetMeta, ImageMeta, ImageCase

from dataset_utils.raw.ConverterHelper import ConverterHelper
from dataset_utils.raw.NoduleDataUtil import NoduleDataUtil
from dataset_utils.raw.NoduleRawGenerator import NoduleRawGenerator


class NoduleRawConverter:

    def __init__(
            self,
            raw_generator: NoduleRawGenerator,
            output_dir: str,
    ):
        self.raw_generator = raw_generator
        self.helper = ConverterHelper()

        self.output_dir = output_dir

    def __len__(self):
        return len(self.raw_generator)

    def get_converted_image_with_annotations(self, index) -> Tuple[np.ndarray, List[np.ndarray], str]:
        """
        Returns image with regions and class.
        Class can contain None value when annotation from generated does not contain tirads label.
        @param index: Index of image.
        @return: Image with bboxes and assigned class.
        """
        img, annotations = self.raw_generator[index]
        relevant_img, annotations = self.helper.cut_relevant_area(img, annotations, padding=3)
        cls: str = annotations['tirads']
        regions = annotations['regions']
        return np.array(relevant_img), regions, cls

    def _convert_and_save(
            self,
            cut_image_proc_func: Callable
    ):
        """
        Generated generated images to directory specified in (paths.generated)
        @return: None
        """
        images_out_dir = os.path.join(self.output_dir, 'images')
        Path(images_out_dir).mkdir(parents=True, exist_ok=True)

        widgets = [Bar('#'), ' ', Percentage(), ' ', FormatLabel('')]
        pb = ProgressBar(max_value=len(self.raw_generator), widgets=widgets)

        dataset_meta = DatasetMeta()
        for i in range(len(self.raw_generator)):
            orig_image_name = self.raw_generator.get_image_name(i).split('.')[0]
            widgets[-1] = FormatLabel(f'{orig_image_name:20}')
            pb.update(i)

            image, regions, cls = self.get_converted_image_with_annotations(i)

            # if class is undefined set it as 'u'
            cls = 'u' if cls is None else cls

            divided_images = self.helper.divide_doubled_image(image, regions)
            for j, (image, regions) in enumerate(divided_images):

                regions = NoduleDataUtil.cut_regions_to_image(image, regions)

                images, regions, classes = cut_image_proc_func(image, regions, cls)
                case_name = f'{orig_image_name}_{j}'

                image_case = ImageCase(case_name, cls)
                dataset_meta.add_image_case(image_case)
                for k, (sub_image, sub_regions, sub_cls) in enumerate(zip(images, regions, classes)):
                    sub_image_name = f'{case_name}_{k}'

                    image_meta = ImageMeta(sub_image_name, sub_cls, sub_regions)
                    image_case.add_sub_image(image_meta)
                    ndarray_to_pil(sub_image).save(os.path.join(images_out_dir, f'{sub_image_name}.jpg'))

        print(dataset_meta.images_per_class)
        with open(os.path.join(self.output_dir, 'dataset_meta'), 'wb') as meta_file:
            pickle.dump(dataset_meta, meta_file)

    def convert_and_save_cropped(
            self,
            crops_per_region: int,
            bg_crops: int,
            crop_min_size: int,
            random_seed: int,
            annotate_images: bool = False
    ):
        visualizer = NoduleDataVisualizer()
        np.random.seed(random_seed)

        def process(
                image: np.ndarray,
                regions: List[np.ndarray],
                cls: str
        ):
            if annotate_images:
                image = np.array(visualizer.annotate_image_raw(ndarray_to_pil(image), regions, cls), dtype=np.uint8)
            cropped_nodules = self.helper.random_crop_nodule(image, regions, cls, crops_per_region, crop_min_size)
            cropped_bg = self.helper.random_crop_background(image, regions, bg_crops, crop_min_size)

            return [_[0] + _[1] for _ in zip(cropped_nodules, cropped_bg)]

        self._convert_and_save(process)

    def convert_and_save_full(
            self,
            annotate_images: bool = False
    ):
        visualizer = NoduleDataVisualizer()

        def process(
                image: np.ndarray,
                regions: List[np.ndarray],
                cls: str
        ):
            if annotate_images:
                image = np.array(visualizer.annotate_image_raw(ndarray_to_pil(image), regions, cls), dtype=np.uint8)
            return [image], [regions], [cls]

        self._convert_and_save(process)
