from __future__ import annotations

from typing import Dict, List


class DatasetMeta:
    def __init__(self):
        super().__init__()

        self.meta_data: List[ImageCase] = []

    def add_image_case(self, image_case: ImageCase):
        self.meta_data.append(image_case)

    @property
    def by_classes(self) -> Dict[str, List[ImageCase]]:
        cls_dict = {}

        for image_case in self.meta_data:
            cls_case = ImageCase(image_case.name, image_case.image_cls)
            bg_case = ImageCase(image_case.name, 'bg')
            for image_meta in image_case.sub_images:
                if image_meta.cls == 'bg':
                    bg_case.add_sub_image(image_meta)
                else:
                    cls_case.add_sub_image(image_meta)
            if cls_case.image_cls not in cls_dict:
                cls_dict[cls_case.image_cls] = []
            if bg_case.image_cls not in cls_dict:
                cls_dict[bg_case.image_cls] = []
            cls_dict[cls_case.image_cls].append(cls_case)
            if len(bg_case.sub_images) > 0:
                cls_dict['bg'].append(bg_case)
        for key in cls_dict.keys():
            cls_dict[key] = sorted(cls_dict[key], key=lambda item: item.name)

        return cls_dict

    @property
    def images_per_class(self):
        cls_dict = self.by_classes
        count_dict = {}
        for key in cls_dict.keys():
            count_dict[key] = 0

        for key, cases in cls_dict.items():
            for case in cases:
                count_dict[key] += len(case.sub_images)

        return count_dict


class ImageCase:
    def __init__(self, name, image_cls):
        self.name = name
        self.image_cls = image_cls
        self.sub_images: List[ImageMeta] = []

    def add_sub_image(self, image_meta: ImageMeta):
        self.sub_images.append(image_meta)

    def __str__(self):
        return self.name


class ImageMeta:
    def __init__(self, name, cls, regions):
        super().__init__()

        self.name = name
        self.cls = cls
        self.regions = regions

    def __str__(self):
        return self.name
