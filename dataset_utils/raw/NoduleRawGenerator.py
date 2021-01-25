import json
import os
import re
from typing import List
from xml.etree import ElementTree

import numpy as np
from PIL import Image


class NoduleRawGenerator:
    """
    Generator of raw data with images and annotations for generated with thyroid nodules.
    """

    def __init__(
            self,
            raw_images_path: str,
            raw_annotations_path: str
    ):

        image_names = sorted(os.listdir(raw_images_path), key=lambda name: int(name.split('_')[0]))
        raw_gt_names = os.listdir(raw_annotations_path)
        gt_paths = list(map(lambda name: (name.split('.')[0], os.path.join(raw_annotations_path, name)), raw_gt_names))

        self.all_gt_paths = dict(gt_paths)
        self.all_image_paths = list(map(lambda name: os.path.join(raw_images_path, name), image_names))

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        return self.load_image(index), self.load_annotations(index)

    def get_image_path(self, index) -> str:
        return self.all_image_paths[index]

    def get_image_name(self, index) -> str:
        return os.path.split(self.get_image_path(index))[-1]

    def load_image(self, index):
        return Image.open(self.all_image_paths[index]).convert(mode='L')

    def load_annotations(self, index) -> dict:
        """
        Loads annotations to dict based on index of image.
        Extracts annotation filename and image number from image name which is loaded by given index.
        @param index: Index of image
        @return: Dict with loaded annotations
        Structure of output dict:
            'tirads' - str with tirads class
            'regions' - List[List[np.array['x', 'y']]] eg. [[(x, y), (x, y)], [(x, y)]]
        """
        image_name: str = os.path.split(self.all_image_paths[index])[-1]
        image_number, image_sub_index = re.findall(r'(\d+)_(\d+)\.jpg', image_name)[0]
        gt_file = open(self.all_gt_paths[image_number], 'r')

        tree = ElementTree.parse(gt_file)
        root = tree.getroot()

        annotations = {
            'tirads': root.find('tirads').text,
            'regions': []
        }
        for mark in root.findall('mark'):
            image = mark.find('image')
            if image.text == image_sub_index:
                svg_text = mark.find('svg').text
                if svg_text is None:
                    break
                svg = json.loads(svg_text)
                for svg_annotation in svg:
                    points = list(map(lambda pts: np.array((pts['y'], pts['x'])), svg_annotation['points']))
                    annotations['regions'].append(np.array(points))
                break
        else:
            raise Exception()

        return annotations

    def get_classes_list(self) -> List[str]:
        """
        Loads all classes from nodule generated annotations.
        @return: List of classes.
        """
        classes = []
        for i in range(len(self)):
            annotations = self.load_annotations(i)
            cls = annotations['tirads']
            if cls not in classes and cls is not None:
                classes.append(cls)
        classes = sorted(classes)
        return classes
