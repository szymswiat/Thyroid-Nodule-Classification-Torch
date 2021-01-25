import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw


class NoduleDataVisualizer:

    @staticmethod
    def annotate_image_raw(
            image: Image.Image,
            regions: List[np.ndarray],
            cls: str
    ) -> Image.Image:
        """
        Draws polylines (borders of nodule) on given image based on raw annotations.
        @param image: PIL image to annotate.
        @param regions: Regions assigned to image.
        @param cls: Class assigned to image.
        @return: Annotated PIL image.
        """
        image = image.copy()
        draw = Draw(image)
        for region in regions:
            for i in range(region.shape[0] - 1):
                y1, x1 = region[i]
                y2, x2 = region[i + 1]
                draw.line(((x1, y1), (x2, y2)), fill=int('0xFFFFFF', 16), width=3)
        if cls is not None:
            draw.text((10, image.size[1] - 10), cls, fill=int('0x00AA00', 16))
        return image

    @staticmethod
    def annotate_image_bbox(
            image: Image.Image,
            bboxes: List[Tuple[np.ndarray, float]],
            cls: Tuple[str, float]
    ) -> Image.Image:
        """
        Annotates image with bboxes as a rectangles.
        @param image: Image to be annotated.
        @param bboxes: Bboxes assigned to image.
        @param cls: Class assigned to image.
        @return: Annotated image in PIL object.
        """
        if bboxes is None:
            return image
        image = image.copy()

        draw = Draw(image)
        for i in range(len(bboxes)):
            # center_y, center_x, height, width
            bbox, probability = bboxes[i]
            c_y, c_x, h, w = bbox
            # x0, y0, x1, y1
            draw.rectangle([c_x - w // 2, c_y - h // 2, c_x + w // 2, c_y + h // 2])
            draw.text((c_x, c_y - h // 2 - 10), str(probability), fill=int('0x00AA00', 16))

        cls, cls_prob = cls
        draw.text((10, image.size[1] - 10), f'{cls}: {cls_prob}', fill=int('0x00AA00', 16))
        return image

    @staticmethod
    def save_image_with_original_one(
            image: Image.Image,
            orig_image: Image.Image,
            name: str,
            path: str
    ) -> None:
        """
        Takes single image eg. annotated with self.annotate_image along with original one and saves both as one image.
        @param image: Annotated PIL image (has to be smaller or of equal size compared to original).
        @param orig_image: Original image.
        @param name: Name of image.
        @param path: Path to save new image.
        @return: None
        """
        size = orig_image.size
        merged_image = Image.new('RGB', (size[0] * 2, size[1]))
        merged_image.paste(image)
        merged_image.paste(orig_image, box=(size[0], 0))

        merged_image.save(os.path.join(path, name))
