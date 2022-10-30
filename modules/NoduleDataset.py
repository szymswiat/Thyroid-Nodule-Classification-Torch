import os
from typing import List, Tuple, Optional

import numpy as np
from monai.transforms import Transform, Resize
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose

from dataset_utils.generated.DatasetMeta import ImageMeta


class NoduleDataset(Dataset):

    def __init__(
            self,
            metas: List[ImageMeta],
            img_transforms: Compose,
            label_transforms: Compose
    ):
        self.metas = metas
        self.img_transforms = img_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index) -> T_co:
        meta = self.metas[index]
        return self.img_transforms(meta), self.label_transforms(meta)


class MetaToPath(Transform):

    def __init__(self, image_root_dir: str, extension: str = '.jpg'):
        self.image_root_dir = image_root_dir
        self.extension = extension

    def __call__(self, data: ImageMeta) -> str:
        assert isinstance(data, ImageMeta)
        return os.path.join(self.image_root_dir, f'{data.name}{self.extension}')


class MetaToLabels(Transform):

    def __init__(
            self,
            classes: List[str] = None,
            mappings: Optional[List[Tuple[List, str]]] = None
    ):
        """

        :type classes: List of available classes.
        :param mappings: E.g. list of (['old_cls1', 'old_cls2'], 'new_cls0')
        """
        assert not (classes is None and mappings is None)
        assert not (classes is not None and mappings is not None)
        self.classes = classes
        self.mappings = mappings

    def __call__(self, data: ImageMeta) -> int:
        if self.mappings is None:
            return self.classes.index(data.cls)
        else:
            for i, (to_be_mapped, new_cls) in enumerate(self.mappings):
                if data.cls in to_be_mapped:
                    return i


class CustomResize(Transform):

    def __init__(self, longer_edge: int):
        self.longer_edge = longer_edge

    def __call__(self, data: np.ndarray) -> np.ndarray:
        max_edge = max(data.shape[1:])
        scale = self.longer_edge / max_edge
        new_size = int(data.shape[1] * scale), int(data.shape[2] * scale)
        return Resize(spatial_size=new_size)(data)
