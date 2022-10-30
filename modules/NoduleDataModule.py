from os.path import join
import os
from pathlib import Path
from typing import Union, List, Optional
import shutil

from monai.transforms import LoadImage, PILReader, ToTensor, RandRotate, ScaleIntensity, RandZoom, RandFlip, AddChannel, \
    RepeatChannel, SpatialPad
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset_utils.generated.NoduleDatasetSplitter import NoduleDatasetSplitter
from modules.NoduleDataset import NoduleDataset, MetaToLabels, MetaToPath, CustomResize


class NoduleDataModule(LightningDataModule):
    NUM_WORKERS = 10

    CLS_MAPPINGS = [
        # (['2', '3', '4a'], 'benign'),
        # (['4b', '4c', '5'], 'malignant')
        (['2', '3', '4a', '4b', '4c', '5'], 'nodule'),
        (['bg'], 'bg')
    ]

    def __init__(
            self,
            dataset_path: str,
            batch_size=16,
            image_size=160,
            val_split=0.1,
            test_split=0.2,
            copy_to_scratch=False
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.val_split = val_split
        self.test_split = test_split
        self.copy_to_scratch = copy_to_scratch

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        if self.copy_to_scratch:
            print('Copying dataset to scratch.')
            scratch_path = Path(os.environ['SCRATCH_LOCAL'])
            project_path = scratch_path / 'mgrImpl_torch'
            project_path.mkdir()
            dataset_path = project_path / 'nodule_dataset'
            shutil.copytree('nodule_dataset', str(dataset_path))
            print('Dataset copied.')
            self.dataset_path = project_path / self.dataset_path

    def setup(self, stage: Optional[str] = None):
        splitter = NoduleDatasetSplitter(
            self.dataset_path,
            val_split=self.val_split,
            test_split=self.test_split
        )
        train_metas, val_metas, test_metas = splitter.split_distributed_cases()

        train_img_transforms = Compose([
            MetaToPath(join(self.dataset_path, 'images')),
            LoadImage(reader=PILReader(), image_only=True),
            AddChannel(),
            CustomResize(self.image_size),
            SpatialPad([self.image_size] * 2),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ScaleIntensity(),
            RepeatChannel(repeats=3),
            ToTensor()
        ])
        val_img_transforms = Compose([
            MetaToPath(join(self.dataset_path, 'images')),
            LoadImage(reader=PILReader(), image_only=True),
            AddChannel(),
            CustomResize(self.image_size),
            SpatialPad([self.image_size] * 2),
            ScaleIntensity(),
            RepeatChannel(repeats=3),
            ToTensor()
        ])
        label_transforms = Compose([
            MetaToLabels(mappings=self.CLS_MAPPINGS)
        ])

        self.train_ds = NoduleDataset(train_metas, train_img_transforms, label_transforms)
        self.val_ds = NoduleDataset(val_metas, val_img_transforms, label_transforms)
        self.test_ds = NoduleDataset(test_metas, val_img_transforms, label_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.NUM_WORKERS, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.NUM_WORKERS)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.NUM_WORKERS)
