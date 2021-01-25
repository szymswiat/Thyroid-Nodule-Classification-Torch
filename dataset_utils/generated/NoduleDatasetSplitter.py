import os
import pickle
from typing import List, Tuple
from copy import deepcopy
import numpy as np

from dataset_utils.generated.DatasetMeta import DatasetMeta, ImageMeta, ImageCase
from sklearn.model_selection import KFold

from torch.utils.data import random_split


class NoduleDatasetSplitter:
    """
    This class implements data loading from h5py generated and allows to split it into three parts
    for training, validation and testing.
    """

    def __init__(
            self,
            dataset_path: str,
            seed: int = 0,
            val_split=0.1,
            test_split=0.2
    ):
        self.dataset_path = dataset_path
        self.seed = seed
        self.val_split = val_split
        self.test_split = test_split

        dataset_meta_path = os.path.join(dataset_path, 'dataset_meta')

        with open(dataset_meta_path, 'rb') as meta_file:
            self._dataset_meta: DatasetMeta = pickle.load(meta_file)

        self.class_per_subset = {}

    def split_distributed_cases(self) -> List[List[ImageMeta]]:
        # indices for self.cases list
        train_im = []
        val_im = []
        test_im = []

        np.random.seed(self.seed)

        for cls, image_cases in self._dataset_meta.by_classes.items():

            random_indices = np.random.permutation(len(image_cases))

            val_count = int(self.val_split * len(image_cases))
            test_count = int(self.test_split * len(image_cases))
            train_count = len(image_cases) - val_count - test_count

            for random_indicator in random_indices[:train_count]:
                train_im = train_im + image_cases[random_indicator].sub_images
            random_indices = random_indices[train_count:]
            for random_indicator in random_indices[:val_count]:
                val_im = val_im + image_cases[random_indicator].sub_images
            random_indices = random_indices[val_count:]
            test_count = test_count if test_count < len(random_indices) else len(random_indices)
            for random_indicator in random_indices[:test_count]:
                test_im = test_im + image_cases[random_indicator].sub_images

        return [sorted(im, key=lambda item: item.name) for im in [train_im, val_im, test_im]]

    def k_fold_generators(
            self,
            groups: List[List[str]],
            folds: int = 10
    ) -> Tuple[List[List[ImageMeta]], List[List[ImageMeta]]]:
        """

        @param groups:
        @param folds:
        @return:
        """
        grouped_cases = self.grouped_cases(groups)

        train_folds, test_folds = [[] for _ in range(folds)], [[] for _ in range(folds)]
        fold_splitter = KFold(n_splits=folds, shuffle=True, random_state=self.seed)
        for case_group in grouped_cases:
            for i, (train_fold, test_fold) in enumerate(fold_splitter.split(case_group)):
                train_folds[i] = train_folds[i] + [case_group[k] for k in [*train_fold]]
                test_folds[i] = test_folds[i] + [case_group[k] for k in [*test_fold]]
        train_gens = [self.extract_metas(fold) for fold in train_folds]
        test_gens = [self.extract_metas(fold) for fold in test_folds]

        return train_gens, test_gens

    @staticmethod
    def extract_metas(cases: List[ImageCase]) -> List[ImageMeta]:
        metas = []
        for case in cases:
            metas = metas + case.sub_images
        return metas

    def grouped_cases(
            self,
            groups: List[List[str]]
    ):
        grouped_cases = [[] for _ in range(len(groups))]
        for cls, image_cases in self._dataset_meta.by_classes.items():
            if len(image_cases) == 0:
                continue
            for i, group in enumerate(groups):
                if cls in group:
                    grouped_cases[i] = grouped_cases[i] + image_cases
                    break
            else:
                raise ValueError()
            pass
        return grouped_cases
