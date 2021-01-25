class NoduleDatasetPaths:

    def __init__(self):
        # nodule dataset paths
        self.dataset_root = 'nodule_dataset'
        self.generated_root = f'{self.dataset_root}/generated'

        self.raw_images = f'{self.dataset_root}/images/valid'
        self.raw_annotations = f'{self.dataset_root}/gt'
