from os.path import join
from pytorch_lightning import Trainer

from dataset_utils.NoduleDatasetPaths import NoduleDatasetPaths
from modules.ClsModule import ClsModule
from modules.NoduleDataModule import NoduleDataModule

if __name__ == '__main__':
    gen_name = 'test'

    paths = NoduleDatasetPaths()
    dataset_out_path = join(paths.generated_root, gen_name)

    cls_dm = NoduleDataModule(dataset_out_path)

    cls_model = ClsModule(num_classes=len(NoduleDataModule.CLS_MAPPINGS))

    trainer = Trainer(
        max_epochs=4
    )

    trainer.fit(cls_model, datamodule=cls_dm)

    # trainer.test()
