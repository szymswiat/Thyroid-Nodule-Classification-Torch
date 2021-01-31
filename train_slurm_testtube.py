import sys
from os.path import join

from pytorch_lightning import Trainer
from test_tube import SlurmCluster, HyperOptArgumentParser

from common_utils.SlurmLauncher import SlurmLauncher
from dataset_utils.NoduleDatasetPaths import NoduleDatasetPaths
from modules.ClsModule import ClsModule
from modules.NoduleDataModule import NoduleDataModule


class NihClassifierLauncher(SlurmLauncher):

    def setup_hparams(self, parser: HyperOptArgumentParser) -> int:
        return 1

    @staticmethod
    def execute_task(h_params, cluster: SlurmCluster) -> None:
        gen_name = 'test'
        print('slurm task launched')
        paths = NoduleDatasetPaths()
        dataset_out_path = join(paths.generated_root, gen_name)

        cls_dm = NoduleDataModule(dataset_out_path, copy_to_scratch=True)

        cls_model = ClsModule(num_classes=len(NoduleDataModule.CLS_MAPPINGS))

        trainer = Trainer(
            max_epochs=40,
            accelerator='ddp',
            gpus=cluster.per_experiment_nb_gpus,
            num_nodes=cluster.per_experiment_nb_nodes,
            progress_bar_refresh_rate=0,
            prepare_data_per_node=True
        )
        print('starting training')
        trainer.fit(cls_model, datamodule=cls_dm)
        print('training completed')


if __name__ == '__main__':
    NihClassifierLauncher(sys.argv[1:]).launch()
