import sys

from argparse import ArgumentParser
from os.path import join

from pytorch_lightning import Trainer
from test_tube import SlurmCluster, HyperOptArgumentParser

from common_utils.ArgLauncher import ArgLauncher
from dataset_utils.NoduleDatasetPaths import NoduleDatasetPaths
from modules.ClsModule import ClsModule
from modules.NoduleDataModule import NoduleDataModule


class SlurmLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        pass
        parser.add_argument('-g', '--gpus_per_exp', type=int, default=2, help='Count of GPU used for one experiment.')
        parser.add_argument('-n', '--nodes_per_exp', type=int, default=2, help='Count of nodes assigned to task.')

    def execute(self, args) -> None:
        # create cluster object
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='./slurm_logs'
        )

        # configure cluster
        cluster.per_experiment_nb_nodes = 1
        cluster.per_experiment_nb_gpus = args.gpus_per_exp
        cluster.memory_mb_per_node = 16000  # 8GB mem per node
        cluster.per_experiment_nb_cpus = 8  # 8 CPU cores

        cluster.add_slurm_cmd(cmd='ntasks-per-node', value=args.gpus_per_exp, comment='1 task per gpu')
        cluster.add_slurm_cmd(cmd='constraint', value='localfs', comment='Enable local filesystem')
        cluster.add_slurm_cmd(cmd='partition', value='plgrid-gpu', comment='Use partition dedicated for GPUs.')

        # cluster.add_command('conda activate xrays')

        # submit a script
        cluster.optimize_parallel_cluster_gpu(
            self.slurm_task,
            nb_trials=1,
            job_name='nodule_cls_training'
        )

    @staticmethod
    def slurm_task(h_params, cluster: SlurmCluster):
        gen_name = 'test'
        print('slurm task launched')
        paths = NoduleDatasetPaths()
        dataset_out_path = join(paths.generated_root, gen_name)

        cls_dm = NoduleDataModule(dataset_out_path, copy_to_scratch=True)

        cls_model = ClsModule(num_classes=len(NoduleDataModule.CLS_MAPPINGS))

        trainer = Trainer(
            max_epochs=3,
            accelerator='ddp',
            gpus=2
        )
        print('starting training')
        trainer.fit(cls_model, datamodule=cls_dm)
        print('training completed')


if __name__ == '__main__':
    SlurmLauncher(sys.argv[1:], parser=HyperOptArgumentParser()).launch()
