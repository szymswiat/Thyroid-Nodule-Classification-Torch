from abc import abstractmethod
from argparse import ArgumentParser

from test_tube import SlurmCluster, HyperOptArgumentParser


class SlurmLauncher:
    AVAILABLE_PARTITIONS = ['plgrid-gpu', 'plgrid-gpu-v100']

    def __init__(self, args):
        self._args = args
        self._parser = HyperOptArgumentParser(strategy='grid_search')

    def setup_cluster_params(self, parser: ArgumentParser) -> None:
        """
        Setups basic options for slurm task launcher.
        :param parser: Argument parser.
        :return: None
        """
        # cluster resources
        # assuming that single experiment is executed on single node
        parser.add_argument('-n', '--nodes_per_exp', type=int, default=1, help='Count of nodes assigned to task.')
        parser.add_argument('-g', '--gpus_per_node', type=int, default=1, help='Count of GPU units per node.')
        parser.add_argument('-m', '--mem_per_node', type=int, default=16, help='Memory per node [GB].')
        parser.add_argument('-c', '--cpus_per_node', type=int, default=8, help='Count of CPU cores per node.')
        parser.add_argument('-p', '--partition', type=str, default='plgrid-gpu',
                            help=f'Slurm partition. Available options {self.AVAILABLE_PARTITIONS}.')
        parser.add_argument('-j', '--job_name', type=str, required=True, help='Name of slurm job.')
        parser.add_argument('-t', '--time', type=str, default='00-00:20:00', help='Walltime.')

    def setup_hparams(self, parser: HyperOptArgumentParser) -> int:
        """
        Override this method to add hyperparams for grid search.
        Every hyperparam name have to start with 'h_'.
        See example below.
        :param parser: Argument parser.
        :return: Returns amount of generated slurm tasks, based on count of hyperparams combinations.
        """
        # parser.opt_list('--h_learning_rate', type=float, tunable=True,
        #                 default=0.001, options=[1e-3, 1e-2, 1e-1])

        # return '1' since there is no hyperparams to tune
        return 1

    def validate_args(self, args):
        """
        Base method which allows subclasses to validate supplied input from CLI.
        @param args: Object with CLI args.
        @return: None
        """
        pass

    def convert_args(self, args):
        """
        Allows user to convert supplied args into different form.
        @param args: Object with CLI args.
        @return: None
        """
        pass

    def configure_cluster(self, args, cluster: SlurmCluster) -> None:
        """
        Configures slurm cluster resources allocation.
        :param args: Parser args.
        :param cluster: Cluster object.
        :return: None
        """
        # configure cluster
        cluster.per_experiment_nb_nodes = args.nodes_per_exp
        cluster.per_experiment_nb_gpus = args.gpus_per_node
        cluster.memory_mb_per_node = args.mem_per_node * 1024
        cluster.per_experiment_nb_cpus = args.cpus_per_node

        # TODO cluster.add_slurm_cmd(cmd='ntasks-per-node', value=1, comment='1 task per GPU')
        cluster.add_slurm_cmd(cmd='constraint', value='localfs', comment='Enable local filesystem')
        cluster.add_slurm_cmd(cmd='partition', value=args.partition, comment='Use partition dedicated for GPUs.')

    @abstractmethod
    def execute_task(self, hparams, cluster: SlurmCluster) -> None:
        """
        Runs training with given hparams.
        :param hparams: Generated hparams.
        :param cluster: SlurmCluster object.
        :return: None
        """
        raise NotImplementedError()

    def launch(self) -> None:
        """
        Runs launcher.
        @return: None
        """
        self.setup_cluster_params(self._parser)
        nb_trials = self.setup_hparams(self._parser)
        args = self._parser.parse_args(self._args)
        self.validate_args(args)
        self.convert_args(args)

        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='./slurm_logs'
        )
        self.configure_cluster(args, cluster)

        # submit a script
        cluster.optimize_parallel_cluster_gpu(
            self.execute_task,
            nb_trials=nb_trials,
            job_name=args.job_name
        )
