import sys
import yaml

import Utils.parser as _parser

import argparse
import json
import os

global synflow_parser_args

class SynflowArgsHelper:
    def parse_arguments(self, jupyter_mode=False):

        parser = argparse.ArgumentParser(description='Network Compression')


        # Training Hyperparameters
        training_args = parser.add_argument_group('training')
        training_args.add_argument('--dataset', type=str, default='mnist',
                            choices=['mnist','cifar10'],
                            help='dataset (default: mnist)')
        training_args.add_argument('--model', type=str, default='fc', choices=['fc','resnet20','resnet32','resnet44'],
                            help='model architecture (default: fc)')
        training_args.add_argument('--model-class', type=str, default='default', choices=['default','lottery'],
                            help='model class (default: default)')
        training_args.add_argument('--dense-classifier', type=bool, default=False,
                            help='ensure last layer of model is dense (default: False)')
        training_args.add_argument('--pretrained', type=bool, default=False,
                            help='load pretrained weights (default: False)')
        training_args.add_argument('--optimizer', type=str, default='adam', choices=['sgd','momentum','adam','rms'],
                            help='optimizer (default: adam)')
        training_args.add_argument('--train-batch-size', type=int, default=64,
                            help='input batch size for training (default: 64)')
        training_args.add_argument('--test-batch-size', type=int, default=256,
                            help='input batch size for testing (default: 256)')
        training_args.add_argument('--pre-epochs', type=int, default=0,
                            help='number of epochs to train before pruning (default: 0)')
        training_args.add_argument('--post-epochs', type=int, default=10,
                            help='number of epochs to train after pruning (default: 10)')
        training_args.add_argument('--lr', type=float, default=0.001,
                            help='learning rate (default: 0.001)')
        training_args.add_argument('--lr-drops', type=int, nargs='*', default=[],
                            help='list of learning rate drops (default: [])')
        training_args.add_argument('--lr-drop-rate', type=float, default=0.1,
                            help='multiplicative factor of learning rate drop (default: 0.1)')
        training_args.add_argument('--weight-decay', type=float, default=0.0,
                            help='weight decay (default: 0.0)')
        # Pruning Hyperparameters
        pruning_args = parser.add_argument_group('pruning')
        pruning_args.add_argument('--pruner', type=str, default='synflow', 
                            choices=['synflow'],
                            help='prune strategy (default: rand)')
        pruning_args.add_argument('--compression', type=float, default=0.0,
                            help='quotient of prunable non-zero prunable parameters before and after pruning (default: 0.0)')
        pruning_args.add_argument('--prune-epochs', type=int, default=1,
                            help='number of iterations for scoring (default: 1)')
        pruning_args.add_argument('--compression-schedule', type=str, default='exponential', choices=['linear','exponential'],
                            help='whether to use a linear or exponential compression schedule (default: exponential)')
        pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global','local'],
                            help='masking scope (global or layer) (default: global)')
        pruning_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                            help='ratio of prune dataset size and number of classes (default: 10)')
        pruning_args.add_argument('--prune-batch-size', type=int, default=256,
                            help='input batch size for pruning (default: 256)')
        pruning_args.add_argument('--prune-bias', type=bool, default=False,
                            help='whether to prune bias parameters (default: False)')
        pruning_args.add_argument('--prune-batchnorm', type=bool, default=False,
                            help='whether to prune batchnorm layers (default: False)')
        pruning_args.add_argument('--prune-residual', type=bool, default=False,
                            help='whether to prune residual connections (default: False)')
        pruning_args.add_argument('--prune-train-mode', type=bool, default=False,
                            help='whether to prune in train mode (default: False)')
        pruning_args.add_argument('--reinitialize', type=bool, default=False,
                            help='whether to reinitialize weight parameters after pruning (default: False)')
        pruning_args.add_argument('--shuffle', type=bool, default=False,
                            help='whether to shuffle masks after pruning (default: False)')
        pruning_args.add_argument('--invert', type=bool, default=False,
                            help='whether to invert scores during pruning (default: False)')
        pruning_args.add_argument('--pruner-list', type=str, nargs='*', default=[],
                            help='list of pruning strategies for singleshot (default: [])')
        pruning_args.add_argument('--prune-epoch-list', type=int, nargs='*', default=[],
                            help='list of prune epochs for singleshot (default: [])')
        pruning_args.add_argument('--compression-list', type=float, nargs='*', default=[],
                            help='list of compression ratio exponents for singleshot/multishot (default: [])')
        pruning_args.add_argument('--level-list', type=int, nargs='*', default=[],
                            help='list of number of prune-train cycles (levels) for multishot (default: [])')
        ## Experiment Hyperparameters ##
        parser.add_argument('--experiment', type=str, default='None', 
                            choices=['None', 'synflow'],
                            help='experiment name (default: None)')
        parser.add_argument("--config", default= "Configs/synflow_fc.yml",
                            help="Config file to use")
        parser.add_argument('--expid', type=str, default='',
                            help='name used to save results (default: "")')
        parser.add_argument('--result-dir', type=str, default='Results/',
                            help='path to directory to save results (default: "Results/")')
        parser.add_argument('--gpu', type=int, default='0',
                            help='number of GPU device to use (default: 0)')
        parser.add_argument('--workers', type=int, default='4',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--no-cuda', action='store_true',
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1,
                            help='random seed (default: 1)')
        parser.add_argument('--verbose', action='store_true',
                            help='print statistics during training and testing')
        

        if jupyter_mode:
            args = parser.parse_args("")
        else:
            args = parser.parse_args()
        self.get_config(args, jupyter_mode)

        return args


    def get_config(self, synflow_parser_args, jupyter_mode=False):
        # get commands from command line
        override_args = _parser.argv_to_vars(sys.argv)

        # load yaml file
        yaml_txt = open(synflow_parser_args.config).read()

        # override args
        loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
        if not jupyter_mode:
            for v in override_args:
                loaded_yaml[v] = getattr(synflow_parser_args, v)

        print(f"=> Reading YAML config from {synflow_parser_args.config}")
        synflow_parser_args.__dict__.update(loaded_yaml)


    def isNotebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    def get_args(self, jupyter_mode=False):
        global synflow_parser_args
        jupyter_mode = self.isNotebook()
        synflow_parser_args = self.parse_arguments(jupyter_mode)

argshelper = SynflowArgsHelper()
argshelper.get_args()





