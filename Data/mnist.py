import os
import torch
from torchvision import datasets, transforms
from gem_miner_args_helper import gem_miner_parser_args
from torch.utils.data import random_split


class MNIST:
    def __init__(self, args):
        super(MNIST, self).__init__()

        data_root = gem_miner_parser_args.data

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": gem_miner_parser_args.workers, "pin_memory": True} if use_cuda else {}
        
        normalize = transforms.Normalize((0.1307,), (0.3081,))

        dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        test_dataset = datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        if gem_miner_parser_args.use_full_data:
            train_dataset = dataset
            # use_full_data => we are not tuning hyperparameters
            validation_dataset = test_dataset
        else:
            val_size = 5000
            train_size = len(dataset) - val_size
            train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=gem_miner_parser_args.batch_size, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=gem_miner_parser_args.batch_size, shuffle=False, **kwargs
        )

        self.actual_val_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=gem_miner_parser_args.batch_size, shuffle=True, **kwargs
        )








