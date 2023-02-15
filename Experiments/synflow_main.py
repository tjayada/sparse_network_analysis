import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from Trainers.train import *
from Pruners.prune import *

from synflow_args_helper import synflow_parser_args



def run():
    
    print(synflow_parser_args)

    ## Construct Result Directory ##
    if synflow_parser_args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(synflow_parser_args, 'save', False)
    else:
        result_dir = '{}/{}/{}'.format(synflow_parser_args.result_dir, synflow_parser_args.experiment, synflow_parser_args.expid)
        setattr(synflow_parser_args, 'save', True)
        setattr(synflow_parser_args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            val = ""
            while val not in ['yes', 'no']:
                val = input("Experiment '{}' with expid '{}' exists.  Overwrite (yes/no)? ".format(synflow_parser_args.experiment, synflow_parser_args.expid))
            if val == 'no':
                quit()

    ## Save Args ##
    if synflow_parser_args.save:
        with open(synflow_parser_args.result_dir + '/synflow_parser_args.json', 'w') as f:
            json.dump(synflow_parser_args.__dict__, f, sort_keys=True, indent=4)


    ## Random Seed and Device ##
    torch.manual_seed(synflow_parser_args.seed)
    device = load.device(synflow_parser_args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(synflow_parser_args.dataset))
    input_shape, num_classes = load.dimension(synflow_parser_args.dataset) 
    prune_loader = load.dataloader(synflow_parser_args.dataset, synflow_parser_args.prune_batch_size, True, synflow_parser_args.workers, synflow_parser_args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(synflow_parser_args.dataset, synflow_parser_args.train_batch_size, True, synflow_parser_args.workers)
    test_loader = load.dataloader(synflow_parser_args.dataset, synflow_parser_args.test_batch_size, False, synflow_parser_args.workers)

    ## Model ##
    print('Creating {} model.'.format(synflow_parser_args.model))
    model = load.model(synflow_parser_args.model, synflow_parser_args.model_class)(input_shape, 
                                                     num_classes, 
                                                     synflow_parser_args.dense_classifier,
                                                     synflow_parser_args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(synflow_parser_args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=synflow_parser_args.lr, weight_decay=synflow_parser_args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=synflow_parser_args.lr_drops, gamma=synflow_parser_args.lr_drop_rate)

    ## Save Original ##
    torch.save(model.state_dict(),"{}/model.pt".format(synflow_parser_args.result_dir))
    torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(synflow_parser_args.result_dir))
    torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(synflow_parser_args.result_dir))

    ## Train-Prune Loop ##
    for compression in synflow_parser_args.compression_list:
        for level in synflow_parser_args.level_list:
            print('{} compression ratio, {} train-prune levels'.format(compression, level))
            
            # Reset Model, Optimizer, and Scheduler
            model.load_state_dict(torch.load("{}/model.pt".format(synflow_parser_args.result_dir), map_location=device))
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(synflow_parser_args.result_dir), map_location=device))
            scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(synflow_parser_args.result_dir), map_location=device))
            
            for l in range(level):

                # Pre Train Model
                pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                test_loader, device, synflow_parser_args.pre_epochs, synflow_parser_args.verbose)

                # Prune Model
                pruner = load.pruner(synflow_parser_args.pruner)(generator.masked_parameters(model, synflow_parser_args.prune_bias, synflow_parser_args.prune_batchnorm, synflow_parser_args.prune_residual))
                sparsity = (10**(-float(compression)))**((l + 1) / level)
                prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                           synflow_parser_args.compression_schedule, synflow_parser_args.mask_scope, synflow_parser_args.prune_epochs, synflow_parser_args.reinitialize, synflow_parser_args.prune_train_mode, synflow_parser_args.shuffle, synflow_parser_args.invert)

                # Reset Model's Weights
                original_dict = torch.load("{}/model.pt".format(synflow_parser_args.result_dir), map_location=device)
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(synflow_parser_args.result_dir), map_location=device))
                scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(synflow_parser_args.result_dir), map_location=device))

            # avoid all the pruning metrics when strucutre is hard-coded, indicated by synflow_parser_args.level_list[0] == 0
            if synflow_parser_args.level_list[0] != 0:
                # Prune Result
                prune_result = metrics.summary(model, 
                                               pruner.scores,
                                               metrics.flop(model, input_shape, device),
                                               lambda p: generator.prunable(p, synflow_parser_args.prune_batchnorm, synflow_parser_args.prune_residual))
                
            # Save init model before training
            torch.save(model.state_dict(),"{}/model_before_train.pt".format(synflow_parser_args.result_dir))

            # Train Model
            post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                          test_loader, device, synflow_parser_args.post_epochs, synflow_parser_args.verbose)
            

            ## Display Results ##
            if synflow_parser_args.level_list[0] != 0:
                frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
                train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
                prune_result = metrics.summary(model, 
                                               pruner.scores,
                                               metrics.flop(model, input_shape, device),
                                               lambda p: generator.prunable(p, synflow_parser_args.prune_batchnorm, synflow_parser_args.prune_residual))
                total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
                possible_params = prune_result['size'].sum()
                total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
                possible_flops = prune_result['flops'].sum()
                print("Train results:\n", train_result)
                print("Prune results:\n", prune_result)
                print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
                print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))


            # Save Data
            post_result.to_pickle("{}/post-train-{}-{}-{}.pkl".format(synflow_parser_args.result_dir, synflow_parser_args.pruner, str(compression),  str(level)))
            if synflow_parser_args.level_list[0] != 0:
                prune_result.to_pickle("{}/compression-{}-{}-{}.pkl".format(synflow_parser_args.result_dir, synflow_parser_args.pruner, str(compression), str(level)))
            torch.save(model.state_dict(),"{}/post-model.pt".format(synflow_parser_args.result_dir))
            if synflow_parser_args.level_list[0] != 0:
                torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(synflow_parser_args.result_dir))
                torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(synflow_parser_args.result_dir))

