from Utils.main_utils import *
from gem_miner_args_helper import gem_miner_parser_args


def run():
    print(gem_miner_parser_args)
    print("\n\nBeginning of process.")
    print_time()
    set_seed(gem_miner_parser_args.seed * gem_miner_parser_args.trial_num)
    #set_seed(gem_miner_parser_args.seed + gem_miner_parser_args.trial_num - 1)

    # gem_miner_parser_args.distributed = gem_miner_parser_args.world_size > 1 or gem_miner_parser_args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if gem_miner_parser_args.multiprocessing_distributed:
        setup_distributed(ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node,), join=True)
    else:
        # Simply call main_worker function
        main_worker(gem_miner_parser_args.gpu, ngpus_per_node)


def main_worker(gpu, ngpus_per_node):
    train, validate, modifier = get_trainer(gem_miner_parser_args)
    gem_miner_parser_args.gpu = gpu
    if gem_miner_parser_args.gpu is not None:
        print("Use GPU: {} for training".format(gem_miner_parser_args.gpu))
    if gem_miner_parser_args.multiprocessing_distributed:
        gem_miner_parser_args.rank = gem_miner_parser_args.rank * ngpus_per_node + gem_miner_parser_args.gpu
        # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        gem_miner_parser_args.batch_size = int(gem_miner_parser_args.batch_size / ngpus_per_node)
        gem_miner_parser_args.num_workers = int(
            (gem_miner_parser_args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        gem_miner_parser_args.world_size = ngpus_per_node * gem_miner_parser_args.world_size
    idty_str = get_idty_str(gem_miner_parser_args)
    if gem_miner_parser_args.subfolder is not None:
        if not os.path.isdir('Results/'):
            os.mkdir('Results/')
        result_subroot = 'Results/' + gem_miner_parser_args.subfolder + '/'
        if not os.path.isdir(result_subroot):
            os.mkdir(result_subroot)
        result_root = result_subroot + '/results_' + idty_str + '/'
    else:
        result_root = 'Results/results_' + idty_str + '/'

    if not os.path.isdir(result_root):
        os.mkdir(result_root)
    model = get_model(gem_miner_parser_args)
    print_model(model, gem_miner_parser_args)

    if gem_miner_parser_args.weight_training:
        model = round_model(model, round_scheme="all_ones", noise=gem_miner_parser_args.noise,
                            ratio=gem_miner_parser_args.noise_ratio, rank=gem_miner_parser_args.gpu)
        model = switch_to_wt(model)
    model = set_gpu(gem_miner_parser_args, model)
    if gem_miner_parser_args.pretrained:
        pretrained(gem_miner_parser_args.pretrained, model)
    if gem_miner_parser_args.pretrained2:
        # model2.load_state_dict(torch.load(gem_miner_parser_args.pretrained2)['state_dict'])
        model2 = copy.deepcopy(model)
        pretrained(gem_miner_parser_args.pretrained2, model2)
    else:
        model2 = None
    optimizer = get_optimizer(gem_miner_parser_args, model)
    data = get_dataset(gem_miner_parser_args)
    scheduler = get_scheduler(optimizer, gem_miner_parser_args.lr_policy)
    #lr_policy = get_policy(gem_miner_parser_args.lr_policy)(optimizer, gem_miner_parser_args)
    if gem_miner_parser_args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=gem_miner_parser_args.label_smoothing)
        # if isinstance(model, nn.parallel.DistributedDataParallel):
        #     model = model.module
    if gem_miner_parser_args.random_subnet: 
        test_random_subnet(model, data, criterion, gem_miner_parser_args, result_root, gem_miner_parser_args.smart_ratio) 
        return
        

    best_acc1, best_acc5, best_acc10, best_train_acc1, best_train_acc5, best_train_acc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # optionally resume from a checkpoint
    if gem_miner_parser_args.resume:
        best_acc1 = resume(gem_miner_parser_args, model, optimizer)
    # when we only evaluate a pretrained model
    if gem_miner_parser_args.evaluate:
        evaluate_without_training(
            gem_miner_parser_args, model, model2, validate, data, criterion)
        return

    # Set up directories & setting
    run_base_dir, ckpt_base_dir, log_base_dir, writer, epoch_time, validation_time, train_time, progress_overall = get_settings(
        gem_miner_parser_args)
    end_epoch = time.time()
    gem_miner_parser_args.start_epoch = gem_miner_parser_args.start_epoch or 0
    acc1 = None
    epoch_list, test_acc_before_round_list, test_acc_list, reg_loss_list, model_sparsity_list, val_acc_list, train_acc_list = [], [], [], [], [], [], []

    # Save the initial model
    torch.save(model.state_dict(), result_root + 'init_model.pth')

    # compute prune_rate to reach target_sparsity
    if not gem_miner_parser_args.override_prune_rate:
        gem_miner_parser_args.prune_rate = get_prune_rate(
            gem_miner_parser_args.target_sparsity, gem_miner_parser_args.iter_period)
        print("Setting prune_rate to {}".format(gem_miner_parser_args.prune_rate))
    else:
        print("Overriding prune_rate to {}".format(gem_miner_parser_args.prune_rate))
    #if gem_miner_parser_args.dataset == 'TinyImageNet':
    #    print_num_dataset(data)
    if not gem_miner_parser_args.weight_training:
        print_layers(gem_miner_parser_args, model)
    


    if gem_miner_parser_args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True) # mixed precision
    else:
        scaler = None

    if gem_miner_parser_args.only_sanity:
        dirs = os.listdir(gem_miner_parser_args.sanity_folder)
        for path in dirs:
            gem_miner_parser_args.results_root = gem_miner_parser_args.sanity_folder +'/'+ path +'/' 
            gem_miner_parser_args.resume =gem_miner_parser_args.results_root + '/model_before_finetune.pth'
            resume(gem_miner_parser_args, model, optimizer)
            
            do_sanity_checks(model, gem_miner_parser_args, data, criterion, epoch_list, test_acc_before_round_list,
                         test_acc_list, reg_loss_list, model_sparsity_list, gem_miner_parser_args.results_root)
            
            #cp_model = round_model(model, round_scheme="all_ones", noise=gem_miner_parser_args.noise,
            #            ratio=gem_miner_parser_args.noise_ratio, rank=gem_miner_parser_args.gpu)
            #print(get_model_sparsity(cp_model))
        return


    # Start training
    for epoch in range(gem_miner_parser_args.start_epoch, gem_miner_parser_args.epochs):
        if gem_miner_parser_args.multiprocessing_distributed:
            data.train_loader.sampler.set_epoch(epoch)

        if gem_miner_parser_args.pretrained and gem_miner_parser_args.drop_bottom_half_weights and epoch == 1:
            print("Loaded pretrained model, so drop the bottom half of the weights in Epoch 1")
            prune(model, drop_bottom_half_weights=True)
            # print("Loaded pretrained model, so randomly drop half the weights in Epoch 1")
            # conv_layers, linear_layers = get_layers(gem_miner_parser_args.arch, model)
            # for layer in (conv_layers+linear_layers):
            #     layer.scores.data = torch.bernoulli(0.5 * torch.ones_like(layer.scores.data))
        # lr_policy(epoch, iteration=None)
        modifier(gem_miner_parser_args, epoch, model)
        cur_lr = get_lr(optimizer)

        # save the score at the beginning of training epoch, so if we set parser.args.rewind_to_epoch to 0
        # that means we save the initialization of score
        if gem_miner_parser_args.rewind_score and gem_miner_parser_args.rewind_to_epoch == epoch:
            # if rewind the score, checkpoint the score when reach the desired epoch (rewind to iteration not yet implemented)
            with torch.no_grad():
                conv_layers, linear_layers = get_layers(
                    gem_miner_parser_args.arch, model)
                for layer in [*conv_layers, *linear_layers]:
                    layer.saved_score.data = layer.score.data


        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5, train_acc10, reg_loss = train(
            data.train_loader, model, criterion, optimizer, epoch, gem_miner_parser_args, writer=writer, scaler=scaler
        )
        train_time.update((time.time() - start_train) / 60)
        scheduler.step()

        # evaluate on validation set
        start_validation = time.time()
        if gem_miner_parser_args.algo in ['hc', 'hc_iter']:
            br_acc1, br_acc5, br_acc10 = validate(
                data.val_loader, model, criterion, gem_miner_parser_args, writer, epoch)  # before rounding
            print('Acc before rounding: {}'.format(br_acc1))
            acc_avg = 0
            for num_trial in range(gem_miner_parser_args.num_test):
                cp_model = round_model(model, gem_miner_parser_args.round, noise=gem_miner_parser_args.noise,
                                       ratio=gem_miner_parser_args.noise_ratio, rank=gem_miner_parser_args.gpu)
                acc1, acc5, acc10 = validate(
                    data.val_loader, cp_model, criterion, gem_miner_parser_args, writer, epoch)
                acc_avg += acc1
            acc_avg /= gem_miner_parser_args.num_test
            acc1 = acc_avg
            print('Acc after rounding: {}'.format(acc1))
            val_acc1, val_acc5, val_acc10 = validate(
                    data.val_loader, cp_model, criterion, gem_miner_parser_args, writer, epoch)
            print('Validation Acc after rounding: {}'.format(val_acc1))
        else:
            acc1, acc5, acc10 = validate(
                data.val_loader, model, criterion, gem_miner_parser_args, writer, epoch)
            print('Acc: {}'.format(acc1))
            val_acc1, val_acc5, val_acc10 = validate(
                data.val_loader, model, criterion, gem_miner_parser_args, writer, epoch)
            print('Validation Acc: {}'.format(val_acc1))

        validation_time.update((time.time() - start_validation) / 60)

        # prune the model every T_{prune} epochs
        if not gem_miner_parser_args.weight_training and gem_miner_parser_args.algo in ['hc_iter', 'global_ep_iter'] and epoch % (gem_miner_parser_args.iter_period) == 0 and epoch != 0:
            if gem_miner_parser_args.algo == 'hc_iter':
                prune(model)
                if gem_miner_parser_args.checkpoint_at_prune:
                    save_checkpoint_at_prune(model, gem_miner_parser_args)
            elif gem_miner_parser_args.algo == 'global_ep_iter':
                # just update prune_rate because the pruning happens on forward anyway
                p = get_prune_rate(gem_miner_parser_args.target_sparsity, gem_miner_parser_args.iter_period)
                gem_miner_parser_args.prune_rate =  1 - (1-p)**np.floor((epoch+1) / gem_miner_parser_args.iter_period)

        # get model sparsity
        if not gem_miner_parser_args.weight_training:
            if gem_miner_parser_args.bottom_k_on_forward:
                cp_model = copy.deepcopy(model)
                prune(cp_model, update_scores=True)
                avg_sparsity = get_model_sparsity(cp_model)
            elif gem_miner_parser_args.algo in ['hc', 'hc_iter']:
                # Round before checking sparsity
                cp_model = round_model(model, gem_miner_parser_args.round, noise=gem_miner_parser_args.noise,
                                       ratio=gem_miner_parser_args.noise_ratio, rank=gem_miner_parser_args.gpu)
                avg_sparsity = get_model_sparsity(cp_model)
            else:
                avg_sparsity = get_model_sparsity(model)
        else:
            # haven't written a weight sparsity function yet
            avg_sparsity = -1
        print('Model avg sparsity: {}'.format(avg_sparsity))

        # if model has been "short-circuited", then no point in continuing training
        if avg_sparsity == 0:
            print("\n\n---------------------------------------------------------------------")
            print("WARNING: Model Sparsity = 0 => Entire network has been pruned!!!")
            print("EXITING and moving to Fine-tune")
            print("---------------------------------------------------------------------\n\n")
            # TODO: Hacky code. Doesn't always work. But quick and easy fix. Just prune all weights to target
            # sparsity, and then continue to finetune so that unflag can do stuff.
            gem_miner_parser_args.prune_rate = 1 - (gem_miner_parser_args.target_sparsity/100)
            prune(model)
            break

        # update all results lists
        epoch_list.append(epoch)
        if gem_miner_parser_args.algo in ['hc', 'hc_iter']:
            test_acc_before_round_list.append(br_acc1)
        else:
            # no before rounding for EP/weight training
            test_acc_before_round_list.append(-1)
        test_acc_list.append(acc1)
        val_acc_list.append(val_acc1)
        train_acc_list.append(train_acc1)
        reg_loss_list.append(reg_loss)
        model_sparsity_list.append(avg_sparsity)

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        if gem_miner_parser_args.ckpt_at_fixed_epochs:
            if epoch in gem_miner_parser_args.ckpt_at_fixed_epochs:
                torch.save(model.state_dict(), result_root + 'wt_model_after_epoch_{}.pth'.format(epoch))

        if gem_miner_parser_args.conv_type == "SampleSubnetConv":
            count = 0
            sum_pr = 0.0
            for n, m in model.named_modules():
                if isinstance(m, SampleSubnetConv):
                    # avg pr across 10 samples
                    pr = 0.0
                    for _ in range(10):
                        pr += (
                            (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                            .float()
                            .mean()
                            .item()
                        )
                    pr /= 10.0
                    writer.add_scalar("pr/{}".format(n), pr, epoch)
                    sum_pr += pr
                    count += 1

            gem_miner_parser_args.prune_rate = sum_pr / count
            writer.add_scalar("pr/average", gem_miner_parser_args.prune_rate, epoch)

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()

        if gem_miner_parser_args.algo in ['hc', 'hc_iter']:
            results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc_before_rounding': test_acc_before_round_list,
                                      'test_acc': test_acc_list, 'val_acc': val_acc_list, 'train_acc': train_acc_list, 'regularization_loss': reg_loss_list, 'model_sparsity': model_sparsity_list})
        else:
            results_df = pd.DataFrame(
                {'epoch': epoch_list, 'test_acc': test_acc_list, 'val_acc': val_acc_list, 'train_acc': train_acc_list, 'model_sparsity': model_sparsity_list})

        if gem_miner_parser_args.results_filename:
            results_filename = gem_miner_parser_args.results_filename
        else:
            results_filename = result_root + 'acc_and_sparsity.csv'
        print("Writing results into: {}".format(results_filename))
        results_df.to_csv(results_filename, index=False)

    # save checkpoint before fine-tuning
    torch.save(model.state_dict(), result_root + 'model_before_finetune.pth')

    print("\n\nHigh accuracy subnetwork found! Rest is just finetuning")
    print_time()

    # finetune weights
    cp_model = copy.deepcopy(model)
    if not gem_miner_parser_args.skip_fine_tune:
        print("Beginning fine-tuning")
        cp_model = finetune(cp_model, gem_miner_parser_args, data, criterion, epoch_list,
                            test_acc_before_round_list, test_acc_list, val_acc_list, train_acc_list, reg_loss_list, model_sparsity_list, result_root)
        # print out the final acc
        eval_and_print(validate, data.val_loader, cp_model, criterion,
                       gem_miner_parser_args, writer=None, description='final model after finetuning')
        # save checkpoint after fine-tuning
        torch.save(cp_model.state_dict(), result_root + 'model_after_finetune.pth')
    else:
        print("Skipping finetuning!!!")

    if not gem_miner_parser_args.skip_sanity_checks:
        do_sanity_checks(model, gem_miner_parser_args, data, criterion, epoch_list, test_acc_before_round_list,
                         test_acc_list, val_acc_list, train_acc_list, reg_loss_list, model_sparsity_list, result_root)

    else:
        print("Skipping sanity checks!!!")

    print("\n\nEnd of process. Exiting")
    print_time()

    if gem_miner_parser_args.multiprocessing_distributed:
        cleanup_distributed()

