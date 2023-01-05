import yaml
import torch


def load_synflow_resnet20_32_sparsity():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('Configs/synflow_resnet20.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.lottery_resnet import resnet20 as syn_resnet20
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    model = syn_resnet20(plan, 10)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/test_results/synflow_resnet20_cifar_seed_42_epochs_100_compr_05/model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model




def load_synflow_resnet20_1_66_sparsity():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/Desktop/NEW_Results/3_synResults_1_6_percent_2_compr_40_acc/synflow_resnet20.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.lottery_resnet import resnet20 as syn_resnet20
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    model = syn_resnet20(plan, 10)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/NEW_Results/3_synResults_1_6_percent_2_compr_40_acc/synflow/synflow_resnet20_mnist_seed_42_epochs_100_compr_05/init_model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_synflow_resnet20_dunno_sparsity():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/resnet20/multi_synflow_resnet20.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.lottery_resnet import resnet20 as syn_resnet20
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    model = syn_resnet20(plan, 10)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/test/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



"""
def load_synflow_resnet20_dunno_sparsity():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/Desktop/NEW_Results/3_synResults_1_6_percent_2_compr_40_acc/synflow_resnet20.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.lottery_resnet import resnet20 as syn_resnet20
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    model = syn_resnet20(plan, 10)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/lel/synflow/synflow_resnet20_cifar_seed_42_epochs_100_compr_2_multi/model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model

"""

def load_synflow_resnet20_18_sparsity():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/Desktop/NEW_Results/3_synResults_1_6_percent_2_compr_40_acc/synflow_resnet20.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.lottery_resnet import resnet20 as syn_resnet20
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    model = syn_resnet20(plan, 10)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/NEW_Results/2_synResults_18_percent_2_5_compt_40_acc/synflow/synflow_resnet20_mnist_seed_42_epochs_100_compr_05/model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model






def load_gemini_resnet20_1_44_sparsity():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/Desktop/syngem_v4/Configs/gemini_resnet20_CIFAR_sparsity_1_44_unflagT.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.resnet_kaiming import resnet20 as gem_resnet20

    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    model = gem_resnet20()
    
    #model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/test_results/hc_sparsity_1_44_unflagT/results_pruning_CIFAR10_resnet20_hc_iter_0_5_5_reg_L2_0_0001_sgd_cosine_lr_0_1_0_1_50_finetune_0_01_MAML_-1_10_fan_False_signed_constant_unif_width_1_0_seed_42_idx_None/model_after_finetune.pth", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/lel/hc_sparsity_1_44_unflagT/results_pruning_CIFAR10_resnet20_hc_iter_0_5_5_reg_L2_0_0001_sgd_cosine_lr_0_1_0_1_50_finetune_0_01_MAML_-1_10_fan_False_signed_constant_unif_width_1_0_seed_21_idx_None/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


#######################################

######## Load 0 sparsity FC Models #########

#######################################




def load_synflow_fc_sparsity_0_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/100/100_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_0_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/100/100_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_0_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/100/100_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model





#######################################

######## Load MNIST FC Models #########

#######################################


def load_gemini_fc_50_sparsity_seed_21():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/gem_50_21/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_gemini_fc_50_sparsity_seed_42():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/gem_50_42/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_gemini_fc_50_sparsity_seed_63():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/gem_50_63/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_gemini_fc_20_sparsity_seed_21():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("All_Results/20/gem_20_21/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_gemini_fc_20_sparsity_seed_42():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("All_Results/20/gem_20_42/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_20_sparsity_seed_63():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("All_Results/20/gem_20_63/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_gemini_fc_10_sparsity_seed_21():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/gem_10_21/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_10_sparsity_seed_42():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/gem_10_42/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_10_sparsity_seed_63():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/gem_10_63/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_gemini_fc_5_sparsity_seed_21():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/gem_5_21/model_after_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_5_sparsity_seed_42():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/gem_5_42/model_before_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_5_sparsity_seed_63():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/gem_5_63/model_before_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_2_sparsity_seed_21():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/gem_2_21/model_before_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_2_sparsity_seed_42():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/gem_2_42/model_before_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_gemini_fc_2_sparsity_seed_63():
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/fc_gem_50.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    from Models.mlp import FC as gem_fc

    
    #D = 20
    #W = 16
    #plan = [(W, D), (2*W, D), (4*W, D)]
    input_shape, num_classes = (1, 28, 28), 10
    #input_shape, num_classes = (3, 32, 32), 10

    model = gem_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/gem_2_63/model_before_finetune.pth", map_location=torch.device('cpu')))

    model.eval()

    return model




################
# synflow
###############




def load_synflow_fc_sparsity_50_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/syn_50_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_50_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/syn_50_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_50_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/syn_50_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_synflow_fc_sparsity_20_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/20/syn_20_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_20_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/20/syn_20_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_20_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/20/syn_20_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_synflow_fc_sparsity_10_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/syn_10_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_10_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/syn_10_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_10_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/syn_10_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_5_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/syn_5_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_synflow_fc_sparsity_5_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/syn_5_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_synflow_fc_sparsity_5_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/syn_5_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_2_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/syn_2_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_2_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/syn_2_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_synflow_fc_sparsity_2_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/syn_2_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



############################################################
## random ##
############################################################



def load_random_fc_sparsity_50_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/rnd_50_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_50_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/rnd_50_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_50_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/50/rnd_50_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_random_fc_sparsity_20_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/20/rnd_20_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_20_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/20/rnd_20_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_20_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/20/rnd_20_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_10_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/rnd_10_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_10_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/rnd_10_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_10_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/10/rnd_10_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_5_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/rnd_5_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_5_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/rnd_5_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_5_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/5/rnd_5_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model


def load_random_fc_sparsity_2_seed_21():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/rnd_2_21/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_random_fc_sparsity_2_seed_42():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/rnd_2_42/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model



def load_random_fc_sparsity_2_seed_63():
    from synflow_args_helper import synflow_parser_args
    syn_yaml_txt = open('/Users/tjarkdarius/BA_repo/place_holder/FC/synflow_fc.yml').read()
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    from Models.mlp import fc as syn_fc
    
    D = 20
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]

    input_shape, num_classes = (1, 28, 28), 10
    model = syn_fc(input_shape, num_classes)
    
    model.load_state_dict(torch.load("/Users/tjarkdarius/Desktop/New_results/2/rnd_2_63/post-model.pt", map_location=torch.device('cpu')))

    model.eval()

    return model





