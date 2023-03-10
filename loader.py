import pandas as pd
import numpy as np
import yaml
import torch
from syngem_utils import *



############################

## Load Gem-Miner Models ##

############################


def load_gemini_model(model_name, sparse, seed):
    from gem_miner_args_helper import gem_miner_parser_args    
    gem_yaml_txt = open(f'Configs/gemini_{model_name}.yml').read()

    gem_loaded_yaml = yaml.load(gem_yaml_txt, Loader=yaml.FullLoader)

    gem_miner_parser_args.__dict__.update(gem_loaded_yaml)
    
    if model_name == "FC":
        from Models.mlp import FC as gem_fc
        input_shape, num_classes = (1, 28, 28), 10
        model = gem_fc(input_shape, num_classes)
    
    else:
        from Models.resnet_kaiming import resnet20 as gem_resnet20
        model = gem_resnet20()

    model.load_state_dict(torch.load(f"All_Results/{model_name}/{sparse}/gem_{sparse}_{seed}/model_after_finetune.pth", map_location=torch.device('cpu')))
    model.eval()

    return model



##########################

## Load Synflow Models ##

##########################

def load_synflow_model(model_name, sparse, seed):
    from synflow_args_helper import synflow_parser_args
    
    if model_name == "FC":
        syn_yaml_txt = open(f'Configs/synflow_{model_name}.yml').read()
    else:

        syn_yaml_txt = open(f'Configs/multi_synflow_{model_name}.yml').read()
    
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    
    if model_name == "FC":
        from Models.mlp import fc as syn_fc
        input_shape, num_classes = (1, 28, 28), 10
        model = syn_fc(input_shape, num_classes)
    
    else:
        from Models.lottery_resnet import resnet20 as syn_resnet20
        D = 20
        W = 16
        plan = [(W, D), (2*W, D), (4*W, D)]
        model = syn_resnet20(plan, 10)
    
    model.load_state_dict(torch.load(f"All_Results/{model_name}/{sparse}/syn_{sparse}_{seed}/post-model.pt", map_location=torch.device('cpu')))
    model.eval()

    return model



##########################

## Load Random Models ##

##########################


def load_random_model(model_name, sparse, seed):
    from synflow_args_helper import synflow_parser_args

    if model_name == "FC":
        syn_yaml_txt = open(f'Configs/synflow_{model_name}.yml').read()
    else:
        syn_yaml_txt = open(f'Configs/multi_synflow_{model_name}.yml').read()
    
    syn_loaded_yaml = yaml.load(syn_yaml_txt, Loader=yaml.FullLoader)
    
    synflow_parser_args.__dict__.update(syn_loaded_yaml)
    
    if model_name == "FC":
        from Models.mlp import fc as syn_fc
        input_shape, num_classes = (1, 28, 28), 10
        model = syn_fc(input_shape, num_classes)
    
    else:
        from Models.lottery_resnet import resnet20 as syn_resnet20
        D = 20
        W = 16
        plan = [(W, D), (2*W, D), (4*W, D)]
        model = syn_resnet20(plan, 10)
    
    model.load_state_dict(torch.load(f"All_Results/{model_name}/{sparse}/rnd_{sparse}_{seed}/post-model.pt", map_location=torch.device('cpu')))
    model.eval()

    return model




#############################

## Load Gem-Miner Accuracy ##

#############################


def load_gem_acc(model, sparse):
    seed_21 = pd.read_csv(f"All_Results/{model}/{sparse}/gem_{sparse}_21/acc_and_sparsity.csv")
    seed_42 = pd.read_csv(f"All_Results/{model}/{sparse}/gem_{sparse}_42/acc_and_sparsity.csv")
    seed_63 = pd.read_csv(f"All_Results/{model}/{sparse}/gem_{sparse}_63/acc_and_sparsity.csv")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "val_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    
    if model == "FC":
        df_div = df_div.drop(np.arange(0,25,1))
    else:
        df_div = df_div.drop(np.arange(0,150,1))
        
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div



###########################

## Load Synflow Accuracy ##

###########################


def load_syn_acc(model, sparse):
    seed_21 = pd.read_pickle(f"All_Results/{model}/{sparse}/syn_{sparse}_21/post-train.pkl")
    seed_42 = pd.read_pickle(f"All_Results/{model}/{sparse}/syn_{sparse}_42/post-train.pkl")
    seed_63 = pd.read_pickle(f"All_Results/{model}/{sparse}/syn_{sparse}_63/post-train.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div



###########################

## Load Random Accuracy ##

###########################


def load_rnd_acc(model, sparse):
    seed_21 = pd.read_pickle(f"All_Results/{model}/{sparse}/rnd_{sparse}_21/post-train.pkl")
    seed_42 = pd.read_pickle(f"All_Results/{model}/{sparse}/rnd_{sparse}_42/post-train.pkl")
    seed_63 = pd.read_pickle(f"All_Results/{model}/{sparse}/rnd_{sparse}_63/post-train.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div




######################################

## Load Gem-Miner First Layer Units ##

######################################


def load_gem_first_layer_units(sparse):
    gem_model_50_21 = load_gemini_model("FC", sparse, 21)
    gem_model_50_42 = load_gemini_model("FC", sparse, 42)
    gem_model_50_63 = load_gemini_model("FC", sparse, 63)

    gem_fil_50_21 = get_filters(gem_model_50_21)
    gem_fil_50_42 = get_filters(gem_model_50_42)
    gem_fil_50_63 = get_filters(gem_model_50_63)

    all_models = [gem_fil_50_21[0] , gem_fil_50_42[0], gem_fil_50_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units



#####################################

## Load Synflow First Layer Units ##

#####################################

def load_syn_first_layer_units(sparse):
    syn_model_50_21 = load_synflow_model("FC", sparse, 21)
    syn_model_50_42 = load_synflow_model("FC", sparse, 42)
    syn_model_50_63 = load_synflow_model("FC", sparse, 63)

    syn_fil_50_21 = get_filters(syn_model_50_21)
    syn_fil_50_42 = get_filters(syn_model_50_42)
    syn_fil_50_63 = get_filters(syn_model_50_63)

    all_models = [syn_fil_50_21[0] , syn_fil_50_42[0], syn_fil_50_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units



##################################

## Load RandomFirst Layer Units ##

##################################

def load_rnd_first_layer_units(sparse):
    rnd_model_50_21 = load_random_model("FC", sparse, 21)
    rnd_model_50_42 = load_random_model("FC", sparse, 21)
    rnd_model_50_63 = load_random_model("FC", sparse, 21)

    rnd_fil_50_21 = get_filters(rnd_model_50_21)
    rnd_fil_50_42 = get_filters(rnd_model_50_42)
    rnd_fil_50_63 = get_filters(rnd_model_50_63)

    all_models = [rnd_fil_50_21[0] , rnd_fil_50_42[0], rnd_fil_50_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units
