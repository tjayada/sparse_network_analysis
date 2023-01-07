import pandas as pd
import numpy as np
from load_models import *
from syngem_utils import *


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
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "test_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    df_div = df_div.drop(np.arange(0,25,1))
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div


def load_gem_resnet_acc(sparse):
    seed_21 = pd.read_csv("All_Results/Resnet/{sparse}/gem_{sparse}_21/acc_and_sparsity.csv")
    seed_42 = pd.read_csv("All_Results/Resnet/{sparse}/gem_{sparse}_42/acc_and_sparsity.csv")
    seed_63 = pd.read_csv("All_Results/Resnet/{sparse}/gem_{sparse}_63/acc_and_sparsity.csv")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "test_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    df_div = df_div.drop(np.arange(0,25,1))
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div


"""
def load_gem_50_acc():
    seed_21 = pd.read_csv("All_Results/50/gem_50_21/acc_and_sparsity.csv")
    seed_42 = pd.read_csv("All_Results/50/gem_50_42/acc_and_sparsity.csv")
    seed_63 = pd.read_csv("All_Results/50/gem_50_63/acc_and_sparsity.csv")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "test_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    df_div = df_div.drop(np.arange(0,25,1))
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div


def load_gem_20_acc():
    seed_21 = pd.read_csv("All_Results/20/gem_20_21/acc_and_sparsity.csv")
    seed_42 = pd.read_csv("All_Results/20/gem_20_42/acc_and_sparsity.csv")
    seed_63 = pd.read_csv("All_Results/20/gem_20_63/acc_and_sparsity.csv")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "test_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    df_div = df_div.drop(np.arange(0,25,1))
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div


def load_gem_10_acc():
    seed_21 = pd.read_csv("All_Results/10/gem_10_21/acc_and_sparsity.csv")
    seed_42 = pd.read_csv("All_Results/10/gem_10_42/acc_and_sparsity.csv")
    seed_63 = pd.read_csv("All_Results/10/gem_10_63/acc_and_sparsity.csv")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "test_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    df_div = df_div.drop(np.arange(0,25,1))
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div


def load_gem_5_acc():
    seed_21 = pd.read_csv("All_Results/5/gem_5_21/acc_and_sparsity.csv")
    seed_42 = pd.read_csv("All_Results/5/gem_5_42/acc_and_sparsity.csv")
    seed_63 = pd.read_csv("All_Results/5/gem_5_63/acc_and_sparsity.csv")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "test_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    df_div = df_div.drop(np.arange(0,25,1))
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div


def load_gem_2_acc():
    seed_21 = pd.read_csv("All_Results/2/gem_2_21/acc_and_sparsity.csv")
    seed_42 = pd.read_csv("All_Results/2/gem_2_42/acc_and_sparsity.csv")
    seed_63 = pd.read_csv("All_Results/2/gem_2_63/acc_and_sparsity.csv")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["epoch", "test_acc_before_rounding", "test_acc","train_acc", "regularization_loss", "model_sparsity"], axis = 1)
    df_div = df_div.drop(np.arange(0,25,1))
    df_div = df_div.reset_index()
    df_div = df_div.drop("index", axis=1)
    return df_div

"""

###########################

## Load Random Accuracy ##

###########################


def load_rnd_50_acc():
    seed_21 = pd.read_pickle("All_Results/50/rnd_50_21/post-train-rand-0.3-10.pkl")
    seed_42 = pd.read_pickle("All_Results/50/rnd_50_42/post-train-rand-0.3-10.pkl")
    seed_63 = pd.read_pickle("All_Results/50/rnd_50_63/post-train-rand-0.3-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_rnd_20_acc():
    seed_21 = pd.read_pickle("All_Results/20/rnd_20_21/post-train-rand-0.695-10.pkl")
    seed_42 = pd.read_pickle("All_Results/20/rnd_20_42/post-train-rand-0.695-10.pkl")
    seed_63 = pd.read_pickle("All_Results/20/rnd_20_63/post-train-rand-0.695-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_rnd_10_acc():
    seed_21 = pd.read_pickle("All_Results/10/rnd_10_21/post-train-rand-1-10.pkl")
    seed_42 = pd.read_pickle("All_Results/10/rnd_10_42/post-train-rand-1-10.pkl")
    seed_63 = pd.read_pickle("All_Results/10/rnd_10_63/post-train-rand-1-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_rnd_5_acc():
    seed_21 = pd.read_pickle("All_Results/5/rnd_5_21/post-train-rand-1.3-10.pkl")
    seed_42 = pd.read_pickle("All_Results/5/rnd_5_42/post-train-rand-1.3-10.pkl")
    seed_63 = pd.read_pickle("All_Results/5/rnd_5_63/post-train-rand-1.3-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_rnd_2_acc():
    seed_21 = pd.read_pickle("All_Results/2/rnd_2_21/post-train-rand-1.7-10.pkl")
    seed_42 = pd.read_pickle("All_Results/2/rnd_2_42/post-train-rand-1.7-10.pkl")
    seed_63 = pd.read_pickle("All_Results/2/rnd_2_63/post-train-rand-1.7-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div





#############################

## Load Synflow Accuracy ##

#############################


def load_syn_50_acc():
    seed_21 = pd.read_pickle("All_Results/50/syn_50_21/post-train-synflow-0.3-10.pkl")
    seed_42 = pd.read_pickle("All_Results/50/syn_50_42/post-train-synflow-0.3-10.pkl")
    seed_63 = pd.read_pickle("All_Results/50/syn_50_63/post-train-synflow-0.3-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_syn_20_acc():
    seed_21 = pd.read_pickle("All_Results/20/syn_20_21/post-train-synflow-0.695-10.pkl")
    seed_42 = pd.read_pickle("All_Results/20/syn_20_42/post-train-synflow-0.695-10.pkl")
    seed_63 = pd.read_pickle("All_Results/20/syn_20_63/post-train-synflow-0.695-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_syn_10_acc():
    seed_21 = pd.read_pickle("All_Results/10/syn_10_21/post-train-synflow-1-10.pkl")
    seed_42 = pd.read_pickle("All_Results/10/syn_10_42/post-train-synflow-1-10.pkl")
    seed_63 = pd.read_pickle("All_Results/10/syn_10_63/post-train-synflow-1-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_syn_5_acc():
    seed_21 = pd.read_pickle("All_Results/5/syn_5_21/post-train-synflow-1.3-10.pkl")
    seed_42 = pd.read_pickle("All_Results/5/syn_5_42/post-train-synflow-1.3-10.pkl")
    seed_63 = pd.read_pickle("All_Results/5/syn_5_63/post-train-synflow-1.3-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div


def load_syn_2_acc():
    seed_21 = pd.read_pickle("All_Results/2/syn_2_21/post-train-synflow-1.7-10.pkl")
    seed_42 = pd.read_pickle("All_Results/2/syn_2_42/post-train-synflow-1.7-10.pkl")
    seed_63 = pd.read_pickle("All_Results/2/syn_2_63/post-train-synflow-1.7-10.pkl")
    
    df_add = seed_21.add(seed_42, fill_value=0)
    df_add = df_add.add(seed_63, fill_value=0)
    
    df_div = df_add.div(3)
    
    df_div = df_div.drop(["train_loss", "test_loss", "top5_accuracy"], axis = 1)
    
    return df_div



######################################

## Load Gem-Miner First Layer Units ##

######################################


def load_gem_50_first_layer_units():
    gem_model_50_21 = load_gemini_fc_50_sparsity_seed_21()
    gem_model_50_42 = load_gemini_fc_50_sparsity_seed_42()
    gem_model_50_63 = load_gemini_fc_50_sparsity_seed_63()

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


def load_gem_20_first_layer_units():
    gem_model_20_21 = load_gemini_fc_20_sparsity_seed_21()
    gem_model_20_42 = load_gemini_fc_20_sparsity_seed_42()
    gem_model_20_63 = load_gemini_fc_20_sparsity_seed_63()

    gem_fil_20_21 = get_filters(gem_model_20_21)
    gem_fil_20_42 = get_filters(gem_model_20_42)
    gem_fil_20_63 = get_filters(gem_model_20_63)

    all_models = [gem_fil_20_21[0] , gem_fil_20_42[0], gem_fil_20_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_gem_10_first_layer_units():
    gem_model_10_21 = load_gemini_fc_10_sparsity_seed_21()
    gem_model_10_42 = load_gemini_fc_10_sparsity_seed_42()
    gem_model_10_63 = load_gemini_fc_10_sparsity_seed_63()

    gem_fil_10_21 = get_filters(gem_model_10_21)
    gem_fil_10_42 = get_filters(gem_model_10_42)
    gem_fil_10_63 = get_filters(gem_model_10_63)

    all_models = [gem_fil_10_21[0] , gem_fil_10_42[0], gem_fil_10_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_gem_5_first_layer_units():
    gem_model_5_21 = load_gemini_fc_5_sparsity_seed_21()
    gem_model_5_42 = load_gemini_fc_5_sparsity_seed_42()
    gem_model_5_63 = load_gemini_fc_5_sparsity_seed_63()

    gem_fil_5_21 = get_filters(gem_model_5_21)
    gem_fil_5_42 = get_filters(gem_model_5_42)
    gem_fil_5_63 = get_filters(gem_model_5_63)

    all_models = [gem_fil_5_21[0] , gem_fil_5_42[0], gem_fil_5_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_gem_2_first_layer_units():
    gem_model_2_21 = load_gemini_fc_2_sparsity_seed_21()
    gem_model_2_42 = load_gemini_fc_2_sparsity_seed_42()
    gem_model_2_63 = load_gemini_fc_2_sparsity_seed_63()

    gem_fil_2_21 = get_filters(gem_model_2_21)
    gem_fil_2_42 = get_filters(gem_model_2_42)
    gem_fil_2_63 = get_filters(gem_model_2_63)

    all_models = [gem_fil_2_21[0] , gem_fil_2_42[0], gem_fil_2_63[0]]
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

def load_syn_50_first_layer_units():
    syn_model_50_21 = load_synflow_fc_sparsity_50_seed_21()
    syn_model_50_42 = load_synflow_fc_sparsity_50_seed_42()
    syn_model_50_63 = load_synflow_fc_sparsity_50_seed_63()

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


def load_syn_20_first_layer_units():
    syn_model_20_21 = load_synflow_fc_sparsity_20_seed_21()
    syn_model_20_42 = load_synflow_fc_sparsity_20_seed_42()
    syn_model_20_63 = load_synflow_fc_sparsity_20_seed_63()

    syn_fil_20_21 = get_filters(syn_model_20_21)
    syn_fil_20_42 = get_filters(syn_model_20_42)
    syn_fil_20_63 = get_filters(syn_model_20_63)

    all_models = [syn_fil_20_21[0] , syn_fil_20_42[0], syn_fil_20_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_syn_10_first_layer_units():
    syn_model_10_21 = load_synflow_fc_sparsity_10_seed_21()
    syn_model_10_42 = load_synflow_fc_sparsity_10_seed_42()
    syn_model_10_63 = load_synflow_fc_sparsity_10_seed_63()

    syn_fil_10_21 = get_filters(syn_model_10_21)
    syn_fil_10_42 = get_filters(syn_model_10_42)
    syn_fil_10_63 = get_filters(syn_model_10_63)

    all_models = [syn_fil_10_21[0] , syn_fil_10_42[0], syn_fil_10_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_syn_5_first_layer_units():
    syn_model_5_21 = load_synflow_fc_sparsity_5_seed_21()
    syn_model_5_42 = load_synflow_fc_sparsity_5_seed_42()
    syn_model_5_63 = load_synflow_fc_sparsity_5_seed_63()

    syn_fil_5_21 = get_filters(syn_model_5_21)
    syn_fil_5_42 = get_filters(syn_model_5_42)
    syn_fil_5_63 = get_filters(syn_model_5_63)

    all_models = [syn_fil_5_21[0] , syn_fil_5_42[0], syn_fil_5_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_syn_2_first_layer_units():
    syn_model_2_21 = load_synflow_fc_sparsity_2_seed_21()
    syn_model_2_42 = load_synflow_fc_sparsity_2_seed_42()
    syn_model_2_63 = load_synflow_fc_sparsity_2_seed_63()


    syn_fil_2_21 = get_filters(syn_model_2_21)
    syn_fil_2_42 = get_filters(syn_model_2_42)
    syn_fil_2_63 = get_filters(syn_model_2_63)



    all_models = [syn_fil_2_21[0] , syn_fil_2_42[0], syn_fil_2_63[0]]
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

def load_rnd_50_first_layer_units():
    rnd_model_50_21 = load_random_fc_sparsity_50_seed_21()
    rnd_model_50_42 = load_random_fc_sparsity_50_seed_42()
    rnd_model_50_63 = load_random_fc_sparsity_50_seed_63()

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


def load_rnd_20_first_layer_units():
    rnd_model_20_21 = load_random_fc_sparsity_20_seed_21()
    rnd_model_20_42 = load_random_fc_sparsity_20_seed_42()
    rnd_model_20_63 = load_random_fc_sparsity_20_seed_63()

    rnd_fil_20_21 = get_filters(rnd_model_20_21)
    rnd_fil_20_42 = get_filters(rnd_model_20_42)
    rnd_fil_20_63 = get_filters(rnd_model_20_63)

    all_models = [rnd_fil_20_21[0] , rnd_fil_20_42[0], rnd_fil_20_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_rnd_10_first_layer_units():
    rnd_model_10_21 = load_random_fc_sparsity_10_seed_21()
    rnd_model_10_42 = load_random_fc_sparsity_10_seed_42()
    rnd_model_10_63 = load_random_fc_sparsity_10_seed_63()

    rnd_fil_10_21 = get_filters(rnd_model_10_21)
    rnd_fil_10_42 = get_filters(rnd_model_10_42)
    rnd_fil_10_63 = get_filters(rnd_model_10_63)

    all_models = [rnd_fil_10_21[0] , rnd_fil_10_42[0], rnd_fil_10_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_rnd_5_first_layer_units():
    rnd_model_5_21 = load_random_fc_sparsity_5_seed_21()
    rnd_model_5_42 = load_random_fc_sparsity_5_seed_42()
    rnd_model_5_63 = load_random_fc_sparsity_5_seed_63()

    rnd_fil_5_21 = get_filters(rnd_model_5_21)
    rnd_fil_5_42 = get_filters(rnd_model_5_42)
    rnd_fil_5_63 = get_filters(rnd_model_5_63)

    all_models = [rnd_fil_5_21[0] , rnd_fil_5_42[0], rnd_fil_5_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units


def load_rnd_2_first_layer_units():
    rnd_model_2_21 = load_random_fc_sparsity_2_seed_21()
    rnd_model_2_42 = load_random_fc_sparsity_2_seed_42()
    rnd_model_2_63 = load_random_fc_sparsity_2_seed_63()

    rnd_fil_2_21 = get_filters(rnd_model_2_21)
    rnd_fil_2_42 = get_filters(rnd_model_2_42)
    rnd_fil_2_63 = get_filters(rnd_model_2_63)

    all_models = [rnd_fil_2_21[0] , rnd_fil_2_42[0], rnd_fil_2_63[0]]
    all_units = []

    for model in all_models:
        model_units = 0
        for unit in model:
            model_units += unit
        all_units.append(model_units)
        
    return all_units
