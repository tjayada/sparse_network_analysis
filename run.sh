# uncomment the experiments you want to do
# make sure to adjust the hyper-parameters inside the configfiles directory if needed

##################################################
########### Fully-Connected Models ###############
##################################################

# FC - SynFlow (Random if selected in configfile as such)
#python main.py --config Configs/synflow_fc.yml


# FC - Gem-Miner 
#python main.py --config Configs/gemini_fc.yml

# FC - Random with good Structure
python main.py --config Configs/synflow_rnd_fc.yml



##################################################
############### ResNet-20 Models #################
##################################################


# ResNet-20 SynFlow
#python main.py --config Configs/multi_synflow_resnet20.yml

# ResNet-20 Gem-Miner
#python main.py --config Configs/gemini_resnet20.yml


