# change args accordingly 

# 1 Experiment
#python main.py --dataset cifar10 --model resnet20 --model-class lottery --optimizer momentum --train-batch-size 128 --post-epochs 160 --lr 0.01 --lr-drops 60 120 --lr-drop-rate 0.2 --weight-decay 0.0005 --pruner synflow --prune-epochs 100 --result-dir synflow_results  --workers 1 --verbose --seed 21 --expid synflow_seed_21

# maybe next change --compression from default 0 to 1 ?


# test with fully connected model
#python main.py --dataset mnist --model fc --model-class default --optimizer adam --post-epochs 50 --pruner synflow --compression 0 --result-dir synflow_results --workers 1 --verbose --seed 42 --expid synflow_fc_mnist_seed_42_epochs_50_compr_1

# new test with config files
python main.py --config Configs/synflow_fc.yml

# random fc synflow
#python main.py --config Configs/synflow_rnd_fc.yml

# test with gemini mnist fc
#python main.py --config Configs/gemini_fc.yml

# random fc gem miner
#python main.py --config Configs/gemini_fc_rnd.yml


# test synflow resnet20 on mnist
#python main.py --config Configs/synflow_resnet20.yml

# synflow multishot
#python main.py --config Configs/multi_synflow_resnet20.yml


# create random network created with gemini
#python main.py --config Configs/random_resnet20_CIFAR_sparsity_1_44_unflagT.yml


# gemini test
#python main.py --config Configs/hypercube/resnet20/resnet20_wt.yml


# test gemini on mnist with resnet20 and 1.44 sparsity
#python main.py --config Configs/gemini_resnet20_MNIST_sparsity_1_44_unflagT.yml


# test gemini on cifar with resnet20 and 1.44 sparsity
#python main.py --config Configs/gemini_resnet20_CIFAR_sparsity_1_44_unflagT.yml
