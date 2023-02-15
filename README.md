# An Analysis Of Sparse Neural Networks


## Introduction
This GitHub repository contains all code created for my bachelor thesis. It is a combination of already existing pruning algorithms and sparse neural network analysis methods, as well as new approaches discussed in my thesis.

## Navigating this repository

### Directories
* [All Results](https://github.com/tjayada/sparse_network_analysis/tree/main/All_Results) - Contains all sparse networks trained and obtained
* [Configs](https://github.com/tjayada/sparse_network_analysis/tree/main/Configs) - Contains configuration files of models for better repruceabilty of results
* [Data](https://github.com/tjayada/sparse_network_analysis/tree/main/Data) - Contains the dataset that were used
* [Experiments](https://github.com/tjayada/sparse_network_analysis/tree/main/Experiments) - Contains the main files of both pruning algorithms, which get called from the main.py
* [Figures](https://github.com/tjayada/sparse_network_analysis/tree/main/Figures) - Contains the plots created from the notebooks
* [Layers](https://github.com/tjayada/sparse_network_analysis/tree/main/Layers) - Contains the unique implementations of layers that the pruning algorithms require to work properly
* [Models](https://github.com/tjayada/sparse_network_analysis/tree/main/Models) - Contains the implementations of the Fully-Connected and ResNet-20 models
* [Notebooks](https://github.com/tjayada/sparse_network_analysis/tree/main/Notebooks) - Contains all the notebooks created for analysis of each sparsity level and model
* [Pruners](https://github.com/tjayada/sparse_network_analysis/tree/main/Pruners) - Contains parts of the pruning algorithms
* [Trainers](https://github.com/tjayada/sparse_network_analysis/tree/main/Trainers) - Contains training loops from the pruning algorithms
* [Utils](https://github.com/tjayada/sparse_network_analysis/tree/main/Utils) - Contains most of the functions necessary to make the creation, the pruning and the training of models possible

### Files
* [thesis_notebook](https://github.com/tjayada/sparse_network_analysis/blob/main/thesis_notebook.ipynb) - Contains implementations of all plots used in the thesis
* [README](https://github.com/tjayada/sparse_network_analysis/blob/main/README.md) - Contains the words you are currently reading
* [gem\_miner\_args_helper](https://github.com/tjayada/sparse_network_analysis/blob/main/gem_miner_args_helper.py) - File for processing the configfiles of Gem-Miner
* [loader](https://github.com/tjayada/sparse_network_analysis/blob/main/loader.py) - File for loading functions, such as models or test accuracies
* [main](https://github.com/tjayada/sparse_network_analysis/blob/main/main.py) - Is the file calling the main files of the pruning algorithms, depending on the experiment choosen in the configfile
* [synflow\_args_helper](https://github.com/tjayada/sparse_network_analysis/blob/main/synflow_args_helper.py) - File for processing the configfiles of SynFlow
* [syngem_utils](https://github.com/tjayada/sparse_network_analysis/blob/main/syngem_utils.py) - Contains all functions and analysis methods created for this thesis
* [run](https://github.com/tjayada/sparse_network_analysis/blob/main/run.sh) - the file being called to run the main plus config file
* [environment](https://github.com/tjayada/sparse_network_analysis/blob/main/environment.yaml) - Contains the libraries and packages used for making the code in this repo work

## Example
To reproduce the results of this thesis choose your experiment by un-commenting it in the run.sh file and then executing it or alternatively run the main file manually such as :
```
python main.py --config Configs/synflow_fc.yml
``` 
<br>
Depending on the experiment you want to run, you may need to adjust the configfile in the Configs directory accordingly, such as setting the random seed, pruning algorithm or desired sparsity. 

With the sparse neural network obtained, you can choose an analysis notebook to reproduce the results, but the loading functions may need to be adjusted, if the networks are not stored in the respective directories. Also the notebook works best when run in this directory, thus being able to access all models and functions properly. 

For any problems or questions feel free to contact me.

## Credits
### Pruning Algorithms
The pruning algorithms used can be found at the GitHub repositories of [SynFlow](https://github.com/ganguli-lab/Synaptic-Flow) and [Gem-Miner](https://github.com/ksreenivasan/pruning_is_enough) and can also be found in an adjusted version in the respective directories in this repo. The adjustments include usabilty aspects as well as crucial adaptions, for example, letting Gem-Miner prune Fully-Connected models. 

### Analysis Methods
The original analysis methods can be found as implementations in their respective GitHub repositories as well.

* [NNSTD](https://github.com/Shiweiliuiiiiiii/Sparse_Topology_Distance)
* [SaSD](https://github.com/Holleri/code_bachelor_thesis)
* [Convergent Learning](https://github.com/yixuanli/convergent_learning)
