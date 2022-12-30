import argparse
from Experiments import synflow_main
from Experiments import gem_miner_main
from synflow_args_helper import synflow_parser_args
from gem_miner_args_helper import gem_miner_parser_args



if __name__ == '__main__':


    ## Run Experiment ##
    if synflow_parser_args.experiment == 'synflow':
        print("Synflow Experiment selected!")
        synflow_main.run()

    elif gem_miner_parser_args.experiment == 'gem_miner':
        print("Gem-Miner Experiment selected!")
        gem_miner_main.run()

    else: print("No Experiment selected!")


