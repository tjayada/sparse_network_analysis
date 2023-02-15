import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import networkx as nx

torch.manual_seed(21)
np.random.seed(21)



class get_images_cifar10(object):
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        
    
        normalize = transforms.Normalize(
                    mean=[0.491, 0.482, 0.447],
                    std=[0.247, 0.243, 0.262]
                )

        transform_train = transforms.Compose(
                            [
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]
                        )

        transform_test = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                normalize
                            ]
                        )

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=1)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=1)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        grid_sizes = {'16': []}        
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.classes = classes

        
        self.get_new_images(output = True)
   
    
    def get_new_images(self, target: str = "random", output: bool = False):
        # get some random training images
        if target == "random":
            dataiter = iter(self.trainloader)
            images, labels = next(dataiter)

            self.images = images
            self.labels = labels
            
        elif target in self.classes:
            target_images = torch.zeros((self.batch_size, 3, 32, 32))
            counter = 0
            correct_label = None
            dataiter = iter(self.trainloader)

            while counter < self.batch_size:
                images, labels = next(dataiter)
                
                for i, j in enumerate(labels):
                    labl = self.classes[j]
                    
                    if target == labl:
                        if correct_label == None:
                            correct_label = j  
                        target_images[counter] = images[i]
                        counter += 1
                        
                        
            self.images = target_images
            self.labels = [correct_label] * self.batch_size
                
            
        
        if output:
            return self.images, self.labels
        
    
    
    def display(self, size = (5,5), unique = False):

        good_grid = make_good_grid(self.batch_size, unique)
        
        if self.batch_size == 1:
            img = self.images[0]
        
            img = img / 2 + 0.5     # unnormalize
            npimg = np.clip(img.numpy(), 0, 1)

            plt.subplots(figsize = size)
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.suptitle(self.classes[self.labels[0]])
            plt.axis('off')
            plt.show()

        else:
            fig, axs = plt.subplots(nrows=good_grid[1], ncols=good_grid[0], figsize=size)

            for idx,ax in enumerate(axs.flat):
                img = self.images[idx]
                img = img / 2 + 0.5     # unnormalize
                npimg = np.clip(img.numpy(), 0, 1)
                ax.set_title(self.classes[self.labels[idx]])
                ax.imshow(np.transpose(npimg, (1, 2, 0)))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            fig.tight_layout()
            plt.show()



class get_images_mnist(object):
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        
    
        normalize = transforms.Normalize((0.1307,), (0.3081,))

        transform_train = transforms.Compose(
                            [
                                #transforms.RandomCrop(32, padding=4),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]
                        )

        transform_test = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                normalize
                            ]
                        )

        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=1)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=1)

        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        grid_sizes = {'16': []}        
        
        self.trainloader = trainloader
        self.testloader = testloader
        self.classes = classes

        
        self.get_new_images(output = True)
   
    
    def get_new_images(self, target: str = "random", output: bool = False):
        # get some random training images
        if target == "random":
            dataiter = iter(self.trainloader)
            images, labels = next(dataiter)

            self.images = images
            self.labels = labels
            
        elif target in self.classes:
            target_images = torch.zeros((self.batch_size, 3, 32, 32))
            counter = 0
            correct_label = None
            dataiter = iter(self.trainloader)

            while counter < self.batch_size:
                images, labels = next(dataiter)
                
                for i, j in enumerate(labels):
                    labl = self.classes[j]
                    
                    if target == labl:
                        if correct_label == None:
                            correct_label = j  
                        target_images[counter] = images[i]
                        counter += 1
                        
                        
            self.images = target_images
            self.labels = [correct_label] * self.batch_size
                
            
        
        if output:
            return self.images, self.labels
        
    
    
    def display(self, size = (5,5), unique = False):

        good_grid = make_good_grid(self.batch_size, unique)
        
        if self.batch_size == 1:
            img = self.images[0]
        
            img = img / 2 + 0.5     # unnormalize
            npimg = np.clip(img.numpy(), 0, 1)

            plt.subplots(figsize = size)
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.suptitle(self.classes[self.labels[0]])
            plt.axis('off')
            plt.show()

        else:
            fig, axs = plt.subplots(nrows=good_grid[1], ncols=good_grid[0], figsize=size)

            for idx,ax in enumerate(axs.flat):
                img = self.images[idx]
                img = img / 2 + 0.5     # unnormalize
                npimg = np.clip(img.numpy(), 0, 1)
                ax.set_title(self.classes[self.labels[idx]])
                ax.imshow(np.transpose(npimg, (1, 2, 0)))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            fig.tight_layout()
            plt.show()



def make_good_grid(batch_size, unique = False):

    ''' creates a grid to better display images
    '''

    if not unique:
        target = batch_size
    else:
        target = unique
    
    candidates = []
    for i in range(target + 1):
        for j in range(target + 1):
            if i * j == target:
                candidates.append([i,j])

    for i,j in candidates:
        result = abs(i - j)
        if result <= target:
            target = result
            best_result = i,j

    return best_result


# author: Kees @ https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model/69544742#69544742
def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children



def check_sparsity(model: torch.nn.Module, layers: int = None, single: bool = False, relative: bool = False):
    """Return the sparsity percentage of a given artificial neural network.

    Keyword arguments:
    model -- the model of which the sparsity should be calculated (no default)
    layers -- a list of integers which specify the layers that should be investigated, if None, every layer is selected (default None)
    single -- a boolean, if True, returns the sparsity of the single layer(s) parsed (default False)
    relative -- a boolean, if True, returns the sparsity percentages relative to the whole model (default False)
    """
    
    # get a list of all the layers in the model
    each_layer = get_children(model)
    
    # layers that do not contribute to sparsity
    banned_layers = ["Identity2d()",
                     "ReLU()",
                     "Flatten(start_dim=1, end_dim=-1)",
                     "Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)", 
                     "Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)"]

    # declare empty sparsities dictionary
    sparsities = {}
    
    # declare counter variables
    all_zeros , all_ones = 0 , 0
        
    # if list of layers is passed, we iterate over them    
    if layers != None:
        for i in layers:
            # if single arg is True, the sparsity of each passed layer is calculated while differentiating between the different keywords "flag" and "weight_mask"
            # also if relative arg is True, then the sparsity percentages will be relative to the whole model, thus summing up to the overall model sparsity
             if single:
                if "running_mean" not in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                    if "flag" in each_layer[i].state_dict():
                        key_word = "flag"
                    elif "weight_mask" in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                        key_word = "weight_mask"

                    arr , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True) 
                    
                    # if the array with unique values is 1, 
                    # then all weights must be pruned or none of the weights are pruned
                    # thus fixing the calc by hand
                    if len(arr) == 1:
                        if int(arr[0]) == 1:
                            if relative:
                                calc = counts[0]
                            else:
                                calc = 100
                        
                        elif int(arr[0]) == 0:
                            calc = 0
                    
                    else:
                        if relative:
                            calc = counts[1]
                        else:
                            calc = (counts[1] / (counts[0] + counts[1])) * 100     
                              
             # if single arg is False, the sparsity of all parsed layers is calculated at once, while differentiating between the different keywords "flag" and "weight_mask"
             elif not single:
                if "running_mean" not in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                    if "flag" in each_layer[i].state_dict():
                        key_word = "flag"
                    elif "weight_mask" in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                        key_word = "weight_mask"

                    arr , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True)
                    
                    # if the array with unique values is 1, 
                    # then all weights must be pruned or none of the weights are pruned
                    # thus fixing the calc by hand
                    if len(arr) == 1:
                        if int(arr[0]) == 1:
                                all_zeros += 0
                                all_ones += counts[0]
                            
                        elif int(arr[0]) == 0:
                            all_zeros += counts[0]
                            all_ones += 0
                    
                    else:
                        all_zeros += counts[0]
                        all_ones += counts[1]
                        
        
        # if single arg is False, then the sparsity of multiple layers must be calculated, thus using the all_ones and all_zeros variables to calculate after for loop above is done        
        if not single:
            calc = (all_ones / (all_zeros + all_ones)) * 100
            sparsities[f"selected_layers_{layers}"] = round(calc , 3)

    # if no layers are passed, that means every layer is selected, combined with single arg True this means iterating over all layers and saving single layers sparsities        
    elif layers == None and single:
        for i in range((len(each_layer))):
            if "running_mean" not in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                if "flag" in each_layer[i].state_dict():
                    key_word = "flag"
                elif "weight_mask" in each_layer[i].state_dict():
                    key_word = "weight_mask"
                
                arr , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True)
                
                # if the array with unique values is 1, 
                # then all weights must be pruned or none of the weights are pruned
                # thus fixing the calc by hand
                if len(arr) == 1:
                        if int(arr[0]) == 1:
                            if relative:
                                calc = counts[0]
                            else:
                                calc = 100
                        
                        elif int(arr[0]) == 0:
                            calc = 0
                    
                else:
                    if relative:
                        calc = counts[1]
                    else:
                        calc = (counts[1] / (counts[0] + counts[1])) * 100
                        
                sparsities[f"layer_{i}"] = round(calc , 3)
            

    # declaring the overall sparsity which will always be part of the dictionary regardles of the parsed args    
    sparsities["overall_sparsity"] = 0

    # declaring the counter variables again for the overall sparsity
    all_zeros = 0
    all_ones = 0

    # iterate over all layers and calculate overall model sparsity
    for i in range(len(each_layer)):
        if "running_mean" not in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
            if "flag" in each_layer[i].state_dict():
                key_word = "flag"
            elif "weight_mask" in each_layer[i].state_dict():
                key_word = "weight_mask"
                    
            arr , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True)
            
            if len(arr) == 1:
                if int(arr[0]) == 1:
                        all_zeros += 0
                        all_ones += counts[0]

                elif int(arr[0]) == 0:
                    all_zeros += counts[0]
                    all_ones += 0
                    
            else:
                all_zeros += counts[0]
                all_ones += counts[1]

    calc = (all_ones / (all_zeros + all_ones)) * 100
    sparsities["overall_sparsity"] = round(calc , 3)
    
    # if single and relative args are True, then all the single layers sparsities will be made relative to the whole model
    if single and relative:
        for i, j in sparsities.items():
             if i != "overall_sparsity":
                sparsities[i] = round((j / (all_zeros + all_ones)) * 100, 3)
            
    
    return sparsities



def get_filters(model: torch.nn.Module, layers: int = None):
    """
    Get the units or filters of the model passed
    """
    
    # get a list of all the layers in the model
    each_layer = get_children(model)
    
    # layers that do not contribute to sparsity
    banned_layers = ["Identity2d()",
                     "ReLU()",
                     "Flatten(start_dim=1, end_dim=-1)",
                     "Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)", 
                     "Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)",
                     ]

    # declare empty list for all filters in which single filter will be appended
    all_filters = []
    
        
    # if list of layers is passed, we iterate over them    
    if layers != None:
        for i in layers:
            if "running_mean" not in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                if "flag" in each_layer[i].state_dict():
                    key_word = "flag"
                elif "weight_mask" in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                    key_word = "weight_mask"

                    single_filter = np.array(each_layer[i].state_dict()[key_word]) * np.array(each_layer[i].state_dict()["weight"])

                    all_filters.append(single_filter)   

    # if no layers are passed, that means every layer is selected       
    elif layers == None:
        for i in range((len(each_layer))):
            if "running_mean" not in each_layer[i].state_dict() and str(each_layer[i]) not in banned_layers:
                if "flag" in each_layer[i].state_dict():
                    key_word = "flag"
                elif "weight_mask" in each_layer[i].state_dict():
                    key_word = "weight_mask"
                
                single_filter = np.array(each_layer[i].state_dict()[key_word]) * np.array(each_layer[i].state_dict()["weight"])

                all_filters.append(single_filter) 
                
        
    return all_filters



def get_feature_map(image: np.ndarray, filters: np.ndarray, layers: int = None, dense: str = None):
    
    #### return feature maps of the model passed. model passed as units or filters created by def get_filters(
    assert dense in [None, "last", "all"]
    all_feature_maps = []
    single = False
    
    if isinstance(layers, int):
        layers = [layers]
    
    # take max of layer list 
    if layers == None:
        wanted_layer = len(filters)
    else:
        wanted_layer = max(layers) + 1
        
        if len(layers) == 1:
            single = True
    
    
    for i in range(wanted_layer):

        filter_for_layer = torch.from_numpy(filters[i])
        
        if dense == "all":
            if i == 0:
                first_image = image.flatten()
                feature_maps = torch.nn.functional.linear(first_image, filter_for_layer)
                
            else:
                feature_maps = torch.nn.functional.linear(image, filter_for_layer)
            
            
        elif dense == "last" and i == wanted_layer - 1:
            out = torch.nn.functional.avg_pool2d(image, image.size()[3])
            out = out.view(out.size(0), -1)
            feature_maps = torch.nn.functional.linear(out, filter_for_layer)
        
        else:
            feature_maps = torch.nn.functional.conv2d(image, filter_for_layer, padding=1, dilation=1)

        image = feature_maps
        
        if layers == None:
            if dense == "all":
                all_feature_maps.append(feature_maps.numpy())
            else:   
                all_feature_maps.append(feature_maps[0].numpy())
        elif i in layers:
            if dense == "all":
                all_feature_maps.append(feature_maps.numpy())
            else:   
                all_feature_maps.append(feature_maps[0].numpy())
    
    if single:
        return all_feature_maps[0]     
    
    return all_feature_maps




def get_activation_series(images, filters, dense):
    ''' returns cocatenated feature maps as described by Li et al in paper Convergent Learning
    '''
    
    
    # create list in which all other activation values will be put into
    series_of_activations = []

    # get activation values for first image so we can iterate over them and the
    # flattend feature map values can be appended into series_of_activations
    if images.shape[0] == 1:
        first_activation = get_feature_map(images, filters, dense = dense) 
    else:
        first_activation = get_feature_map(images[0][None], filters, dense = dense)    


    # iterate over all feature maps of first image and flatten them so they can be concatenated with other feature maps later on
    for idx_layer, layer in enumerate(first_activation):
        units_of_layer = []
        for idx_unit, unit in enumerate(layer):
            flat_unit = unit.flatten()
            units_of_layer.append(flat_unit)
        series_of_activations.append(units_of_layer)
 
  
    # iterate over images and call feature map function
    for idx_img, img in enumerate(images, start = 1):
        units_activation = get_feature_map(img[None], filters, dense = dense)

        for idx_layer, layer in enumerate(units_activation):
            for idx_unit, unit in enumerate(layer):
                series_of_activations[idx_layer][idx_unit] = np.concatenate((series_of_activations[idx_layer][idx_unit].flatten(), units_activation[idx_layer][idx_unit].flatten()))

    
    return series_of_activations



def get_correlation(activations_model_1, activations_model_2 = None):
    '''return correlation of two activation series 
    '''

    # For convolutional layers, we compute the mean and standard deviation of each channel. 

    # The mean and standard deviation for a given network and layer is a vector with length equal to 
    # the number of channels (for convolutional layers)
    
    # --> calculate corelation after converging learning paper

    
    if activations_model_2 == None:
        activations_model_2 = activations_model_1
        
    
    number_of_layers = min(len(activations_model_1), len(activations_model_2))
    
    all_cors = []

    for layer in range(number_of_layers):
        layer_cors = np.zeros((len(activations_model_1[layer]), len(activations_model_2[layer])))
        for i in range(len(activations_model_1[layer])):
            for j in range(len(activations_model_2[layer])):
                '''
                calc_mean_i = sum(activations_model_1[layer][i]) / len(activations_model_1[layer][i])
                calc_mean_j = sum(activations_model_2[layer][j]) / len(activations_model_2[layer][j])
                calc_std_i  = np.sqrt(sum((activations_model_1[layer][i] - calc_mean_i) ** 2) / len(activations_model_1[layer][i]))
                calc_std_j  = np.sqrt(sum((activations_model_2[layer][j] - calc_mean_j) ** 2) / len(activations_model_2[layer][j]))

                x_minus_mean_i = activations_model_1[layer][i] - calc_mean_i
                x_minus_mean_j = activations_model_2[layer][j] - calc_mean_j

                cor_i_j = (sum(x_minus_mean_i * x_minus_mean_j) / (len(x_minus_mean_i * x_minus_mean_j))) / (calc_std_i * calc_std_j)
                '''
                # built in function of mean and std way faster but less accurate
                cor_i_j = ((activations_model_1[layer][i] - activations_model_1[layer][i].mean()) * (activations_model_2[layer][j] - activations_model_2[layer][j].mean())).mean() / (activations_model_1[layer][i].std() * activations_model_2[layer][j].std())

                
                if np.isnan(cor_i_j) or np.isinf(cor_i_j):
                    cor_i_j = 0

                layer_cors[i][j] = cor_i_j

        all_cors.append(layer_cors)
        
    return all_cors


def set_to_dic(graph_nodes_as_set):
    
    matching = np.array(list(graph_nodes_as_set))  
    updated_matching = []
    
    set_to_dic = np.zeros(len(matching), dtype=int)
    
    for node in matching:
        
        new_node = [node[0], node[1]]
        
        if node[1] < node[0]:
            new_node = [node[1], node[0]]
            
        updated_matching.append(new_node)
    
    for node in updated_matching:
        for i in range(len(updated_matching)):
            if node[0] == i:
                set_to_dic[i] = node[1]             
    return set_to_dic
    


def find_max_matching(mat, ignore_diag = False):
    # build bipartite graph
    gg = nx.Graph()
    assert mat.shape[0] == mat.shape[1]
    size = mat.shape[0]
    for ii in range(size):
        for jj in range(size):
            if ignore_diag and ii == jj:
                continue
            gg.add_edge(ii, jj+size, weight=mat[ii,jj])

    matching = nx.max_weight_matching(gg, maxcardinality=True)
    
    matching = set_to_dic(matching)
    
    order = np.array([matching[ii]-size for ii in range(size)])
    
    return order


def find_semi_matching(mat, ignore_diag = False):
    ''' for each unit in Net1, we find the unit in Net2 with maximum correlation to it, 
        which is the max along each row
    '''
    assert mat.shape[0] == mat.shape[1]
    size = mat.shape[0]
    
    order = np.zeros(size, dtype=int)
    
    for unit in range(size):
            
            find_max = mat[unit,:].max()
            find_max_index = list(mat[unit,:]).index(find_max)
            
            order[unit] = find_max_index
             
    return order



def change_mat(mat, order):
    """used to create matrix from greedy semi matching"""
    # "The matching is returned as a dictionary, such that matches[v] == w if node v is matched to node w. " - von networkx 
    new_matrix = mat.copy()
    
    for idx, i in enumerate(order):
        new_matrix[:,idx] = mat[:,i].copy()
    
    return new_matrix



def get_image_patch(images, layer, unit, filters, dense, how_many = 1) :
    """similar to get_activation_series but returns single most high correlation oder so"""
    
    #create max cor
    max_cor = float('-inf')

    # if only one image is parsed
    for i in range(how_many):    
        if images.shape[0] == 1:
            first_activation = get_feature_map(images, filters, dense = dense)
            wanted_layer_unit = first_activation[layer][unit]
            
            if  abs(np.amax(wanted_layer_unit)) > abs(np.amin(wanted_layer_unit)):
                max_idx = np.unravel_index(np.argmax(wanted_layer_unit, axis=None), wanted_layer_unit.shape)

            else: 
                max_idx = np.unravel_index(np.argmin(wanted_layer_unit, axis=None), wanted_layer_unit.shape)

            max_img = images

        else:
            # iterate over images and call feature map function
            for idx_img, img in enumerate(images):
                units_activation = get_feature_map(img[None], filters, dense = dense)
                wanted_layer_unit = units_activation[layer][unit]

                if  abs(np.amax(wanted_layer_unit)) > abs(np.amin(wanted_layer_unit)):
                    max_mag = abs(np.amax(wanted_layer_unit))
                else: 
                    max_mag = abs(np.amin(wanted_layer_unit))

                if  max_mag > max_cor:
                    max_cor = np.amax(wanted_layer_unit)
                    max_idx = np.unravel_index(np.argmax(wanted_layer_unit, axis=None), wanted_layer_unit.shape)
                    max_img = img
    
    # normalize image before returning for cleaner notebook appearance
    img = max_img / 2 + 0.5     # unnormalize
    npimg = np.clip(img.numpy(), 0, 1)
    npimg = np.transpose(npimg, (1, 2, 0))
    max_img = np.pad(npimg, ((1,1), (1,1), (0,0)), 'constant', constant_values=((1,1)))
    
    # add 1 to each index for added padding 
    max_idx = [max_idx[0] + 1 , max_idx[1] + 1]
    
    return max_idx, max_img



def order_by_dist(mod1, mod2, dist_measure, fc = False, nED = False):
    """ order a model based on similarity through distance measure
    
    """
    
    mod2_copy = copy.deepcopy(mod2)
      
    for layer in range(len(mod1)):
        # look at layer and create distance matrix for it
        new_mat = np.zeros((len(mod1[layer]), len(mod2[layer])), dtype=float)
    
        for i in range(len(mod1[layer])):
            for j in range(len(mod2[layer])):
                dist = 0
                if nED:
                    dist = dist_measure(np.nonzero(mod1[layer][i].flatten())[0], np.nonzero(mod2_copy[layer][j].flatten())[0])

                else:    
                    dist = dist_measure(mod1[layer][i].flatten(),mod2_copy[layer][j].flatten())

                new_mat[i][j] = dist
        # find best match of units in each network through smallest distance 
        match = find_min_matching(new_mat)
        #print(match)
        # iterate over current layer and change model2 according to best matches order with model1 found above
        mod2_old = copy.deepcopy(mod2_copy)

        for i in range(len(mod2_copy[layer])):
            new_idx_i = match[i]

            mod2_copy[layer][i] =  mod2_old[layer][new_idx_i]


        #"""
        # for each changed unit in layer x, each channel of each unit in layer x + 1 needs to be changed accordingly
        if not fc:
            if layer + 1 < len(mod2_copy) - 1:
                for idx_u, unit in enumerate(mod2_copy[layer + 1]):
                    #print(idx_u)
                    for idx_c, channel in enumerate(unit):
                        new_idx_u = match[idx_c]
                        mod2_copy[layer + 1][idx_u][idx_c] = mod2_old[layer + 1][idx_u][new_idx_u]

    return mod2_copy


def find_dist_matching_semi(mat, ignore_diag = False):
    ''' for each unit in Net1, we find the unit in Net2 with maximum correlation to it, 
        which is the max along each row
    '''
    assert mat.shape[0] == mat.shape[1]
    size = mat.shape[0]
    
    order = np.zeros(size, dtype=int)
    
    for unit in range(size):
            
            find_min = mat[unit,:].min()
            find_min_index = list(mat[unit,:]).index(find_min)
            
            order[unit] = find_min_index
             
    return order



def find_min_matching(mat, ignore_diag = False):
    # build bipartite graph
    gg = nx.Graph()
    assert mat.shape[0] == mat.shape[1]
    size = mat.shape[0]
    for ii in range(size):
        for jj in range(size):
            if ignore_diag and ii == jj:
                continue
            gg.add_edge(ii, jj+size, weight=mat[ii,jj])

    matching = nx.min_weight_matching(gg, maxcardinality=True)
    
    matching = set_to_dic(matching)
    
    order = np.array([matching[ii]-size for ii in range(size)])
    
    return order



def make_table(seed_21, seed_42, seed_63, title):
    assert len(seed_21) == len(seed_42) == len(seed_63)

    df = pd.DataFrame({})
    df[0] = list(seed_21.values())
    df[1] = list(seed_42.values())
    df[2] = list(seed_63.values())

    if len(seed_21) == 7:
        adjust_title = 0.75
        df.index = ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6", "overall_sparsity"]
        
    else:
        adjust_title = None
        df.index = ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", 
                    "layer_6", "layer_7", "layer_8", "layer_9", "layer_10", 
                    "layer_11", "layer_12", "layer_13", "layer_14", "layer_15",
                    "layer_16", "layer_17", "layer_18", "layer_19", "layer_20",
                    "overall_sparsity"]
    
    df.columns = ["seed 21", "seed 42", "seed 63"]


    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=df.values, 
                     rowLabels = df.index,
                     rowColours = plt.cm.BuPu(np.full(len(df.index), 0.1)),
                     colLabels=df.columns,
                     colColours = plt.cm.BuPu(np.full(len(df.columns), 0.1)),
                     loc='center',
                     cellLoc='center')

    ax.set_title(f'{title}', y= adjust_title)
    fig.tight_layout()
    plt.show()
 


def plot_units(units, model, sparse):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(24,6))

    titles = ["21", "42", "63"]
    count = 0

    for ax in axs.flat:
        img = ax.imshow(units[count].reshape((28,28)), cmap = "rainbow")
        fig.colorbar(img)
        ax.set_title(f"{model}     {sparse}% sparsity     seed {titles[count]}")
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        fig.show()
        count += 1



def get_weight_distance(unit):
    
    weight_idxs = np.argwhere(unit)
    
    weight_dist = 0
    for idx in range(len(weight_idxs)):
        if len(weight_idxs) == 1:
            return 0
        try:
            weight_dist += abs(weight_idxs[idx] - weight_idxs[idx + 1])
        except:
            return int(weight_dist / (len(weight_idxs) - 1))
    return np.nan




def get_model_weight_distances(model):
    distances_model = []
    for layer in model:
        distances_layer = []
        for unit in layer:
            dist = get_weight_distance(unit)
            distances_layer.append(dist)
        distances_model.append(distances_layer)
    
    return distances_model



def get_weight_positions(model):
    
    positions_model = []
    for layer in model:
        positions_layer = []
        for unit in layer:
            weight_idxs = np.argwhere(unit.flatten())
            positions_layer = np.concatenate((positions_layer, weight_idxs.flatten()))
        
        positions_model.append(positions_layer)   
    
    return positions_model



def count_clusters(model):
     
    clusters_model = []
    clusters_size_model = []
    clusters_sign_model = []
    
    
    for layer in model:
        clusters_layer = []
        clusters_size_layer = []
        clusters_sign_layer = []
        
        for unit in layer:
            clusters_unit = []
            sign_cluster = []
            seen_sign = []
            
            weight_idxs = np.argwhere(unit.flatten())
            count = 0
            for i in range(len(weight_idxs)):
                try:
                    if int(weight_idxs[i] + 1) == int(weight_idxs[i + 1]):
                        clusters_unit.append(count)
                        
                        if weight_idxs[i] not in seen_sign:
                            if unit[weight_idxs[i]] > 0:
                                sign_cluster.append(1)
                            else:
                                sign_cluster.append(0)
                            
                            seen_sign.append(weight_idxs[i])
                        
                        if weight_idxs[i + 1] not in seen_sign:
                            if unit[weight_idxs[i + 1]] > 0:
                                sign_cluster.append(1)
                            else:
                                sign_cluster.append(0)
                            seen_sign.append(weight_idxs[i + 1])
                        
                        
                    else:
                        count += 1
                        clusters_unit.append(count)
                        if sign_cluster != []:
                            positive_idx = np.argwhere(sign_cluster)
                            if len(positive_idx) == 0:
                                clusters_sign_layer.append(0)
                            else:   
                                clusters_sign_layer.append(len(positive_idx) / len(sign_cluster))
                            sign_cluster = []
                        
                except:
                    pass
                        
            clusters, counts = np.unique(np.array(clusters_unit), return_counts=True)
            
            clusters_layer.append(len(clusters))
            
            if counts != []:
                clusters_size_layer.append(np.round(np.mean(counts)))
            
        
        if clusters_sign_layer == []:
            clusters_sign_layer.append(0)


        clusters_model.append(clusters_layer)   
        clusters_size_model.append(clusters_size_layer)
        clusters_sign_model.append(clusters_sign_layer)
    
    return clusters_model, clusters_size_model, clusters_sign_model





def graph_score_kernel(kernel, cluster_size, distance):
    gg = nx.Graph()
    
    idx_kernel = np.argwhere(kernel)
    size = idx_kernel.shape[0]
    
    if idx_kernel.shape[1] == 1:
        for ii in range(size):
            for jj in range(size):
                if abs(idx_kernel[ii] - idx_kernel[jj]) <= 1:
                    gg.add_edge(ii, jj)
    
    
    else:
        for ii in range(size):
                for jj in range(size):
                    
                    if distance == "manhattan":
                        if (abs(idx_kernel[ii][0] - idx_kernel[jj][0]) + abs(idx_kernel[ii][1] - idx_kernel[jj][1])) <= 1:
                            gg.add_edge(ii, jj)
                    
                    elif distance == "euclidean":
                        if np.sqrt((idx_kernel[ii][0] - idx_kernel[jj][0]) ** 2 + (idx_kernel[ii][1] - idx_kernel[jj][1]) ** 2) <= np.sqrt(2):
                            gg.add_edge(ii, jj)
    
    if len(idx_kernel) == len(list(nx.connected_components(gg))):
        return 0
    
    elif len(list(nx.connected_components(gg))) == 1:
        return 1
    
    
    else:
        big_clust = 0
        small_clust = 0
        for cluster in nx.connected_components(gg):
            if len(cluster) > cluster_size:
                big_clust += 1
            else:
                small_clust += 1
        
        if big_clust == 0:
            return 0
        elif small_clust == 0:
            return 1
    
        return np.round(big_clust / (small_clust + big_clust), 2)
    

def graph_score_model(model, dense = False, cluster_size = 1, distance = "manhattan"):
    
    if dense in ["all", "last"]:
        dense = True
    
    model_scores = []
    count = 0
    for idx, layer in enumerate(model):
        
        filters_scores = []
        if idx == (len(model) - 1) and dense:
            for unit in layer:
                scores = []
                scores.append(graph_score_kernel(unit, cluster_size, distance))
                filters_scores.append(np.mean(scores))
            
        else:
            for filters in layer:
                scores = []
                for kernel in filters:
                    scores.append(graph_score_kernel(kernel, cluster_size, distance))

                filters_scores.append(np.mean(scores))
        model_scores.append(np.round(filters_scores, 2))
    
    return model_scores



def show_common_kernels(model, number_of_tops, display, model_name):
    
    # put all kernels in single list
    final_mat = []

    for i in range(len(model)-1):    
        for j in range(len(model[i])):
            fil = model[i][j].copy()
            fil = np.where(fil != 0, 1, 0)

            for unit in fil:
                final_mat.append(unit.flatten())
    
    values, counts = np.unique(final_mat, axis=0, return_counts=True)
    
    top_kernels = []
    count = 0
    while count < number_of_tops:
        idx = np.argwhere(counts == np.flip(np.sort(counts))[count])
        for i in idx:
            top_kernels.append((values[i], np.flip(np.sort(counts))[count]))
            count += 1
    
    fig, axs = plt.subplots(nrows=display[0], ncols=display[1], figsize=(18,4))
    count = 0
    for ax in axs.flat:
        
        values, counts = top_kernels[count]
        
        ax.imshow(values.reshape(3,3))
    
        ax.set_title(counts)
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    
        count += 1

    fig.suptitle(f"{model_name}", size=16)
    plt.show()
    print(" ")
    print(" ")
    print(" ")
    
    
        