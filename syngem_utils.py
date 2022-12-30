import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


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

                    _ , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True) 
                    
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

                    _ , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True)
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
                
                _ , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True)

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
                    
            _ , counts = np.unique(each_layer[i].state_dict()[key_word].numpy().flatten()  , return_counts=True)
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
        
    
    
    def display(self):
        img = torchvision.utils.make_grid(self.images)
        
        img = img / 2 + 0.5     # unnormalize
        npimg = np.clip(img.numpy(), 0, 1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        print('GroundTruth: ', ' '.join(f'{self.classes[self.labels[j]]:5s}' for j in range(self.batch_size)))
        
        