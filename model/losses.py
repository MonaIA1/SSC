import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import List, cast, Iterable, Set, Tuple
from torch import Tensor, einsum
from scipy.ndimage import distance_transform_edt as eucl_distance

#######################################################################
# inverse class frquency weight (ICF) 
class WeightBalancingCE(): 
    def __init__(self, device, **kwargs):
        super(WeightBalancingCE, self).__init__()
        # claculate the normalised inverse weights for each class in the data set based on the class voxel distribution. for each class the weight is (1/class_vox_distriution) 
        # empty = 0.0001, ciel = 0.17 , floor = 0.01, wall = 0.02, window = 0.12, chair = 0.1, bed = 0.03, sofa = 0.05, table= 0.05, tvs = 0.5, furn = 0.01, obj = 0.02
        # class weights
        class_weights = torch.tensor([0.0001, 0.17, 0.01, 0.02, 0.12, 0.1, 0.03, 0.05, 0.05, 0.5, 0.01, 0.02]).to(device)
        self.loss_fun = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.loss_fun(pred, target)
        return loss


#####################################################################
# combine re-sampling and reqularized weightes using clustering based on k=4
class WCE_BalancedClusters(nn.Module): # combine random sampling and balanced weights
    def __init__(self,device, **kwargs):
        super(WCE_BalancedClusters, self).__init__()
        # class weights, used k-means k=4
        #empty = 1 random sampling, ciel = 4 , floor = 2, wall = 2, window = 4, chair = 4, bed =3, sofa = 3, table= 3, tvs = 4, furn = 1, obj = 2
        class_weights = torch.tensor([1.0, 4.0, 2.0, 2.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 1.0, 2.0]).to(device)
        self.loss_fun = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:  # weight for random sampling  
        weights_expanded: Tensor = weight.unsqueeze(1).expand_as(target)
        weighted_target: Tensor = target * weights_expanded
        
        loss = self.loss_fun(pred, weighted_target)
        return loss
######################################################################

# combine re-sampling and reqularized weightes using clustering based on k=3
class WCE_k3Clusters(nn.Module): # combine random sampling and balanced weights
    def __init__(self,device, **kwargs):
        super(WCE_DBSCANClusters, self).__init__()
        # class weights, used k-means k=3
        #empty = 1 random sampling, ciel = 3 , floor = 2, wall = 2, window = 3, chair = 3, bed =3, sofa = 3, table= 3, tvs = 3, furn = 1, obj = 2
        class_weights = torch.tensor([1.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0]).to(device)
        self.loss_fun = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:  # weight for random sampling  
        weights_expanded: Tensor = weight.unsqueeze(1).expand_as(target)
        weighted_target: Tensor = target * weights_expanded
        
        loss = self.loss_fun(pred, weighted_target)
        return loss 


#####################################################################
# cross entropy with re-sampling only
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()
        
    def forward(self, pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:    
        weights_expanded: Tensor = weight.unsqueeze(1).expand_as(target)
        weighted_target: Tensor = target * weights_expanded
        
        loss = self.loss_fun(pred, weighted_target)
        return loss
######################################################################
