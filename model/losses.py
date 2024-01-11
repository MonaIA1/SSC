import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
######################################################################
######################################################################

# combine re-sampling and reqularized weightes using clustering based on k=3
class WCE_k3Clusters(nn.Module): # combine random sampling and balanced weights
    def __init__(self,device, **kwargs):
        super(WCE_DBSCANClusters, self).__init__()
        # class weights, used k-means k=3
        #occl_empty = 1, ciel = 3 , floor = 2, wall = 2, window = 3, chair = 3, bed =3, sofa = 3, table= 3, tvs = 3, furn = 1, obj = 2
        class_weights = torch.tensor([1.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 2.0]).to(device)
        self.loss_fun = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:  # for random sampling  
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
