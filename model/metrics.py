import torch

def comp_IoU(pred, target, mask): # 0.25 occluded empty, 0.5 visible occubied (on surface), 1 occluded occubied

  eps = 1e-8
  tp, fp, fn, tn, inter, union, precision, recall, mask_occl, occl_target, occl_pred = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  
  # expand dimensions of mask tensor to match target and predicted tensors
  # use torch.where() to replace non-zero values with 1 and execlude the surface values
  mask_occl= torch.where(mask == 0.5, torch.tensor(0.0), mask)
  mask_occl = torch.where((mask_occl == 0.25), torch.tensor(1.0), mask_occl)
  
  occl_target = torch.clamp(torch.argmax(target, dim=1), 0, 1)* mask_occl
  occl_pred= torch.clamp(torch.argmax(pred, dim=1), 0, 1)*mask_occl
  
  
  inter = torch.sum(torch.logical_and(occl_target,occl_pred)).item()
  tp = inter
  fp = torch.sum(torch.logical_and(occl_target == 0 , occl_pred > 0)).item()
  fn = torch.sum(torch.logical_and(occl_target > 0, occl_pred == 0 )).item()
  tn = torch.sum(torch.logical_and(occl_target == 0 , occl_pred == 0)).item()
  
  union = tp + fp + fn

  if (tp + fp)> 0:
     precision = tp / (tp + fp)
  else:
     precision = 0
  if (tp + fn)> 0:
     recall = tp / (tp + fn)
  else:
     recall = 0 
  
  comp_iou = inter / (union + eps)
  return comp_iou*100, precision *100, recall*100
  ####################################################################################################
def m_IoU(pred, target):

    mIoU = 0
    result = 0
    total_cl =0
    total_classes = 0
    target_max = target.argmax(dim=1)
    pred_max = pred.argmax(dim=1)
    
    # Get the unique values
    unique_cl = torch.unique(target.argmax(dim=1))
    #print(f'unique_cl= torch.unique(target.argmax(dim=1)){unique_cl}')
    #print(f'total_classes= unique_cl.numel(){unique_cl.numel()}')
    
    # Exclude empty class
    total_cl = unique_cl[unique_cl != 0]
    total_classes= total_cl.numel()
    #print(f'after execluding zero total_classes= total_cl.numel(){total_classes}')
    
    # list to stmIoU values of each class
    seg_lst =[None]*11
    
    # loop over the class while ignoring the empty class at 0
    for cl in range(0,11):
       inter, union, cl_IoU = 0, 0, 0
       tp, fp, tn= 0, 0, 0
       
       inter = torch.sum(torch.logical_and(pred_max == cl+1, target_max == cl+1))
       tp = inter
       fp = torch.sum(torch.logical_and(torch.logical_and(target_max != cl+1,target_max != 0) , pred_max == cl+1)) # eliminate empty class from the evaluation
       fn = torch.sum(torch.logical_and(target_max == cl+1, torch.logical_and(pred_max != cl+1,target_max != 0))) # eliminate empty class from the evaluation
       union = tp + fp + fn

       # avoid nan 
       if (inter == 0):  
         cl_IoU = 0  
       else:
         cl_IoU = (inter / union)*100
      
       seg_lst[cl] = cl_IoU
       #print(f"seg_lst{cl}: {seg_lst[cl]}")
            
       result += cl_IoU
       
    
    # mIoU per batch size
    if (total_classes==0):
        mIoU= 0
    else:
        mIoU = result/total_classes
    
    return mIoU, seg_lst


