import statistics
import torch
import numpy as np

def calculate_iou(pred, target, num_classes):
    """
    Parameters:
    pred (torch.Tensor): The predicted segmentation map, shape (N, H, W).
    target (torch.Tensor): The ground truth segmentation map, shape (N, H, W).
    num_classes (int): Number of classes.
    
    Returns:
    dict: A dictionary with class indices as keys and IoU values as values.
    """
    ious = {}
    pred = pred.view(-1)
    target = target.view(-1)
    array_ious = list()
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = (pred_inds & target_inds).float().sum().item()
        union = (pred_inds | target_inds).float().sum().item()
        
        if union == 0:
            array_ious.append(float('nan'))  # If there is no ground truth, IoU is undefined
            ious[cls] = array_ious[cls]
        else:
            array_ious.append(intersection / union)
            ious[cls] = array_ious[cls]
    ious[num_classes] = (array_ious[1]+array_ious[2])/2
    ious[num_classes+1] = statistics.mean(array_ious)
    return ious

# Example usage
if __name__ == "__main__":
    num_classes = 3
    
    # Example prediction and target tensors
    pred = torch.tensor([
        [0, 1, 2],
        [2, 2, 1],
        [0, 1, 0]
    ])
    
    target = torch.tensor([
        [0, 1, 1],
        [2, 0, 1],
        [0, 1, 0]
    ])
    
    ious = calculate_iou(pred, target, num_classes)
    iou_dict = {'1': ious[1], '2': ious[2], 'mIoU': ious[3]}
    print(ious[1])
    print(iou_dict['mIoU'])
    for cls, iou in ious.items():
        print(f"Class {cls}: IoU = {iou:.4f}")