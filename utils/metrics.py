import torch

def IoU_metric(predictions, targets, num_classes):
    """
    Compute per-class Intersection over Union (IoU) given point-wise predictions and targets.
    
    Args:
        predictions (torch.Tensor): Predicted labels for each point.
        targets (torch.Tensor): Ground truth labels for each point.
        num_classes (int): Number of classes.
    
    Returns:
        torch.Tensor: Tensor containing per-class Intersection over Union (IoU) scores.
    """
    class_iou = torch.zeros(num_classes, device=predictions.device)
    class_counts = torch.zeros(num_classes, device=predictions.device)
    
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        class_iou[class_id] = intersection / max(union, 1)
        class_counts[class_id] = (targets == class_id).sum().item()
    
    return class_iou, class_counts