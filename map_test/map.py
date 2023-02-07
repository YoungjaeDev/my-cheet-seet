#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:37:52 2023

@author: yyj
"""

import torch
from collections import Counter

#%%

def mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=0.5,
            box_format="midpoint",
            num_classes=20
        ):
    """
    
    Parameters
    ----------
    pred_boxes : List
        list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2].
    true_boxes : List
        Similar as pred_boxes except all the correct ones.
    iou_threshold : float, optional
        threshold where predicted bboxes is correct. The default is 0.5.
    box_format : str, optional
        "midpoint" or "corners" used to specify bboxes. The default is "midpoint".
    num_classes : int, optional
        number of classes. The default is 20.

    Returns
    -------
    float: mAP value across all classes given a specific IoU threshold .

    """
    
    average_precision = []
    
    epsilon = 1e-6
    
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
                
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        # score대로 sorting 
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes == 0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), 
                                              torch.tensor(gt[3:]), 
                                              box_format)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precision.append(torch.trapz(precisions, recalls))
        
    return sum(average_precision) / len(average_precision)
        
        
#%%
def intersection_over_union(
            boxes_preds,
            boxes_labels,
            box_format="midpoint"
        ):
    """
    
    Parameters
    ----------
    box_preds : torch.Tensor
        Predictions of Bounding Boxes (BATCH_SIZE, 4)
    box_labels : torch.Tensor
        Correct Labels of Boxes (BATCH_SIZE, 4)
    box_format : str, optional
        midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2). The default is "midpoint".

    Returns
    -------
    torch.Tensor: Intersection over union for all examples

    """
    
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
        
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1) 
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    return intersection / (box1_area + box2_area - intersection)
    
#%%

# if __name__ == "__main__":
#     t1_preds = [
#         [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
#         [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
#         [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
#     ]
#     t1_targets = [
#         [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
#         [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
#         [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
#     ]
    
#     mean_average_precision(
#         t1_preds,
#         t1_targets,
#         iou_threshold=0.5,
#         box_format="midpoint",
#         num_classes=1,
#     )