if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
  # 총 N개의 Box, m=i
  iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
  # iou shape = (m, N)
  # scores shape = (1, N)
  weights = iou * scores[None]  # box weights
  # weights shape = (m, N)
  # x[:, :4] shape = (N, 4)
  # weighted mean 적용
  x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
  # 주변 박스가 하나밖에 없는 것들은 제거
  if redundant:
      i = i[iou.sum(1) > 1]  # require redundancy
