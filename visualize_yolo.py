import os
import cv2

CLASS_NAME_TO_ID = {'Buoy': 0, 'Boat': 1, 'Channel Marker': 2, 'Speed Warning Sign': 3}
CLASS_ID_TO_NAME = {0: 'Buoy', 1: 'Boat', 2: 'Channel Marker', 3: 'Speed Warning Sign'}
SIZE=(1920,1080)

def visualize_yolo(image, bboxes, category_ids, color=None):
    image = copy.deepcopy(image)
    
    if color is None:
        colors = Colors()
    else:
        colors = None
        
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = CLASS_ID_TO_NAME[category_id[0]]
        
        x_center, y_center, w, h = bbox
        x_min = int((x_center - w/2) * SIZE[0])
        y_min = int((y_center - h/2) * SIZE[1])
        x_max = int((x_center + w/2) * SIZE[0])
        y_max = int((y_center + h/2) * SIZE[1])

        if colors is not None:
            color = colors(category_id[0])
        
        # set rect_th for boxes
        rect_th = max(round(sum(image.shape) / 2 * 0.001), 1)
        # set text_th for category names
        text_th = max(rect_th - 1, 1)
        # set text_size for category names
        text_size = rect_th / 3

        cv2.rectangle(image, 
                      (x_min, y_min),
                      (x_max, y_max), 
                      color=color,
                      thickness=rect_th)
        
        p1, p2 = (int(x_min), int(y_min)), (int(x_max), int(y_max))
        label = f"{class_name}"
        w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
        )
        
    return image

label_files = sorted(glob(yolo_label_dir+"/*.txt"))
image_files = sorted(glob(yolo_image_dir+"/*.jpg"))

@interact(index=(0,len(image_files)))
def verify_gt(index=0):
    image = cv2.imread(image_files[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = label_files[index]
    
    bboxes = []
    category_ids = []
    
    with open(label, 'r') as f:
        
        line = f.readline().strip()
        while line:
            id, cx, cy, w, h = list(map(float, line.split()))
            bboxes.append([cx, cy, w, h])
            category_ids.append([int(id)])
            line = f.readline().strip()

    if len(bboxes) > 0:    
        canvas = visualize_yolo(image, bboxes, category_ids)
        plt.figure(figsize=(16,16))
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()
