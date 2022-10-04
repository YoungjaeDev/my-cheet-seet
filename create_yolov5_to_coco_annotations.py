def create_yolov5_to_coco_annotations(img_dir, label_dir, out_dir):
    licenses = [
        {
            "name": "",
            "id": 0,
            "url": ""
        }
    ]

    info_ = [
        {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": ""
        }
    ]

    categories = [
        {
            "id": 1,
            "name": "Buoy",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "Sailing yacht",
            "supercategory": ""
        },
        {
            "id": 3,
            "name": "Boat",
            "supercategory": ""
        },
        {
            "id": 4,
            "name": "Structure",
            "supercategory": ""
        },
        {
            "id": 5,
            "name": "Other",
            "supercategory": ""
        },
        {
            "id": 6,
            "name": "Channel Marker",
            "supercategory": ""
        },
        {
            "id": 7,
            "name": "Speed Warning Sign",
            "supercategory": ""
        },
        {
            "id": 8,
            "name": "Extra boat",
            "supercategory": ""
        }
    ]

    img_idx = 0
    annot_idx = 0

    imgs_list = []
    annots_list = []

    for label_file in os.listdir(label_dir):
        label_file_ = os.path.join(label_dir, label_file)
        img_file_ = os.path.join(img_dir, f'{label_file.split(".")[0]}.jpg')
        img = Image.open(img_file_)
        image_w, image_h = img.size

        imgs_list.append({
            'id': img_idx,
            'width': image_w,
            'height': image_h,
            'file_name': f'{label_file.split(".")[0]}.jpg',
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

        with open(label_file_, 'r') as label_f:
            labels = label_f.readlines()

            for label in labels:
                cat, xc, yc, label_normalized_w, label_normalized_h = list(map(lambda x: int(x) if len(x) == 1 else float(x), label.split()))
                label_w, label_h = image_w * label_normalized_w, image_h * label_normalized_h
                xmin, ymin = (image_w * xc) - (label_w / 2), (image_h * yc) - (label_h / 2)
                
                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin

                annots_list.append({
                    'id': annot_idx,
                    'image_id': img_idx,
                    'category_id': cat + 1,
                    'area': int(label_h * label_w),
                    'bbox': [
                        xmin,
                        ymin,
                        label_w,
                        label_h
                    ],
                    'iscrowd': 0,
                    'attributes': {
                        'type': '',
                        'occluded': False
                    }
                })

                annot_idx += 1

        img_idx += 1

    out_dict = {
        'licenses': licenses,
        'info': info_,
        'categories': categories,
        'images': imgs_list,
        'annotations': annots_list
    }

    with open(os.path.join(out_dir, 'val.json'), 'w') as out_f:
        print(os.path.join(out_dir, 'val.json'))
        json.dump(out_dict, out_f)
