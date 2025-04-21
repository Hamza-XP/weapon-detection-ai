import numpy as np
import cv2

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    line_thickness=4):

    for i in range(min(boxes.shape[0], max_boxes_to_draw)):
        if scores[i] > min_score_thresh:
            box = boxes[i]
            class_name = category_index[classes[i]]['name'] if classes[i] in category_index else 'N/A'
            display_str = f'{class_name}: {int(100*scores[i])}%'

            if use_normalized_coordinates:
                height, width, _ = image.shape
                box = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
            else:
                box = list(map(int, box))

            cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), line_thickness)
            cv2.putText(image, display_str, (box[1], box[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
