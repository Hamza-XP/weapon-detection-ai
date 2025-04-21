import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import label_map_util
from utils import visualization_utils as vis_util

class WeaponDetector:
    def __init__(self):
        self.detection_graph = tf.Graph()
        self.category_index = None
        self.sess = None
        
        # Paths based on directory structure
        self.PATH_TO_MODEL = os.path.join(os.path.dirname(__file__), '../model/frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(os.path.dirname(__file__), '../model/label_map.pbtxt')
        self.NUM_CLASSES = 1

        self.load_model()

    def load_model(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)

        self.sess = tf.Session(graph=self.detection_graph)

    def detect(self, image_np):
        detection_results = {
            'boxes': [],
            'scores': [],
            'classes': [],
            'annotated_image': None
        }

        with self.detection_graph.as_default():
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Process results
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # Filter detections with confidence > 50%
            min_score_thresh = 0.5
            for i in range(boxes.shape[0]):
                if scores[i] > min_score_thresh:
                    detection_results['boxes'].append(boxes[i].tolist())
                    detection_results['scores'].append(float(scores[i]))
                    detection_results['classes'].append(
                        self.category_index[classes[i]]['name'])

            # Generate annotated image
            annotated_image = np.copy(image_np)
            vis_util.visualize_boxes_and_labels_on_image_array(
                annotated_image,
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=min_score_thresh)

            # Convert to JPEG bytes
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            detection_results['annotated_image'] = buffer.tobytes()

        return detection_results

def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Example usage for testing
if __name__ == '__main__':
    detector = WeaponDetector()
    test_image = read_image('test_image.jpg')
    results = detector.detect(test_image)
    
    print(f"Detected {len(results['boxes'])} weapons")
    print("Detection details:", results['scores'], results['classes'])
    with open('output.jpg', 'wb') as f:
        f.write(results['annotated_image'])