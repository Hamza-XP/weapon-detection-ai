# weapon_detection.py
import sys
import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import label_map_util
from utils import visualization_utils as vis_util

class WeaponDetector:
    def __init__(self):
        self.detection_graph = tf.Graph()
        self.category_index = None
        self.sess = None
        
        # Paths to model files
        self.PATH_TO_MODEL = os.path.join(os.path.dirname(__file__), '../model/frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(os.path.dirname(__file__), '../model/label_map.pbtxt')
        self.NUM_CLASSES = 1

        self.load_model()

    def load_model(self):
        with self.detection_graph.as_default():
            # Load frozen TensorFlow model
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # Load label map
            label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)

        # Create TensorFlow session
        self.sess = tf.Session(graph=self.detection_graph)

    def detect(self, image_np):
        """Detect weapons in a numpy image array (RGB format).
        
        Args:
            image_np: numpy array in RGB format
            
        Returns:
            dict: Detection results containing:
                - boxes: List of bounding boxes [ymin, xmin, ymax, xmax]
                - scores: List of confidence scores
                - classes: List of class names
                - annotated_image: JPEG bytes of annotated image
        """
        detection_results = {
            'boxes': [],
            'scores': [],
            'classes': [],
            'annotated_image': None
        }

        with self.detection_graph.as_default():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            # Get tensors from graph
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            # Run inference
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

    def close(self):
        """Clean up TensorFlow session"""
        if self.sess:
            self.sess.close()

def read_image(image_path):
    """Helper function to read and convert images"""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)