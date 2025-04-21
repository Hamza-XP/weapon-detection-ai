import tensorflow.compat.v1 as tf
from object_detection.utils import label_map_util as tf_label_map_util

def load_labelmap(path):
    with tf.io.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        return tf_label_map_util.load_labelmap(path)

def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    return tf_label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=use_display_name
    )

def create_category_index(categories):
    return tf_label_map_util.create_category_index(categories)
