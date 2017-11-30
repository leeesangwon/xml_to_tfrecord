"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv --img_input=data/train_img/  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --img_input=data/test_img/  --output_path=test.record
"""
from __future__ import division, print_function

import os, sys
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
sys.path.append('C:/Projects/tf/models/research')
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('csv_input', 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/medical_A_train_0.csv', 'Path to the CSV input')
flags.DEFINE_string('img_input', 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/data/', 'Path to the images input')
flags.DEFINE_string('output_path', 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/medical_A_train_0.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS


class InvalidFileNameError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def class_text_to_int(row_label):
    if row_label == 'Benign':
        return 1
    if row_label == 'Malignant':
        return 2
    else:
        None


def is_benign(row):
    return is_valid_file(row, ['u', 'b'], 'Benign')


def is_cancer(row):
    return is_valid_file(row, ['c', 'm'], 'Malignant')


def is_valid_file(row, filename_head, classname):
    is_filename_valid = (row['filename'][0].lower() in filename_head)
    is_class_valid = (row['class'] == classname)
    return is_filename_valid and is_class_valid


def create_tf_example(row, img_input):
    if is_benign(row):
        folder_name = 'benign/'
    elif is_cancer(row):
        folder_name = 'cancer/'
    else:
        raise InvalidFileNameError("Invalid Filename")
    full_path = os.path.join(img_input, folder_name, '{}'.format(row['filename']))
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = row['filename'].encode('utf8')
    channels = row['channels']
    shape = [int(height), int(width), int(channels)]
    image_format = b'jpg'
    xmins = [row['xmin'] / width]
    xmaxs = [row['xmax'] / width]
    ymins = [row['ymin'] / height]
    ymaxs = [row['ymax'] / height]
    classes_text = [row['class'].encode('utf8')]
    classes = [class_text_to_int(row['class'])]
    difficult = [0]
    truncated = [0]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/channels': dataset_util.int64_feature(channels),
        'image/shape': dataset_util.int64_list_feature(shape),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/bbox/label': dataset_util.int64_list_feature(classes),
        'image/object/bbox/label_text': dataset_util.bytes_list_feature(classes_text),
        'image/object/bbox/difficult': dataset_util.int64_list_feature(difficult),
        'image/object/bbox/truncated': dataset_util.int64_list_feature(truncated),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    }))
    return tf_example


def main(_):
    FLAGS.img_input = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/data/'
    for i in range(5):
        for trainval in ['train', 'validation']:
            FLAGS.csv_input = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/DATA_A/medical_A_' + trainval + '_' + str(i)+'.csv'
            FLAGS.output_path = 'C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127/medical_A_' + trainval + '_'+str(i)+'.tfrecord'
            writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
            examples = pd.read_csv(FLAGS.csv_input)
            for index, row in examples.iterrows():
                tf_example = create_tf_example(row, FLAGS.img_input)
                writer.write(tf_example.SerializeToString())

            writer.close()


if __name__ == '__main__':
    tf.app.run()