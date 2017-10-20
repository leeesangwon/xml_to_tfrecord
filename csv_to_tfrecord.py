"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv --img_input=data/train_img/  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --img_input=data/test_img/  --output_path=test.record
"""
from __future__ import division, print_function

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from models.object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('csv_input', 'DATA_edit/data1013/medical_train_2.csv', 'Path to the CSV input')
flags.DEFINE_string('img_input', 'DATA_edit/data1013/train/', 'Path to the images input')
flags.DEFINE_string('output_path', 'DATA_edit/data1013/medical_train_2.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label == 'Benign':
        return 1
    if row_label == 'Malignant':
        return 2
    else:
        None


def create_tf_example(row, img_input):
    full_path = os.path.join(os.getcwd(), img_input, '{}'.format(row['filename']))
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
        'image/channels': dataset_util.int64_feature(channels),
        'image/shape': dataset_util.int64_list_feature(shape),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/bbox/label': dataset_util.int64_list_feature(classes),
        'image/object/bbox/label_text': dataset_util.bytes_list_feature(classes_text),
        'image/object/bbox/difficult': dataset_util.int64_list_feature(difficult),
        'image/object/bbox/truncated': dataset_util.int64_list_feature(truncated),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples = pd.read_csv(FLAGS.csv_input)
    for index, row in examples.iterrows():
        tf_example = create_tf_example(row, FLAGS.img_input)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()