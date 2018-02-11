"""
Convert images and csv files to tfrecords
"""
from __future__ import division, print_function

import os, sys
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
sys.path.append('C:/Projects/tf/models/research')
from object_detection.utils import dataset_util


class InvalidFileNameError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class InvalidClassNameError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def run(input_path, csv_name, output_name, num_of_cross_val=5):
    csv_path = os.path.join(input_path, csv_name)
    output_path = os.path.join(input_path, output_name)

    # trainval_path = os.path.join(input_path, 'data')
    # _gen_tfrecords_for_trainval(trainval_path, csv_path, output_path, num_of_cross_val)
    
    test_path = os.path.join(input_path, 'test')
    _gen_tfrecords_for_test(test_path, csv_path, output_path)

    print('Successfully converted images and csv to tfrecord.')


def _gen_tfrecords_for_trainval(image_path, csv_path, output_path, num_of_cross_val):
    csv_path_pattern = csv_path + '_%s_%d.csv'
    output_path_pattern = output_path + '_%s_%d.tfrecord'

    for i in range(num_of_cross_val):
        for trainval in ['train', 'validation']:
            csv_input = csv_path_pattern % (trainval, i)
            output_path = output_path_pattern % (trainval, i)
            writer = tf.python_io.TFRecordWriter(output_path)
            examples = pd.read_csv(csv_input)
            for index, row in examples.iterrows():
                tf_example = _create_tf_example(row, image_path)
                writer.write(tf_example.SerializeToString())

            writer.close()


def _gen_tfrecords_for_test(image_path, csv_path, output_path):
    csv_input = csv_path + '_test.csv'
    output_path = output_path + '_test.tfrecord'

    writer = tf.python_io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_input)
    for index, row in examples.iterrows():
        tf_example = _create_tf_example(row, image_path)
        writer.write(tf_example.SerializeToString())

    writer.close()


def _create_tf_example(row, img_input):
    if _is_benign(row):
        folder_name = 'benign'
    elif _is_cancer(row):
        folder_name = 'cancer'
    else:
        raise InvalidFileNameError("Invalid Filename")
    feature_dict = {}
    for i in range(1, 4):
        full_path = os.path.join(img_input, folder_name, '{}'.format(row['filename1']))
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = row['filename%s' % i].encode('utf8')
        channels = row['channels%s' % i]
        shape = [int(height), int(width), int(channels)]
        image_format = b'jpg'
        xmins = [row['xmin%s' % i] / width]
        xmaxs = [row['xmax%s' % i] / width]
        ymins = [row['ymin%s' % i] / height]
        ymaxs = [row['ymax%s' % i] / height]
        classes_text = [row['class%s' % i].encode('utf8')]
        classes = [_class_text_to_int(row['class%s' % i])]
        difficult = [0]
        truncated = [0]
        feature_dict.update({
            'image/height%s' % i: dataset_util.int64_feature(height),
            'image/width%s' % i: dataset_util.int64_feature(width),
            'image/filename%s' % i: dataset_util.bytes_feature(filename),
            'image/source_id%s' % i: dataset_util.bytes_feature(filename),
            'image/channels%s' % i: dataset_util.int64_feature(channels),
            'image/shape%s' % i: dataset_util.int64_list_feature(shape),
            'image/class%s' % i: dataset_util.int64_list_feature(classes),
            'image/object/bbox/xmin%s' % i: dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax%s' % i: dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin%s' % i: dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax%s' % i: dataset_util.float_list_feature(ymaxs),
            'image/object/class/text%s' % i: dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label%s' % i: dataset_util.int64_list_feature(classes),
            'image/object/difficult%s' % i: dataset_util.int64_list_feature(difficult),
            'image/object/truncated%s' % i: dataset_util.int64_list_feature(truncated),
            'image/object/bbox/label%s' % i: dataset_util.int64_list_feature(classes),
            'image/object/bbox/label_text%s' % i: dataset_util.bytes_list_feature(classes_text),
            'image/object/bbox/difficult%s' % i: dataset_util.int64_list_feature(difficult),
            'image/object/bbox/truncated%s' % i: dataset_util.int64_list_feature(truncated),
            'image/format%s' % i: dataset_util.bytes_feature(image_format),
            'image/encoded%s' % i: dataset_util.bytes_feature(encoded_jpg),
        })

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return tf_example


def _is_benign(row):
    return _is_valid_file(row, ['u', 'b'], 'Benign')


def _is_cancer(row):
    return _is_valid_file(row, ['c', 'm'], 'Malignant')


def _is_valid_file(row, filename_head, classname):
    is_filename_valid = (row['filename1'][0].lower() in filename_head)
    is_class_valid = (row['class1'] == classname)
    return is_filename_valid and is_class_valid


def _class_text_to_int(row_label):
    if row_label == 'Benign':
        return 1
    if row_label == 'Malignant':
        return 2
    else:
        raise InvalidClassNameError("Class name is not Benign or Malignant")


if __name__ == '__main__':
    input_path = 'C:/Projects/xml_to_tfrecord/data/0'
    csv_name = 'medical_0'
    output_name = 'medical_0'
    num_of_cross_val = 5

    run(input_path, csv_name, output_name, num_of_cross_val)