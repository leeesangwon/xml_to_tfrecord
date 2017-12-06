"""
Convert images and xml files to tfrecords.

Args:
    data_path:
        Path to the images and xmls input.
    output_tfrecord_name:
        Name of the output TFRecords.
        Output tfrecords would be produced in data_path.
        ex) {data_path}/{output_tfrecord_name}_train_0.tfrecord
            {data_path}/{output_tfrecord_name}_validation_0.tfrecord
            {data_path}/{output_tfrecord_name}_test.tfrecord
    clear_temp_files:
        Option to clear temporary files or not.
        Default is False(not clear).

Usage:
    python xml_to_tfrecord.py --data_path=/path/to/top/of/data/folder --output_tfrecord_name=outputName
    If you want to clear tmp files
    python xml_to_tfrecord.py --data_path=/path/to/top/of/data/folder --output_tfrecord_name=outputName --clear_temp_files=True

Example:
    python xml_to_tfrecord.py ^
        --data_path='C:/Projects/Medical_image/Endoscopic/DATA_edit/detection_1127_tmp/DATA_A/' ^
        --output_tfrecord_name='medical_A'
"""
import os
import tensorflow as tf

import preprocess_dataset
import xml_to_csv
import csv_to_tfrecord

flags = tf.app.flags
flags.DEFINE_string('data_path', None, 'Path to the images input')
flags.DEFINE_string('output_tfrecord_name', None, 'Name of the output TFRecord')
flags.DEFINE_string('clear_temp_files', False, 'Option to clear temp files or not. defalut: False')
FLAGS = flags.FLAGS


def main(_):
    data_path = os.path.abspath(FLAGS.data_path)
    output_tfrecord_name = FLAGS.output_tfrecord_name
    clear_temp_files = FLAGS.clear_temp_files
    if data_path is None:
        print("ERROR: data_path shoud be defined.")
        return
    if output_tfrecord_name is None:
        print("ERROR: output_tfrecord_name should be defined.")
        return
    if clear_temp_files:
        print("Warning: Temporarily produced csv files will be removed.")

    trainval_path = os.path.join(data_path, 'data')
    test_path = os.path.join(data_path, 'test')

    preprocess_dataset.run(trainval_path, is_test_data=False)
    preprocess_dataset.run(test_path, is_test_data=True)
    
    csv_name = output_tfrecord_name
    num_of_cross_val = 5

    xml_to_csv.run(data_path, csv_name, num_of_cross_val)
    csv_to_tfrecord.run(data_path, csv_name, output_tfrecord_name, num_of_cross_val)

    if clear_temp_files:
        remove_temp_csv_files(data_path, csv_name, num_of_cross_val)


def remove_temp_csv_files(data_path, csv_name, num_of_cross_val):
    csv_path = os.path.join(data_path, csv_name)
    
    # Remove trainval csv
    csv_path_pattern = csv_path + '_%s_%d.csv'
    for i in range(num_of_cross_val):
        for trainval in ['train', 'validation']:
            os.remove(csv_path_pattern % (trainval, i))
    # Remove test csv
    os.remove(csv_path + '_test.csv')

    print("Successfully removed temporary csv files.")


if __name__ == '__main__':
    tf.app.run()
