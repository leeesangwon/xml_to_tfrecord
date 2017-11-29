# reference /Users/terrycho/dev/workspace/objectdetection/models/object_detection/data_decoders
# reference http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

tfrecord_filename = '../DATA_edit/data1013/medical_train_2.tfrecord'


def readRecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # '''
    keys_to_features = {
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/channels': tf.FixedLenFeature((), tf.int64, 1),
        'image/shape': tf.VarLenFeature(tf.int64),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        # Object boxes and classes.
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(tf.int64),
        'image/object/bbox/label_text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/bbox/difficult': tf.VarLenFeature(tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(tf.int64),
    }

    features = tf.parse_single_example(serialized_example, features=keys_to_features)

    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    encoded = tf.cast(features['image/encoded'], tf.string)
    image_format = tf.cast(features['image/format'], tf.string)
    xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)
    # '''
    # height = tf.Variable(1.0)
    return height, width, encoded, image_format, xmin, ymin, xmax, ymax


def main():
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    height, width, encoded, image_format, xmin, ymin, xmax, ymax \
        = readRecord(filename_queue)
    X = tf.Variable(initial_value=[0., 0., 0., 0.], dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        vheight, vwidth, vencoded, vimage_format, vxmin, vymin, vxmax, vymax = sess.run(
            [height, width, encoded, image_format, xmin, ymin, xmax, ymax])
        print(vheight, vwidth, vimage_format, vxmin.values, vymin.values, vxmax.values, vymax.values)
        image = Image.open(io.BytesIO(vencoded))
        img_tensor = sess.run(tf.image.decode_jpeg(vencoded))
        plt.imshow(img_tensor)
        plt.show()
        img_tensor = sess.run(tf.expand_dims(tf.image.convert_image_dtype(img_tensor, dtype=tf.float32), 0))
        img_tensor = sess.run(tf.squeeze(tf.image.draw_bounding_boxes(img_tensor, [[[vymin.values[0], vxmin.values[0], vymax.values[0], vxmax.values[0]]]])))
        img_tensor = sess.run(tf.cast(tf.multiply(img_tensor,255.0),tf.float32))
        img_tensor = sess.run(tf.subtract(tf.multiply(tf.ones_like(img_tensor),255), img_tensor))
        #Image.fromarray(img_tensor).show()
        plt.imshow(img_tensor)
        plt.show()

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
