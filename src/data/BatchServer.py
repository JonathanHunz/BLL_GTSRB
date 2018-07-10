import tensorflow as tf


class BatchServer:
    def __init__(self, filename, image_length, label_length, image_type, batch_size):
        # Create reader and read serialized example from files in filename_queue
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([filename])
        _, serialized_example = reader.read(filename_queue)

        feature_set = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

        # Read features
        features = tf.parse_single_example(serialized_example, features=feature_set)
        label_feature = features['label']
        image_feature = features['image']


        # One hot encode labels
        label_feature = tf.one_hot(label_feature, label_length)

        # Decode and reshape image feature
        image_feature = tf.decode_raw(image_feature, image_type)
        image_feature = tf.reshape(image_feature, [image_length])

        self.images, self.labels = tf.train.shuffle_batch([image_feature, label_feature], batch_size=batch_size,
                                                          capacity=100000, num_threads=2,
                                                          min_after_dequeue=int(batch_size/2))

    def next_batch(self):
        return [self.images, self.labels]
