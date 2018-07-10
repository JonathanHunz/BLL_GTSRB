import tensorflow as tf


def get_number_of_records(filename):
    count = 0
    for record in tf.python_io.tf_record_iterator(filename):
        count += 1

    return count
