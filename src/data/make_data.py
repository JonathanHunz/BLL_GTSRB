import os
import sys
import random
import glob

import tensorflow as tf
import numpy as np

from PIL import Image


def process_image(img, size):
    # Fetch with and height
    width, height = img.size

    if width < size or height < size:
        raise Exception("Image size smaller than thumbnail size")

    # Convert image to greyscale
    # img = img.convert("LA")

    # The longer side of the image is equally cropped on both ends to the same size as the other side. First the
    # difference (delta) between width and height is calculated. Notice that if the image is taller than wide, delta is
    # negative. This is used to decide which side to crop. The number of pixels that need to be removed to achieve a
    # square equals half of the absolute value of the difference.
    delta = width - height
    crop = abs(delta) / 2

    # Crop width if image is wider than tall
    if delta > 0:
        img = img.crop((crop, 0, width - crop, height))
    # Crop height if image is taller than wide
    elif delta < 0:
        img = img.crop((0, crop, width, height - crop))

    # Resize the image to the desired size
    img.thumbnail((size, size))

    return img


def make_data(argv):
    # Read command line arguments
    size = int(argv[1])
    in_path = argv[2]
    out_path = argv[3]

    # Fetch all filenames in the input directory
    in_filenames = glob.glob(in_path + "/**/*.ppm", recursive=True)

    # Initialize TFRecordWriter
    train_writer = tf.python_io.TFRecordWriter(out_path + "/gtsrb-train.tfrecords")
    test_writer = tf.python_io.TFRecordWriter(out_path + "/gtsrb-test.tfrecords")

    # Generate array of random indices that will be written to the test set.
    num_test = int(39209 * 0.15)
    test_indices = random.sample(range(0, len(in_filenames)), num_test)

    # Iterate over all files
    index = 0
    for filename in in_filenames:

        feature_list = {}
        try:
            # Find label in from parent directory name
            label = int(os.path.dirname(filename).split("/")[-1])

            # Read image
            img = Image.open(filename)

            # Perform image processing actions
            img = process_image(img, size)

        except Exception as e:
            print("Could not process file {}: {}".format(filename, e))
            continue

        # Convert image to numpy array of type float32
        image_data = np.array(img).astype(np.uint8)

        if image_data.shape[0] != size or image_data.shape[1] != size:
            print("Shape not correct")
            continue

        # Create image feature and add to feature list
        feature_list["image"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data.tostring()]))

        # Create label feature and add to feature list
        feature_list["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

        # Create example
        example = tf.train.Example(features=tf.train.Features(feature=feature_list))

        # A part of the entries are written to a test set. The proportion of train to test data is given by
        # train_test_prop. All examples with an index in test_indices are written to the test set.
        if index in test_indices:
            test_writer.write(example.SerializeToString())
        else:
            train_writer.write(example.SerializeToString())

        print("Write file: {}".format(filename))

        index += 1

    print("Wrote {} files".format(index))

if __name__ == "__main__":
    make_data(sys.argv)
