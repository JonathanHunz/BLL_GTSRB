import time
import tensorflow as tf
from src.data import BatchServer

class FeedForward():
    def __init__(self, name, batch_size, learning_rate, iterations, train_capacity, test_capacity):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.train_capacity = train_capacity
        self.test_capacity = test_capacity

        self.log_path = "../logs/{}/".format(name) + time.strftime("%Y-%m-%d_%H-%M-%S")

    def build_graph(self):
        # Create input and output placeholders
        X = tf.placeholder(tf.float32, [None, 30 * 30 * 3], "X")
        Y = tf.placeholder(tf.float32, [None, 43], "Y")

        # Initialize weights
        w_h1 = tf.Variable(tf.random_normal([30 * 30 * 3, 2000], stddev=0.01), name="w_h1")  # Layer 1
        w_h2 = tf.Variable(tf.random_normal([2000, 300], stddev=0.01), name="w_h2")  # Layer 2
        w_h3 = tf.Variable(tf.random_normal([300, 60], stddev=0.01), name="w_h3")  # Layer 2
        w_o = tf.Variable(tf.random_normal([60, 43], stddev=0.01), name="w_o")  # Output layer

        # Define model
        with tf.name_scope("hidden_1"):
            X = tf.nn.dropout(X, 0.7)
            h1 = tf.nn.relu(tf.matmul(X, w_h1))
        with tf.name_scope("hidden_2"):
            h1 = tf.nn.dropout(h1, 0.5)
            h2 = tf.nn.relu(tf.matmul(h1, w_h2))
        with tf.name_scope("hidden_3"):
            h2 = tf.nn.dropout(h2, 0.5)
            h3 = tf.nn.relu(tf.matmul(h2, w_h3))
        with tf.name_scope("output"):
            h3 = tf.nn.dropout(h3, 0.5)
            p_y = tf.nn.softmax(tf.matmul(h3, w_o))

        # Define cost function
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p_y, labels=Y))
            # Define train operation
            train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)
            # Add summary to cost tensor
            tf.summary.scalar("cost", cost)

        # Accuracy function
        with tf.name_scope("accuracy"):
            #  Count correct predictions
            correct_pred = tf.equal(tf.argmax(p_y, 1), tf.argmax(Y, 1))
            # Calculate average accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        return X, Y, train, accuracy

    def run(self):
        # Read input data
        trainData = BatchServer("../data/gtsrb-train.tfrecords", self.batch_size)
        testData = BatchServer("../data/gtsrb-test.tfrecords", 800)

        # Build graph
        X, Y, train, accuracy = self.build_graph()

        # Create session
        with tf.Session() as sess:
            # Create log writer
            writer = tf.summary.FileWriter(self.log_path, sess.graph)
            merged_summary = tf.summary.merge_all()

            # Initialize variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            # Create training coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(self.iterations):

                # Get next batch of img and labels
                image_batch, label_batch = sess.run(trainData.next_batch())

                # Run training
                sess.run(train, feed_dict={X: image_batch, Y: label_batch})

                summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: image_batch, Y: label_batch})

                # Write summary
                writer.add_summary(summary, i)

                test_images, test_labels = sess.run(testData.next_batch())

                # Output accuracy 100 times in a full run
                if i % (self.iterations * 0.01) == 0:
                    print('Step {:02d} \t Accuracy: {:f} \t Test Accuracy: {:f}'.format(i, acc, accuracy.eval(feed_dict={X: test_images, Y: test_labels})))



            coord.request_stop()
            coord.join(threads)
