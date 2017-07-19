import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import azure_blob_helper

SAVE_DIR = "/data/mnist/checkpoints/"

tf.app.flags.DEFINE_integer('model_version', 2, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

class Model:
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, w) + b

  values, indices = tf.nn.top_k(y, 10)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in xrange(10)]))
  prediction_classes = table.lookup(tf.to_int64(indices))

  def train(self):
    # Import training data
    mnist = input_data.read_data_sets('/app/MNIST_data/', one_hot=True)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    tf.global_variables_initializer().run()

    # Train
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(self.sess.run(accuracy, feed_dict={self.x: mnist.test.images,
                                        y_: mnist.test.labels}))

  def predict(self, x):
    feed_dict = {self.x: x}
    prediction = self.sess.run(tf.nn.softmax(self.y), feed_dict)
    return prediction

  def save(self, toblob = False):
    if os.path.isdir(SAVE_DIR) == False:
      os.makedirs(SAVE_DIR)
    saver = tf.train.Saver()
    save_path = saver.save(self.sess, os.path.join(SAVE_DIR, "model"))
    print("Model saved in file: %s" % save_path)
    if toblob:
      azure_blob_helper.upload_checkpoint_files(SAVE_DIR)
      print("Model saved to blob")

  def export(self, toblob = False):

    tf.global_variables_initializer().run()

    # Export model to tensorflow serving
    export_path = os.path.join(SAVE_DIR, str(FLAGS.model_version))
    print("Exporting trained model to %s" % export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(
        self.serialized_tf_example)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
        self.prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(self.values)

    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classification_inputs
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classification_outputs_classes,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    classification_outputs_scores
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(self.x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(self.y)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'scores': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        self.sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()
    print("Done exporting!")

  def restore(self, fromblob = False):
    if os.path.isdir(SAVE_DIR) == False:
      os.makedirs(SAVE_DIR)
    if fromblob:
      azure_blob_helper.download_checkpoint_files(SAVE_DIR)
    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph(os.path.join(save_dir, "model.meta"))
    saver.restore(self.sess, os.path.join(SAVE_DIR, "model"))
    print("Model restored from: %s" % os.path.join(SAVE_DIR, "model"))

