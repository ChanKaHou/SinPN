'''
SinPN program source code
for research article "SinP[N]: A Fast Convergence Activation Function for CNNs"

Version 1.0
(c) Copyright 2018 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The SinPN program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

The SinPN program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def active(x):
  #return tf.nn.relu(x)
  #return tf.nn.swish(x)
  #return tf.sin(x) + x
  return tf.sin(x) + 1.5*x
  #return tf.sin(x) + 2*x

def cnn_model_fn(features, labels, mode):

  # Input Shape: [batch_size, 28 * 28]
  datas = tf.reshape(features["mnist"], [-1, 28, 28, 1])

  # Input Shape: [batch_size, 28, 28, 1]
  # Output Shape: [batch_size, 28, 28, 8]
  conv1 = tf.layers.conv2d(inputs=datas, filters=8, kernel_size=[7, 7], padding="same", activation=active)

  # Input Shape: [batch_size, 28, 28, 8]
  # Output Shape: [batch_size, 14, 14, 8]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

  # Input Shape: [batch_size, 14, 14, 8]
  # Output Shape: [batch_size, 14, 14, 16]
  conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[7, 7], padding="same", activation=active)

  # Input Shape: [batch_size, 14, 14, 16]
  # Output Shape: [batch_size, 7, 7, 16]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

  # Input Shape: [batch_size, 7, 7, 16]
  # Output Shape: [batch_size, 7 * 7 * 16]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 16])

  # Input Shape: [batch_size, 7 * 7 * 16]
  # Output Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=pool2_flat, units=10, activation=active)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  else:
    predictions = tf.argmax(input=logits, axis=1)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):

  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # (55000, 784)
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32) # (55000,)
  eval_data = mnist.test.images  # (10000, 784)
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32) # (10000,)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./checkpoint/tmp")

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"mnist": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  mnist_classifier.train(input_fn=train_input_fn, steps=55000)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"mnist": eval_data}, #eval_data, train_data
    y=eval_labels, #eval_labels, train_labels
    num_epochs=1,
    shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
