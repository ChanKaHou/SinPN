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

  datas = tf.reshape(features["cifar10"], [-1, 3, 32, 32, 1])
  datas = 0.3*datas[:,0,:,:,:] + 0.59*datas[:,1,:,:,:] + 0.11*datas[:,2,:,:,:]

  # Input Shape: [batch_size, 32, 32, 1]
  # Output Shape: [batch_size, 32, 32, 24]
  conv1 = tf.layers.conv2d(inputs=datas, filters=24, kernel_size=[8, 8], padding="same", activation=active)

  # Input Shape: [batch_size, 32, 32, 24]
  # Output Shape: [batch_size, 16, 16, 24]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

  # Input Shape: [batch_size, 16, 16, 24]
  # Output Shape: [batch_size, 16, 16, 48]
  conv2 = tf.layers.conv2d(inputs=pool1, filters=48, kernel_size=[8, 8], padding="same", activation=active)

  # Input Shape: [batch_size, 16, 16, 48]
  # Output Shape: [batch_size, 8, 8, 48]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

  # Input Shape: [batch_size, 8, 8, 48]
  # Output Shape: [batch_size, 8*8*48]
  pool2_flat = tf.reshape(pool2, [-1, 8*8*48])

  # Input Shape: [batch_size, 8*8*48]
  # Output Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=pool2_flat, units=10, activation=active)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  else:
    predictions = tf.argmax(input=logits, axis=1)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def unpickle(file):
  import pickle
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def main(argv):
  # Load training and eval data
  batch1 = unpickle("./cifar-10-batches-py/data_batch_1")
  batch2 = unpickle("./cifar-10-batches-py/data_batch_2")
  batch3 = unpickle("./cifar-10-batches-py/data_batch_3")
  batch4 = unpickle("./cifar-10-batches-py/data_batch_4")
  batch5 = unpickle("./cifar-10-batches-py/data_batch_5")
  test = unpickle("./cifar-10-batches-py/test_batch")
  data = np.r_[batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data'], test[b'data']]
  labels = np.r_[batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels'], test[b'labels']]

  train_data = np.asarray(data/255.0, dtype=np.float32) #(50000 + 10000, 3072)
  train_labels = np.asarray(labels, dtype=np.int32) #(50000 + 10000,)

  eval_data = np.asarray(test[b'data']/255.0, dtype=np.float32) #(10000, 3072)
  eval_labels = np.asarray(test[b'labels'], dtype=np.int32) #(10000,)

  # Create the Estimator
  cifar10_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./checkpoint/tmp")

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"cifar10": train_data},
    y=train_labels,
    batch_size = 100,
    num_epochs=None,
    shuffle=True)
  cifar10_classifier.train(input_fn=train_input_fn, steps=50000)
  #cifar10_classifier.train(input_fn=train_input_fn)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"cifar10": eval_data}, #eval_data, train_data
    y=eval_labels, #eval_labels, train_labels
    num_epochs=1,
    shuffle=False)
  eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
