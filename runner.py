# Source: https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification#preparing_the_input_data

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# print(len(emnist_train.client_ids))

# print(emnist_train.element_type_structure)

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

example_element = next(iter(example_dataset))

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))

# print(sample_batch)

def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

# print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
# print('First dataset: {d}'.format(d=federated_train_data[0]))

def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# print(str(iterative_process.initialize.type_signature))

state = iterative_process.initialize()

state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

NUM_ROUNDS = 11
for round_num in range(2, NUM_ROUNDS):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

# #@test {"skip": true}
# logdir = "/tmp/logs/scalars/training/"
# summary_writer = tf.summary.create_file_writer(logdir)
# state = iterative_process.initialize()

# #@test {"skip": true}
# with summary_writer.as_default():
#   for round_num in range(1, NUM_ROUNDS):
#     state, metrics = iterative_process.next(state, federated_train_data)
#     for name, value in metrics['train'].items():
#       tf.summary.scalar(name, value, step=round_num)

# #@test {"skip": true}
# !ls {logdir}
# %tensorboard --logdir {logdir} --port=0

# #@test {"skip": true}
# # Uncomment and run this this cell to clean your directory of old output for
# # future graphs from this directory. We don't run it by default so that if 
# # you do a "Runtime > Run all" you don't lose your results.

# # !rm -R /tmp/logs/scalars/*

# MnistVariables = collections.namedtuple(
#     'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

# def create_mnist_variables():
#   return MnistVariables(
#       weights=tf.Variable(
#           lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
#           name='weights',
#           trainable=True),
#       bias=tf.Variable(
#           lambda: tf.zeros(dtype=tf.float32, shape=(10)),
#           name='bias',
#           trainable=True),
#       num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
#       loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
#       accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))

# def predict_on_batch(variables, x):
#   return tf.nn.softmax(tf.matmul(x, variables.weights) + variables.bias)

# def mnist_forward_pass(variables, batch):
#   y = predict_on_batch(variables, batch['x'])
#   predictions = tf.cast(tf.argmax(y, 1), tf.int32)

#   flat_labels = tf.reshape(batch['y'], [-1])
#   loss = -tf.reduce_mean(
#       tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
#   accuracy = tf.reduce_mean(
#       tf.cast(tf.equal(predictions, flat_labels), tf.float32))

#   num_examples = tf.cast(tf.size(batch['y']), tf.float32)

#   variables.num_examples.assign_add(num_examples)
#   variables.loss_sum.assign_add(loss * num_examples)
#   variables.accuracy_sum.assign_add(accuracy * num_examples)

#   return loss, predictions

# def get_local_mnist_metrics(variables):
#   return collections.OrderedDict(
#       num_examples=variables.num_examples,
#       loss=variables.loss_sum / variables.num_examples,
#       accuracy=variables.accuracy_sum / variables.num_examples)

# @tff.federated_computation
# def aggregate_mnist_metrics_across_clients(metrics):
#   return collections.OrderedDict(
#       num_examples=tff.federated_sum(metrics.num_examples),
#       loss=tff.federated_mean(metrics.loss, metrics.num_examples),
#       accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))
  

# from typing import Callable, List, OrderedDict

# class MnistModel(tff.learning.Model):

#   def __init__(self):
#     self._variables = create_mnist_variables()

#   @property
#   def trainable_variables(self):
#     return [self._variables.weights, self._variables.bias]

#   @property
#   def non_trainable_variables(self):
#     return []

#   @property
#   def local_variables(self):
#     return [
#         self._variables.num_examples, self._variables.loss_sum,
#         self._variables.accuracy_sum
#     ]

#   @property
#   def input_spec(self):
#     return collections.OrderedDict(
#         x=tf.TensorSpec([None, 784], tf.float32),
#         y=tf.TensorSpec([None, 1], tf.int32))

#   @tf.function
#   def predict_on_batch(self, x, training=True):
#     del training
#     return predict_on_batch(self._variables, x)
    
#   @tf.function
#   def forward_pass(self, batch, training=True):
#     del training
#     loss, predictions = mnist_forward_pass(self._variables, batch)
#     num_exmaples = tf.shape(batch['x'])[0]
#     return tff.learning.BatchOutput(
#         loss=loss, predictions=predictions, num_examples=num_exmaples)

#   @tf.function
#   def report_local_outputs(self):
#     return get_local_mnist_metrics(self._variables)

#   @property
#   def federated_output_computation(self):
#     return aggregate_mnist_metrics_across_clients

#   @tf.function
#   def report_local_unfinalized_metrics(
#       self) -> OrderedDict[str, List[tf.Tensor]]:
#     """Creates an `OrderedDict` of metric names to unfinalized values."""
#     return collections.OrderedDict(
#         num_examples=[self._variables.num_examples],
#         loss=[self._variables.loss_sum, self._variables.num_examples],
#         accuracy=[self._variables.accuracy_sum, self._variables.num_examples])

#   def metric_finalizers(
#       self) -> OrderedDict[str, Callable[[List[tf.Tensor]], tf.Tensor]]:
#     """Creates an `OrderedDict` of metric names to finalizers."""
#     return collections.OrderedDict(
#         num_examples=tf.function(func=lambda x: x[0]),
#         loss=tf.function(func=lambda x: x[0] / x[1]),
#         accuracy=tf.function(func=lambda x: x[0] / x[1]))

# iterative_process = tff.learning.build_federated_averaging_process(
#     MnistModel,
#     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02))

# state = iterative_process.initialize()

# state, metrics = iterative_process.next(state, federated_train_data)
# print('round  1, metrics={}'.format(metrics))

# for round_num in range(2, 11):
#   state, metrics = iterative_process.next(state, federated_train_data)
#   print('round {:2d}, metrics={}'.format(round_num, metrics))

# evaluation = tff.learning.build_federated_evaluation(MnistModel)

# str(evaluation.type_signature)

# train_metrics = evaluation(state.model, federated_train_data)

# str(train_metrics)

# federated_test_data = make_federated_data(emnist_test, sample_clients)

# len(federated_test_data), federated_test_data[0]

# test_metrics = evaluation(state.model, federated_test_data)

# str(test_metrics)
