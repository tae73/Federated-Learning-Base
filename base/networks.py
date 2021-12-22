from typing import List, Dict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics


class Network(object):
    def __init__(self):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        self.network: models.Model = None
        self.results: List = None
        self.network_id = id

    def create_network(self):
        raise NotImplementedError()

    def build_keras(self, train_input, label_input, verbose):
        raise NotImplementedError()

    def build_with_tape(self, train_input, label_input, verbose):
        raise NotImplementedError()

    def step(self, train_input, label_input):
        raise NotImplementedError()


class MLPNetwork(Network):
    def __init__(self, config):
        super().__init__()
        # network param
        self.task: str = config['task']
        self.input_size: int = config['input_size']
        self.n_layers: int = config['n_layers']
        self.n_hidden_units: int = config['n_hidden_units']
        self.num_outputs: int = config['num_outputs']
        self.loss_function: tf.keras.losses = config['loss_function']
        self.random_seed: int = config['random_seed']
        # train param
        self.learning_rate: float = config['learning_rate']
        self.batch_size: int = config['batch_size']
        self.epochs: int = config['epochs']
        self.buffer_size: int = config['buffer_size']
        self.optimizer_function: tf.keras.optimizers = config['optimizer_function']
        self.loss_metric: tf.keras.metrics = config['loss_metric']
        self.evaluate_metric: tf.keras.metrics = config['evaluate_metric']
        # result param
        self.network: models.Model = None
        self.weights_grads: tf.Tensor = None  # TODO memory 차지만 하려나?
        self.results: Dict = None

    def create_network(self):
        """
        creates the MLP network
        :return: model: models.Model
        """
        # create input layer
        input_layer = layers.Input(shape=self.input_size, name="input")
        # create intermediate layer
        dense = input_layer
        for i in range(self.n_layers):
            dense = layers.Dense(
                units=self.n_hidden_units,
                kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                activation='relu',
                name='intermediate_dense_{}'.format(i + 1)
            )(dense)

        output_layer = layers.Dense(self.num_outputs,
                                    kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                                    activation="linear" if self.task == 'regression' else 'softmax',
                                    name="regressor" if self.task == 'regression' else 'classifier')(dense)
        self.network = models.Model(input_layer, output_layer)
        return self.network

    def create_train_dataset(self, input, label):
        train_dataset = tf.data.Dataset.from_tensor_slices((input, label)).shuffle(
            buffer_size=self.buffer_size,
            seed=self.random_seed).batch(self.batch_size)
        return train_dataset

    def build_keras(self, input, label, verbose):
        self.network.compile(optimizer=self.optimizer_function(lr=self.learning_rate), loss=self.loss_metric())
        self.network.fit(x=input, y=label, batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

    def build_with_tape(self, input, label, verbose):
        train_dataset = self.create_train_dataset(input, label)
        optimizer = self.optimizer_function(learning_rate=self.learning_rate)
        loss_metric = self.loss_metric()
        evaluate_metric = self.evaluate_metric()
        loss_per_epoch = []
        metric_per_epoch = []
        self.results = {}
        for epoch in range(self.epochs):
            if verbose == 1:
                print("=====" * 10, f"epoch {epoch + 1}: ")
            elif verbose == 0:
                pass
            for step, (train_batch, label_batch) in enumerate(train_dataset):
                y_hat = self.step(train_batch, label_batch)
                optimizer.apply_gradients(
                    zip(self.weights_grads, self.network.trainable_variables))

                loss_metric.update_state(y_true=label_batch, y_pred=y_hat)
                evaluate_metric.update_state(y_true=label_batch, y_pred=y_hat)
            loss = loss_metric.result()
            loss_per_epoch.append(loss)
            metric = evaluate_metric.result()
            metric_per_epoch.append(metric)
            if verbose == 1:
                print(f"train loss ({loss_metric.name}): {loss}, train {evaluate_metric.name}: {metric}")
            elif verbose == 0:
                pass
        self.results['loss'] = loss_per_epoch
        self.results[f'{evaluate_metric.name}'] = metric_per_epoch
        # TODO del y_hat / del loss / del metric / del los_per_epoch / del metric_per_epoch 필요?

    def step(self, input, label):
        loss_function = self.loss_function()
        with tf.GradientTape() as tape:
            weights = self.network.trainable_weights
            y_hat = self.network(input)
            empirical_loss = tf.reduce_mean(loss_function(label, y_hat))
            self.weights_grads = tape.gradient(empirical_loss, weights)
        return y_hat


if __name__ == "__main__":
    import numpy as np

    input_data = np.load(
        '/Users/taehyun/PycharmProjects/Federated_Learning_Framework/data/titanic/kaggle/edge_train_input.npy')
    input_label = np.load(
        '/Users/taehyun/PycharmProjects/Federated_Learning_Framework/data/titanic/kaggle/label_train.npy')

    config = {}
    # network param
    config['task'] = "classification"
    config['input_size'] = 4
    config['n_layers'] = 5
    config['n_hidden_units'] = 10
    config['num_outputs'] = 2
    config['loss_function'] = losses.BinaryCrossentropy
    config['random_seed'] = 42
    # train param
    config['learning_rate'] = 0.001
    config['batch_size'] = 1
    config['epochs'] = 64
    config['buffer_size'] = 1
    config['optimizer_function'] = optimizers.Adam
    config['loss_metric'] = metrics.BinaryCrossentropy
    config['evaluate_metric'] = metrics.BinaryAccuracy

    mlp = MLPNetwork(config=config)
    mlp.create_network()
    mlp.build_with_tape(input_data, input_label, verbose=1)
