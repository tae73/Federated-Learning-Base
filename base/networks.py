from typing import List, Dict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics, regularizers


class Network(object):
    def __init__(self):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        self.network: models.Model = None
        self.results: List = None

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
        self.num_classes: int = config['num_classes']
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

        output_layer = layers.Dense(self.num_classes,
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
            for step_idx, (train_batch, label_batch) in enumerate(train_dataset):
                predictions = self.step(train_batch, label_batch)
                optimizer.apply_gradients(
                    zip(self.weights_grads, self.network.trainable_variables))

                loss_metric.update_state(y_true=label_batch, y_pred=predictions)
                evaluate_metric.update_state(y_true=label_batch, y_pred=predictions)
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
            predictions = self.network(input)
            empirical_loss = tf.reduce_mean(loss_function(label, predictions))
            self.weights_grads = tape.gradient(empirical_loss, weights)
        return predictions


class ResNet9(Network):
    def __init__(self, config):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        super().__init__()
        self.network: models.Model = None
        self.results: List = None
        self.network_id = id
        # network param
        self.task: str = config['task']
        self.input_shape: int = config['input_shape']
        self.num_classes: int = config['num_classes']
        self.random_seed: int = config['random_seed']
        self.l2_decay: float = config['l2_decay']
        self.pool_pad: bool = config['pool_pad']
        # train param
        self.learning_rate: float = config['learning_rate']
        self.batch_size: int = config['batch_size']
        self.epochs: int = config['epochs']
        self.buffer_size: int = config['buffer_size']
        self.loss_function: tf.keras.losses = config['loss_function']
        self.optimizer_function: tf.keras.optimizers = config['optimizer_function']
        self.loss_metric: tf.keras.metrics = config['loss_metric']
        self.evaluate_metric: tf.keras.metrics = config['evaluate_metric']
        # result param
        self.network: models.Model = None
        self.weights_grads: tf.Tensor = None  # TODO memory 차지만 하려나?
        self.results: Dict = None

    def create_network(self):

        def conv_block(in_channels, out_channels, pool=False, pool_no=2, pool_pad=self.pool_pad):
            block = [layers.Conv2D(out_channels, kernel_size=(3, 3), padding='same', use_bias=False, strides=(1, 1),
                                   kernel_initializer=initializers.glorot_uniform(seed=self.random_seed),
                                   kernel_regularizer=regularizers.l2(self.l2_decay)),
                     layers.ReLU()]
            if pool:
                if pool_pad:
                    block.append(layers.MaxPooling2D(pool_size=(pool_no, pool_no), padding='same'))
                else:
                    block.append(layers.MaxPooling2D(pool_size=(pool_no, pool_no)))
            return models.Sequential(block)

        inputs = layers.Input(shape=self.input_shape)
        out = conv_block(self.input_shape[-1], 64)(inputs)
        out = conv_block(64, 128, pool=True, pool_no=2, pool_pad=self.pool_pad)(out)
        out = models.Sequential([conv_block(128, 128), conv_block(128, 128)])(out) + out
        out = conv_block(128, 256, pool=True, pool_pad=self.pool_pad)(out)
        out = conv_block(256, 512, pool=True, pool_no=2, pool_pad=self.pool_pad)(out)
        out = models.Sequential([conv_block(512, 512), conv_block(512, 512)])(out) + out
        out = models.Sequential([layers.MaxPooling2D(pool_size=4), layers.Flatten(),
                                 layers.Dense(self.num_classes, use_bias=False, activation='softmax')])(out)
        self.network = models.Model(inputs=inputs, outputs=out)

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
            for step_idx, (train_batch, label_batch) in enumerate(train_dataset):
                predictions = self.step(train_batch, label_batch)
                optimizer.apply_gradients(
                    zip(self.weights_grads, self.network.trainable_variables))

                loss_metric.update_state(y_true=label_batch, y_pred=predictions)
                evaluate_metric.update_state(y_true=label_batch, y_pred=predictions)
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
            predictions = self.network(input)
            empirical_loss = tf.reduce_mean(loss_function(label, predictions))
            self.weights_grads = tape.gradient(empirical_loss, weights)
        return predictions


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
    config['num_classes'] = 2
    config['loss_function'] = losses.BinaryCrossentropy
    config['random_seed'] = 42
    config['pad_pool'] = True
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

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_input, test_input = train_images / 255.0, test_images / 255.0

    train_input = train_input.reshape(-1, 28, 28, 1)
    test_input = test_input.reshape(-1, 28, 28, 1)
    train_label = tf.keras.utils.to_categoricalz(train_labels)
    test_label = tf.keras.utils.to_categorical(test_labels)

    config = {}
    # network param
    config['task'] = "classification"
    config['input_shape'] = (28, 28, 1)
    config['num_classes'] = 10
    config['l2_decay'] = 0.001
    config['loss_function'] = losses.CategoricalCrossentropy
    config['random_seed'] = 42
    # train param
    config['learning_rate'] = 0.01
    config['batch_size'] = 512
    config['epochs'] = 100
    config['buffer_size'] = 1000
    config['optimizer_function'] = optimizers.Adam
    config['loss_metric'] = metrics.CategoricalCrossentropy
    config['evaluate_metric'] = metrics.CategoricalAccuracy

    res = ResNet9(config)
    res.create_network(pool_pad=True)
    res.build_with_tape(train_input, train_label, 1)
