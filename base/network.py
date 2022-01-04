from typing import List, Dict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics, regularizers
from base.learn import NetworkLearningProcedure, BaseNetworkLearn

class Network(object):
    def __init__(self, network_config, train_config, learn_module):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        self.network: models.Model = None
        self.learn_module: BaseNetworkLearn = None # TODO rename learn_module
        self.results:Dict = None
        
    def create_network(self):
        raise NotImplementedError()
    
    def learn(self):
        self.learn_module.learn()
        
class MLPNetwork(Network):
    def __init__(self, network_config, train_config, learn_module):
        super().__init__(network_config, train_config, learn_module)
        # network param
        self.task: str = network_config['task']
        self.input_size: int = network_config['input_size']
        self.n_layers: int = network_config['n_layers']
        self.n_hidden_units: int = network_config['n_hidden_units']
        self.num_classes: int = network_config['num_classes']
        self.random_seed: int = network_config['random_seed']
        #Learn module
        self.network: models.Model
        self.create_network()
        self.learn_module = learn_module(self.network, train_config)
        # result param
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

    def learn(self, inputs, labels, valid_data, verbose=True):
        self.learn_module.learn(inputs, labels, valid_data, verbose=verbose)

class ResNet9(Network):
    def __init__(self, network_config, train_config, learn_module=BaseNetworkLearn):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        super().__init__(network_config, train_config, learn_module)
        # network param
        self.task: str = network_config['task']
        self.input_shape: int = network_config['input_shape']
        self.num_classes: int = network_config['num_classes']
        self.random_seed: int = network_config['random_seed']
        self.l2_decay: float = network_config['l2_decay']
        self.pool_pad: bool = network_config['pool_pad']
        # Learn module
        self.network: models.Model
        self.create_network()
        self.learn_module = learn_module(self.network, train_config,)
        # result param
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

    def learn(self, inputs, labels, valid_data, verbose=True):
        self.learn_module.learn(inputs, labels, valid_data, verbose=verbose)

if __name__ == "__main__":

    def test_mlpnet():
        import numpy as np

        input_train = np.load(
            '/home/taehyun/project/Vertical-Cloud-Edge-Learning/data/titanic/kaggle/edge_train_input.npy')
        label_train = np.load(
            '/home/taehyun/project/Vertical-Cloud-Edge-Learning/data/titanic/kaggle/label_train.npy')

        network_config = {}
        train_config = {}
        # network param
        network_config['task'] = "classification"
        network_config['input_size'] = 4
        network_config['n_layers'] = 5
        network_config['n_hidden_units'] = 10
        network_config['num_classes'] = 2
        network_config['random_seed'] = 42
        # train param
        train_config['learning_rate'] = 0.001
        train_config['batch_size'] = 56
        train_config['epochs'] = 64
        train_config['buffer_size'] = 100
        train_config['random_seed'] = 42
        train_config['loss_function'] = losses.BinaryCrossentropy
        train_config['optimize_fn'] = optimizers.Adam
        train_config['loss_metric'] = metrics.BinaryCrossentropy
        train_config['evaluate_metric'] = metrics.BinaryAccuracy

        mlp = MLPNetwork(network_config, train_config)
        mlp.learn(input_train, label_train, valid_data=None, verbose=True)

    def test_resnet9():
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        train_input, test_input = train_images / 255.0, test_images / 255.0

        train_input = train_input.reshape(-1, 28, 28, 1)
        test_input = test_input.reshape(-1, 28, 28, 1)
        train_label = tf.keras.utils.to_categorical(train_labels)
        test_label = tf.keras.utils.to_categorical(test_labels)

        network_config = {}
        train_config = {}
        # network param
        network_config['task'] = "classification"
        network_config['input_shape'] = (28, 28, 1)
        network_config['num_classes'] = 10
        network_config['l2_decay'] = 0.001
        network_config['pool_pad'] = True
        network_config['random_seed'] = 42
        # train param
        train_config['random_seed'] = 42
        train_config['learning_rate'] = 0.01
        train_config['batch_size'] = 512
        train_config['epochs'] = 20
        train_config['buffer_size'] = 1000
        train_config['loss_function'] = losses.CategoricalCrossentropy
        train_config['optimize_fn'] = optimizers.Adam
        train_config['loss_metric'] = metrics.CategoricalCrossentropy
        train_config['evaluate_metric'] = metrics.CategoricalAccuracy

        res = ResNet9(network_config, train_config, BaseNetworkLearn)
        res.learn(train_input, train_label, [test_input, test_label], verbose=True)

    test_resnet9()


