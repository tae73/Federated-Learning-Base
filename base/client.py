import gc
from typing import List, Dict

import keras.backend
import numpy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from base.network import *

class Client(object):
    """
    class for each client
    """
    def __init__(self, client_id, inputs, labels):
        self.client_id: int = client_id
        self.weights: List[np.ndarray]
        self.inputs: np.ndarray = inputs
        self.labels: np.ndarray = labels

        self.n_k_sample: int = len(self.inputs)
        self.client_net: Network = None

    def create_network(self, network_module: Network, learn_module, network_config, train_config):
        self.client_net = network_module(network_config, train_config, learn_module)

        pass  # statement for client network, not implemented, since the limitation of gpu memory in experiment.

    def learn(self, valid_data=None, verbose=0):
        self.client_net.learn(self.inputs, self.labels, valid_data=valid_data, verbose=verbose)

    def end_learn(self):
        self.set_client_weights(self.client_net.network.get_weights())
        gc.collect()
        del self.client_net.network
        K.clear_session()
        self.client_net = None

    def send_weights(self):
        pass  # not implemented, statement for client - server communication

    def receive_global_weights(self):
        return  # not implemented, statement for client - server communication

    def set_client_weights(self, global_weights):
        self.weights = global_weights
        if self.client_net: self.client_net.network.set_weights(self.weights)


def create_clients(config, num_clients, client_data, input_str='input', label_str='label', client_str='client-'):
    """
    create K clients
    :param config: network_config
    :param num_clients: the number of clients
    :param client_data: Dictionary of clients data. data[client][input or label]
    :param input_str: input string of client_data
    :param label_str: label string of label_data
    :return: List(Client)
    """
    clients = []
    for i in range(num_clients):
        client_id = i+1
        client = Client(client_id,
                        client_data[f'{client_str}{client_id}'][input_str],
                        client_data[f'{client_str}{client_id}'][label_str])
        clients.append(client)
    return clients

if __name__ == "__main__":
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
    train_config['optimizer_function'] = optimizers.Adam
    train_config['loss_metric'] = metrics.BinaryCrossentropy
    train_config['evaluate_metric'] = metrics.BinaryAccuracy

    client = Client(1, input_train, label_train)
    client.create_network(MLPNetwork, BaseNetworkLearn, network_config, train_config)
    client.client_net.learn(input_train, label_train, valid_data=None)

    del client.client_net.network
    keras.backend.clear_session()
