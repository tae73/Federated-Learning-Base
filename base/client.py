from typing import List, Dict

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from base.networks import *


class Client(object):
    """
    class for each client
    """
    def __init__(self, network_config, client_id, network_module, input, label):
        self.client_id: int = client_id
        self.client_net: Network = network_module(network_config)
        self.input: np.ndarray = input
        self.label: np.ndarray = label

        self.n_k_sample: int = len(self.input)
        self.init_network()

    def init_network(self):
        self.client_net.create_network()
        self.client_net.network._name = f'client-{self.client_id}_network'

    def send_weights(self):
        pass  # not implemented, statement for client - server communication

    def receive_global_weights(self):
        return  # not implemented, statement for client - server communication

    def set_global_weights(self, global_weights):
        self.client_net.network.set_weights(global_weights)

    def learn(self, verbose=0):
        self.client_net.build_with_tape(self.input, self.label, verbose=verbose)


def create_clients(config, num_clients, client_data, network_module,
                   input_str='input', label_str='label', client_str='client-'):
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
        client = Client(config, client_id, network_module,
                        client_data[f'client-{client_id}'][input_str], client_data[f'client-{client_id}'][label_str])
        clients.append(client)
    return clients
