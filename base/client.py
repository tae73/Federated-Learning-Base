from typing import List, Dict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from base.networks import *

class Client(object):
    def __init__(self, network_config, client_id, input, label, network_type="MLP"):
        self.client_id: int = client_id
        if network_type == "MLP":
            self.netmodule = MLPNetwork(network_config)
        else:
            raise NotImplementedError('In the current progress, network_type must be set as "MLP".')
        self.input = input
        self.label = label

        self.n_k_sample = len(self.input)
        self.init_network()

    def init_network(self):
        self.netmodule.create_network()
        self.netmodule.network._name = f'client-{self.client_id}_network'

    def receive_global_weights(self, global_weights):
        self.netmodule.network.set_weights(global_weights)

    def learn(self, verbose=0):
        self.netmodule.build_with_tape(self.input, self.label, verbose=verbose)

def create_clients(config, num_clients, client_data, input_str, label_str):
    clients = []
    for i in range(num_clients):
        client_id = i+1
        client = Client(config, client_id, client_data[f'client_{client_id}'][input_str], client_data[f'client_{client_id}']['label'])
        clients.append(client)
    return clients




if __name__ == "__main__":
    import numpy as np
    input_data = np.load('/Users/taehyun/PycharmProjects/Federated_Learning_Framework/data/titanic/kaggle/edge_train_input.npy')
    input_label = np.load('/Users/taehyun/PycharmProjects/Federated_Learning_Framework/data/titanic/kaggle/label_train.npy')

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
    config['optimizer_function']= optimizers.Adam
    config['loss_metric'] = metrics.BinaryCrossentropy
    config['evaluate_metric'] = metrics.BinaryAccuracy


    client = Client(network_config=config, client_id='client-1', input=input_data, label=input_label)
    client.learn()



