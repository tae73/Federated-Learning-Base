from typing import List, Dict
from copy import deepcopy
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from tqdm import tqdm
from base.learn import *
from base.network import *
from base.client import *
from base.aggregate import *


class Server(object):
    def __init__(self, network_config, network_train_config, federate_config,
                 network_module, network_learn_module, federated_module, aggregate_fn):
        """
        class for server

        """
        self.num_rounds = federate_config['num_rounds']
        self.c_fraction = federate_config['c_fraction']
        self.num_clients = federate_config['num_clients']
        self.network_config = network_config
        self.network_train_config = network_train_config

        self.network_module = network_module
        self.network_learn_module = network_learn_module

        self.selected_clients_index = None
        self.global_net: Network = None
        self.init_global_network(network_config, network_train_config)

        self.federated_module: FederatedLearningProcedure = federated_module(self, aggregate_fn, federate_config)

        # results during a round
        self.selected_client_n_k_list: List = None
        self.selected_client_weight_list: List = None
        self.selected_client_loss_list: List = None

        # results for train
        self.federated_loss_per_round: List = None
        self.train_loss_per_round: List = None
        self.train_eval_per_round: List = None  # acc , rmse, auc ..
        self.valid_loss_per_round: List = None
        self.valid_eval_metric_per_round: List = None  # acc , rmse, auc ..

    def learn(self, clients: List[Client], valid_data: List = None):
        self.federated_module.learn(clients, valid_data)

    def init_global_network(self, network_config, train_config):
        self.global_net = self.network_module(network_config, train_config, self.network_learn_module)
        self.global_net.network._name = 'server_network'

    def select_clients(self, clients):
        n_selected_clients = max(int(self.num_clients * self.c_fraction), 1)
        self.selected_clients_index = np.random.choice(range(1, self.num_clients + 1), n_selected_clients, replace=False)
        print(f"{len(self.selected_clients_index)} selected clients: ", self.selected_clients_index,
              f"with C fraction {self.c_fraction}")
        selected_clients = [client for client in clients if
                            client.client_id in self.selected_clients_index]  # client list
        return selected_clients

    def set_global_weights(self, global_weights):
        self.global_net.network.set_weights(global_weights)  # set global weights as server weights

    def send_global_weights(self):
        pass  # not implemented, statement for client - server communication

    def receive_weights(self):
        """
        receive weights from selected clients
        :return:
        """
        pass  # not implemented, statement for client - server communication

if __name__ == "__main__":
    server_config = {}
    network_config = {}
    train_config = {}
    # server param
    server_config['num_rounds'] = 10
    server_config['c_fraction'] = 0.7
    server_config['num_clients'] = 10
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
    train_config['optimizer_function'] = optimizers.Adam
    train_config['loss_metric'] = metrics.CategoricalCrossentropy
    train_config['evaluate_metric'] = metrics.CategoricalAccuracy

    server = Server(network_config, train_config, server_config, ResNet9, BaseNetworkLearn, BaseFederatedLearn, FedAvg)
