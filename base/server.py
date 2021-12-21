from typing import List, Dict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from base.networks import *
from base.client import *
from base.federated import *
from base.federated import *

class Server(object):
    def __init__(self, network_config, server_config, network_type='MLP', aggregator='FedAvg'):
        self.num_rounds = server_config['num_rounds']
        self.c_fraction = server_config['c_fraction']
        self.n_clients = server_config['n_clients']
        self.selected_clients_index = None

        if network_type == "MLP":
            self.netmodule = MLPNetwork(network_config)
        else:
            raise NotImplementedError('In the current progress, network_type must be set as "MLP".')
        self.init_global_network()

        if aggregator=='FedAvg':
            self.aggmodule = FedAvg()

        self.federated_loss_per_round: List = []

    def learn(self, clients: List[Client]):

        for round in range(self.num_rounds):
            print("==========" * 5, f"Round {round + 1}")
            self.train_one_round(clients)
            self.send_global_weights()
            print(f"federated loss: {self.federated_loss_per_round[-1]}")
            # Valid
        for client in clients:
            client.receive_global_weights(self.netmodule.network.get_weights())

    def train_one_round(self, clients: List[Client]):
        # select clients
        self.select_clients()  # 이 때는 client 객체 필요 없음.
        selected_clients = [client for client in clients if client.client_id in self.selected_clients_index] # client list
        n_sample = sum(client.n_k_sample for client in selected_clients) # total n

        selected_client_n_k_list = []
        selected_client_weight_list = []
        selected_client_loss_list = []
        for client in selected_clients: # parallel
            client.receive_global_weights(self.netmodule.network.get_weights())  # global weights to clients and set weights
            client.learn() # client learn
            # send client information to server (weights, loss, n_k_sample)
            selected_client_n_k_list.append(client.n_k_sample)
            selected_client_weight_list.append(client.netmodule.network.get_weights())
            selected_client_loss_list.append(client.netmodule.metric_dict['loss'][-1])  # TODO acc 추가?
        # aggregate clients weights
        global_weights, global_loss = self.aggmodule.aggregate(n_sample=n_sample,
                                                               selected_client_n_k_list=selected_client_n_k_list,
                                                               selected_client_loss_list=selected_client_loss_list,
                                                               selected_client_weight_list=selected_client_weight_list)
        self.set_global_weights(global_weights)  # set global weights as server weights
        self.federated_loss_per_round.append(global_loss)

    def evaluate_one_round(self, valid_data):
        input_valid = valid_data[0]
        input_label = valid_data[1]
        # TODO evaluate_one_round

    def init_global_network(self):
        self.netmodule.create_network()
        self.netmodule.network._name = 'server_network'

    def select_clients(self):
        n_selected_clients = max(int(self.n_clients * self.c_fraction), 1)
        self.selected_clients_index = np.random.choice(range(1, self.n_clients+1), n_selected_clients, replace=False)
        print(f"{len(self.selected_clients_index)} selected clients: ", self.selected_clients_index,
              f"with C fraction {self.c_fraction}")

    def set_global_weights(self, global_weights):
        self.netmodule.network.set_weights(global_weights)  # set global weights as server weights

    def send_global_weights(self):
        pass
        # raise NotImplementedError