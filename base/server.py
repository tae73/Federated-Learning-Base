from typing import List, Dict
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics
from tqdm import tqdm
from base.networks import *
from base.client import *
from base.federated import *


class Server(object):
    def __init__(self, network_config, server_config, network_module, aggregator):
        """
        class for server

        """
        self.num_rounds = server_config['num_rounds']
        self.c_fraction = server_config['c_fraction']
        self.num_clients = server_config['num_clients']

        self.selected_clients_index = None
        self.global_net: Network = network_module(network_config)

        self.init_global_network()

        self.aggmodule: Aggregator = aggregator()

        # results during a round
        self.selected_client_n_k_list: List = None
        self.selected_client_weight_list: List = None
        self.selected_client_loss_list: List = None

        # results for train
        self.federated_loss_per_round: List = None
        self.train_loss_per_round: List = None
        self.train_eval_metric_per_round: List = None  # acc , rmse, auc ..
        self.valid_loss_per_round: List = None
        self.valid_eval_metric_per_round: List = None  # acc , rmse, auc ..

    def learn(self, clients: List[Client], valid_data: List = None):
        self.federated_loss_per_round = []
        self.train_loss_per_round = []
        self.train_eval_metric_per_round = []
        if valid_data is not None:
            self.valid_loss_per_round = []
            self.valid_eval_metric_per_round = []

        # send initialized global weights to all clients and set client's weights
        self.send_global_weights()
        for client in clients: # parallel
            client.receive_global_weights()
            client.set_global_weights(self.global_net.network.get_weights())
        # start learning round / client parally do
        for round in range(self.num_rounds):
            print("==========" * 5, f"Round {round + 1}")
            self.train_one_round(clients)
            # Valid
            if valid_data is not None:
                self.valid_one_round(valid_data)
                print(f"Federated loss: {np.round(self.federated_loss_per_round[-1], 4)}, "
                      f"Train loss: {np.round(self.train_loss_per_round[-1], 4)}, "
                      f"Train {self.global_net.evaluate_metric().name}: "
                      f"{np.round(self.train_eval_metric_per_round[-1], 4)}, \n"
                      f"Valid loss: {np.round(self.valid_loss_per_round[-1], 4)}, "
                      f"Valid {self.global_net.evaluate_metric().name}: "
                      f"{np.round(self.valid_eval_metric_per_round[-1], 4)}, "
                      )
            else:
                print(f"Federated loss: {np.round(self.federated_loss_per_round[-1], 4)}, "
                      f"Train loss: {np.round(self.train_loss_per_round[-1], 4)}, "
                      f"Train {self.global_net.evaluate_metric().name}: "
                      f"{np.round(self.train_eval_metric_per_round[-1], 4)},"
                      )
        # send and receive final global weights
        self.send_global_weights()
        for client in clients:
            client.receive_global_weights()
            client.set_global_weights(self.global_net.network.get_weights())

    def train_one_round(self, clients: List[Client]):  # server
        """
        train one round of learning procedure
        clients learn its local parameter and sever aggregates the weights of selected clients

        :param clients:
        :return:
        """
        # select clients
        selected_clients = self.select_clients(clients)  # 이 때는 client 객체 필요 없음.

        n_sample = sum(client.n_k_sample for client in selected_clients)  # total n

        # clients update parallel
        self.clients_update_one_round(selected_clients)
        self.receive_weights()
        # aggregate clients weights
        federated_weights, federated_loss = self.aggmodule.aggregate(
            n_sample=n_sample,
            selected_client_n_k_list=self.selected_client_n_k_list,
            selected_client_loss_list=self.selected_client_loss_list,
            selected_client_weight_list=self.selected_client_weight_list
        )
        self.set_global_weights(federated_weights)  # set global weights as server weights
        self.federated_loss_per_round.append(federated_loss)
        # train eval for all client's input
        predictions = self.global_net.network(np.concatenate([client.input for client in selected_clients], axis=0))
        labels = np.concatenate([client.label for client in selected_clients], axis=0)
        # train loss
        train_loss = (tf.reduce_sum(self.global_net.loss_metric()(labels, predictions))).numpy()
        self.train_loss_per_round.append(train_loss)
        # train eval metric
        train_eval_metric = (tf.reduce_sum(self.global_net.evaluate_metric()(labels, predictions))).numpy()
        self.train_eval_metric_per_round.append(train_eval_metric)

    def valid_one_round(self, valid_data: List):
        """

        :param valid_data: List(input_valid, label_valid)
        :return:
        """
        # valid eval for external validation dataset
        predictions = self.global_net.network(valid_data[0])
        labels = valid_data[1]
        # valid loss
        valid_loss = (tf.reduce_sum(self.global_net.loss_metric()(labels, predictions))).numpy()
        self.valid_loss_per_round.append(valid_loss)
        # train eval metric
        valid_eval_metric = (tf.reduce_sum(self.global_net.evaluate_metric()(labels, predictions))).numpy()
        self.valid_eval_metric_per_round.append(valid_eval_metric)

    def clients_update_one_round(self, selected_clients: List[Client]):  # clients
        self.selected_client_n_k_list = []
        self.selected_client_weight_list = []
        self.selected_client_loss_list = []
        for client in tqdm(selected_clients, desc='clients update'):  # parallel
            client.receive_global_weights()  # receive weights from server
            client.set_global_weights(self.global_net.network.get_weights())  # set global weights
            client.learn()  # client learn
            # send client information to server (weights, loss, n_k_sample)
            client.send_weights()
            self.selected_client_n_k_list.append(client.n_k_sample)
            self.selected_client_weight_list.append(client.client_net.network.get_weights())
            self.selected_client_loss_list.append(client.client_net.results['loss'][-1])  # TODO acc 추가?

    def init_global_network(self):
        self.global_net.create_network()
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
