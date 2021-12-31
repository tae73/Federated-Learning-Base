import numpy as np


class Aggregator(object):
    def __init__(self):
        pass

    def aggregate(self, n_sample, selected_client_n_k_list, selected_client_loss_list, selected_client_weight_list):
        raise NotImplementedError()


class FedAvg(Aggregator):
    def __init__(self):
        """
        Federated Averaging
        """
        super().__init__()

    def aggregate(self, n_sample, selected_client_n_k_list, selected_client_loss_list, selected_client_weight_list):
        scaled_term_list = [n_k_sample / n_sample for n_k_sample in selected_client_n_k_list]
        scaled_losses_list = [loss * scaled_term for loss, scaled_term in
                              zip(selected_client_loss_list, scaled_term_list)]
        scaled_weights_list = []
        for client_weight, scaled_term in zip(selected_client_weight_list, scaled_term_list):
            scaled_weights = [layer_weight * scaled_term for layer_weight in client_weight]
            scaled_weights_list.append(scaled_weights)

        weights_average = [np.zeros_like(w) for w in scaled_weights_list[0]]
        for layer_index in range(len(weights_average)):
            scaled_weights_layer_list = [client_scaled_weights[layer_index]
                                         for client_scaled_weights in scaled_weights_list]
            averaged_weights_layer = np.sum(scaled_weights_layer_list, axis=0)
            weights_average[layer_index] = averaged_weights_layer
        loss_average = sum(scaled_losses_list)
        return weights_average, loss_average


if __name__ == "__main__":
    from base.client import *

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
    config['pool_pad'] = True
    # train param
    config['learning_rate'] = 0.001
    config['batch_size'] = 1
    config['epochs'] = 64
    config['buffer_size'] = 1
    config['optimizer_function'] = optimizers.Adam
    config['loss_metric'] = metrics.BinaryCrossentropy
    config['evaluate_metric'] = metrics.BinaryAccuracy

    client1 = Client(network_config=config, client_id='1', network_module=MLPNetwork,
                     input=input_data, label=input_label)
    client1.learn()
    config['learning_rate'] = 0.1
    client2 = Client(network_config=config, client_id='2', network_module=MLPNetwork,
                     input=input_data, label=input_label)
    client2.learn()

    agg = FedAvg()

    selected_clients = [client1, client2]
    n_sample = sum(client.n_k_sample for client in selected_clients)
    selected_client_n_k_list = []
    selected_client_weight_list = []
    selected_client_loss_list = []
    for client in selected_clients:
        selected_client_n_k_list.append(client.n_k_sample)
        selected_client_weight_list.append(client.client_net.network.get_weights())
        selected_client_loss_list.append(client.client_net.metric_dict['loss'][-1])

    avg_weight, loss = agg.aggregate(n_sample=n_sample,
                                     selected_client_n_k_list=selected_client_n_k_list,
                                     selected_client_loss_list=selected_client_loss_list,
                                     selected_client_weight_list=selected_client_weight_list)
