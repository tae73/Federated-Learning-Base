from tensorflow import optimizers
from tensorflow import metrics
from tensorflow import losses

from utils.save_load import *
from base.networks import *
from base.federated import *
from base.client import *
from base.server import *

def load_fl_data(data_dir):
    client_common_train = pkl_to_OrderedDict('icu_client_common_train.pkl', data_dir)
    client_vertical_train = pkl_to_OrderedDict('icu_client_vertical_train.pkl', data_dir)
    client_full_train = pkl_to_OrderedDict('icu_client_full_train.pkl', data_dir)
    # client_common_valid = pkl_to_OrderedDict('icu_client_common_valid.pkl', data_dir)
    # client_vertical_valid = pkl_to_OrderedDict('icu_client_vertical_valid.pkl', data_dir)
    client_full_valid = pkl_to_OrderedDict('icu_client_full_valid.pkl', data_dir)
    # client_common_test = pkl_to_OrderedDict('icu_client_common_test.pkl', data_dir)
    # client_vertical_test = pkl_to_OrderedDict('icu_client_vertical_test.pkl', data_dir)
    client_full_test = pkl_to_OrderedDict('icu_client_full_test.pkl', data_dir)
    external_data = pkl_to_OrderedDict('icu_external_data.pkl', data_dir)

    return client_full_train, client_full_valid, client_full_test, external_data

def create_clients(config, num_clients, client_data, input_str, label_str):
    clients = []
    for i in range(num_clients):
        client_id = i+1
        client = Client(config, client_id, client_data[f'client_{client_id}'][input_str], client_data[f'client_{client_id}']['label'])
        clients.append(client)
    return clients


if __name__=="__main__":

    data_dir =
    client_full_train, client_full_valid, client_full_test, external_data = load_fl_data(data_dir)

    for client in client_full_train:
        client_full_train[client]['label'] = tf.keras.utils.to_categorical(client_full_train[client]['label'].values, num_classes=2)
    for client in client_full_valid:
        client_full_valid[client]['label'] = tf.keras.utils.to_categorical(client_full_valid[client]['label'].values, num_classes=2)
    for client in client_full_test:
        client_full_test[client]['label'] = tf.keras.utils.to_categorical(client_full_test[client]['label'].values, num_classes=2)

    network_config = {}
    # network param
    network_config['task'] = "classification"
    network_config['input_size'] = 40
    network_config['n_layers'] = 5
    network_config['n_hidden_units'] = 10
    network_config['num_outputs'] = 2
    network_config['loss_function'] = losses.BinaryCrossentropy
    network_config['random_seed'] = 42
    # train param
    network_config['learning_rate'] = 0.001
    network_config['batch_size'] = 1
    network_config['epochs'] = 5
    network_config['buffer_size'] = 1
    network_config['optimizer_function']= optimizers.Adam
    network_config['loss_metric'] = metrics.BinaryCrossentropy
    network_config['evaluate_metric'] = metrics.BinaryAccuracy
    
    # server config
    server_config = {}
    server_config['num_rounds'] = 5
    server_config['c_fraction'] = 1
    server_config['n_clients'] = 4
    clients = create_clients(network_config, 4, client_full_train, input_str='input_train', label_str='label')
    
    central_server = Server(network_config, server_config)
    central_server.learn(clients)
    
    