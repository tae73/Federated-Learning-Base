from utils.save_load import *
from base.networks import *
from base.federated import *
from base.client import *
from base.server import *
from pathlib import Path

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

    for client in client_full_train:
        client_full_train[client]['label'] = tf.keras.utils.to_categorical(client_full_train[client]['label'].values, num_classes=2)
    for client in client_full_valid:
        client_full_valid[client]['label'] = tf.keras.utils.to_categorical(client_full_valid[client]['label'].values, num_classes=2)
    for client in client_full_test:
        client_full_test[client]['label'] = tf.keras.utils.to_categorical(client_full_test[client]['label'].values, num_classes=2)
    external_data['label'] = tf.keras.utils.to_categorical(external_data['label'])

    return client_full_train, client_full_valid, client_full_test, external_data

if __name__=="__main__":
    # PROJECT_PATH = Path('.').absolute().parents[1]
    PROJECT_PATH = Path('.').absolute()
    DATA_PATH = Path(PROJECT_PATH, 'processed_data', 'physionet2012')

    client_full_train, client_full_valid, client_full_test, external_data = load_fl_data(DATA_PATH)

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
    network_config['learning_rate'] = 0.0001
    network_config['batch_size'] = 64
    network_config['epochs'] = 10
    network_config['buffer_size'] = 500
    network_config['optimizer_function'] = optimizers.Adam
    network_config['loss_metric'] = metrics.BinaryCrossentropy
    network_config['evaluate_metric'] = metrics.BinaryAccuracy
    # server config
    server_config = {}
    server_config['num_rounds'] = 50
    server_config['c_fraction'] = 1
    server_config['n_clients'] = 4

    clients = create_clients(network_config, 4, client_full_train, input_str='input_train', label_str='label')
    central_server = Server(network_config, server_config)
    central_server.learn(clients, valid_data=[external_data['input_full'], external_data['label']])
