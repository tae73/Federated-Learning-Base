from base.networks import *
from base.federated import *
from base.client import *
from base.server import *
from utils.save_load import *
from utils.federated_dataset import create_base_client_data
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

def exp_physionet():
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
    network_config['num_classes'] = 2
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
    server_config['num_clients'] = 4

    clients = create_clients(network_config, 4, client_full_train, input_str='input_train', label_str='label',
                             network_module=MLPNetwork)
    central_server = Server(network_config, server_config, network_module=MLPNetwork, aggregator=FedAvg)
    central_server.learn(clients, valid_data=[external_data['input_full'], external_data['label']])

def exp_mnist():
    network_config = {}
    # network param
    network_config['task'] = "classification"
    network_config['input_shape'] = (28, 28, 1)
    network_config['num_classes'] = 10
    network_config['l2_decay'] = 0.001
    network_config['loss_function'] = losses.CategoricalCrossentropy
    network_config['random_seed'] = 42
    network_config['pool_pad'] = True
    # train param
    network_config['learning_rate'] = 0.001
    network_config['batch_size'] = 256
    network_config['epochs'] = 10
    network_config['buffer_size'] = 1000
    network_config['optimizer_function'] = optimizers.Adam
    network_config['loss_metric'] = metrics.CategoricalCrossentropy
    network_config['evaluate_metric'] = metrics.CategoricalAccuracy
    # server config
    server_config = {}
    server_config['num_rounds'] = 50
    server_config['c_fraction'] = 0.7
    server_config['num_clients'] = 10

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_input, test_input = train_images / 255.0, test_images / 255.0
    train_input = train_input.reshape(-1, 28, 28, 1)[:10000]
    test_input = test_input.reshape(-1, 28, 28, 1)
    train_label = tf.keras.utils.to_categorical(train_labels)[:10000]
    test_label = tf.keras.utils.to_categorical(test_labels)

    client_data = create_base_client_data(10, train_input, train_label)

    clients = create_clients(network_config, 10, client_data, input_str='input', label_str='label',
                             network_module=ResNet9)
    central_server = Server(network_config, server_config, network_module=ResNet9, aggregator=FedAvg)
    central_server.learn(clients, valid_data=[test_input[:500], test_label[:500]])

if __name__=="__main__":
    exp_mnist()