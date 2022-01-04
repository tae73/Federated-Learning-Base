from base.network import *
from base.aggregate import *
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
    network_config['random_seed'] = 42
    # train param
    train_config = {}
    train_config['learning_rate'] = 0.001
    train_config['batch_size'] = 100
    train_config['epochs'] = 20
    train_config['buffer_size'] = 500
    train_config['random_seed'] = 42
    train_config['loss_fn'] = losses.BinaryCrossentropy
    train_config['optimize_fn'] = optimizers.Adam
    train_config['loss_metric'] = metrics.BinaryCrossentropy
    train_config['evaluate_metric'] = metrics.BinaryAccuracy
    # server config
    federate_config = {}
    federate_config['num_rounds'] = 40
    federate_config['c_fraction'] = 1
    federate_config['num_clients'] = 4
    federate_config['loss_metric'] = metrics.BinaryCrossentropy
    federate_config['evaluate_metric'] = metrics.BinaryAccuracy

    clients = create_clients(network_config, 4, client_full_train, input_str='input_train', label_str='label',
                             client_str='client_')
    central_server = Server(network_config, train_config, federate_config,
                            MLPNetwork, BaseNetworkLearn, BaseFederatedLearn, FedAvg)

    central_server.learn(clients, valid_data=[external_data['input_full'], external_data['label']])

def exp_mnist():

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_input, test_input = train_images / 255.0, test_images / 255.0
    train_input = train_input.reshape(-1, 28, 28, 1)
    test_input = test_input.reshape(-1, 28, 28, 1)
    train_label = tf.keras.utils.to_categorical(train_labels)
    test_label = tf.keras.utils.to_categorical(test_labels)

    network_config = {}
    # network param
    network_config['task'] = "classification"
    network_config['input_shape'] = (28, 28, 1)
    network_config['num_classes'] = 10
    network_config['l2_decay'] = .0
    network_config['random_seed'] = 42
    network_config['pool_pad'] = True
    # train param
    train_config = {}
    train_config['learning_rate'] = 0.001
    train_config['batch_size'] = 256
    train_config['epochs'] = 10
    train_config['buffer_size'] = 1000
    train_config['random_seed'] = 42
    train_config['loss_fn'] = losses.CategoricalCrossentropy
    train_config['optimize_fn'] = optimizers.Adam
    train_config['loss_metric'] = metrics.CategoricalCrossentropy
    train_config['evaluate_metric'] = metrics.CategoricalAccuracy
    # server config
    federate_config = {}
    federate_config['num_rounds'] = 20
    federate_config['c_fraction'] = 0.6
    federate_config['num_clients'] = 100
    federate_config['loss_metric'] = metrics.CategoricalCrossentropy
    federate_config['evaluate_metric'] = metrics.CategoricalAccuracy
    federate_config['predict_batch_size'] = 10000

    client_data = create_base_client_data(federate_config['num_clients']
                                          , train_input, train_label)

    clients = create_clients(network_config, federate_config['num_clients']
                             , client_data, input_str='input', label_str='label')
    central_server = Server(network_config, train_config, federate_config,
                            ResNet9, BaseNetworkLearn, BaseFederatedLearn, FedAvg)
    central_server.learn(clients, valid_data=[test_input, test_label])

if __name__=="__main__":
    exp_mnist()