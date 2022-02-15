from base.network import *
from base.aggregate import *
from base.client import *
from base.server import *
from utils.save_load import *
from utils.federated_dataset import create_base_client_data
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra

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
    config = OmegaConf.load('./config/mlp_config.yaml')
    client_full_train, client_full_valid, client_full_test, external_data = load_fl_data(DATA_PATH)

    clients = create_clients(4, client_full_train, input_str='input_train', label_str='label',
                             client_str='client_')
    central_server = Server(config,
                            MLPNetwork, BaseNetworkLearn, BaseFederatedLearn, FedAvg)

    central_server.learn(clients, valid_data=[external_data['input_full'], external_data['label']])

def exp_mnist(distribute=False):
    from utils import gpu_utils
    gpu_utils.disable_tensorflow_debugging_logs()

    config = OmegaConf.load('./config/cnn_config.yaml')

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_input, test_input = train_images / 255.0, test_images / 255.0
    train_input = train_input.reshape(-1, 28, 28, 1)
    test_input = test_input.reshape(-1, 28, 28, 1)
    train_label = tf.keras.utils.to_categorical(train_labels)
    test_label = tf.keras.utils.to_categorical(test_labels)

    client_data = create_base_client_data(config.federate.num_clients
                                          , train_input, train_label)

    clients = create_clients(config.federate.num_clients,
                             client_data, input_str='input', label_str='label',
                             distribute=distribute)
    central_server = Server(config,
                            ResNet9, BaseNetworkLearn, BaseFederatedLearn, FedAvg,
                            distribute)
    central_server.learn(clients, valid_data=[test_input, test_label])

if __name__ == "__main__":
    exp_mnist(False)
    # exp_physionet()