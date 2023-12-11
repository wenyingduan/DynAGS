import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('../data/PeMSD3/pems03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PeMSD7/pems07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'CA':
        data_path = os.path.join('../data/CA/CA_COVID.npz')
        data = np.load(data_path)['arr_0']  #onley the first dimension, traffic flow data
    elif dataset == 'TX':
        data_path = os.path.join('../data/TX/TX_COVID.npz')
        data = np.load(data_path)['arr_0']  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
