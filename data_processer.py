from os import remove
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from numpy import savez
import os.path

datasets_map = {'rcv1': 'rcv1_train', 'url': 'url_combined', 'news': 'news20.binary',
                'rcv1_test': 'rcv1_test.binary'}

def get_data(data_name='rcv1', path=''):
    data = load_svmlight_file(path + datasets_map[data_name], zero_based=True)
    return data[0], data[1]
    
def save_data_for_slave(A, b, start_idx, end_idx, slave_number, path=''):
    dump_svmlight_file(A[start_idx:end_idx], b.A[start_idx:end_idx, 0], path + 'slave' + str(slave_number) + '_data', zero_based=True)
    
def remove_tmp_files(M, path=''):
    for i in range(M):
        remove(path + 'slave' + str(i + 1) + '_data')
        
def get_test_data(data_name, path=''):
    data = load_svmlight_file(path + datasets_map[data_name + '_test'], zero_based=True)
    return data[0], data[1]
    
def read_config(path=''):
    if not os.path.exists(path + 'config'):
        return None
    with open(path + 'config', 'r') as f:
        configuration = f.read().split(' ')
    return configuration