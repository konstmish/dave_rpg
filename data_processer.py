from os import remove
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from numpy import savez
import os.path

datasets_map = {'rcv1': 'rcv1_train', 'url': 'url_combined', 'news': 'news20.binary',
                'rcv1_test': 'rcv1_test.binary'}

def get_data(data_name='rcv1'):
    data = load_svmlight_file(datasets_map[data_name])
    return data[0], data[1]
    
def save_data_for_slave(A, b, start_idx, end_idx, slave_number):
    dump_svmlight_file(A[start_idx:end_idx], b[start_idx:end_idx], 'slave' + str(slave_number) + '_data')
    
def remove_tmp_files(M):
    for i in range(M):
        remove('slave' + str(i + 1) + '_data')
        
def get_test_data(data_name):
    data = load_svmlight_file(datasets_map[data_name + '_test'])
    return data[0], data[1]
    
def read_config():
    if not os.path.exists('config'):
        return None
    with open('config', 'r') as f:
        configuration = f.read().split(' ')
    return configuration