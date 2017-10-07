#!/usr/bin/env python

"""
Master that controls prox-grad algorithm
"""

import select
import socket
import signal
import random
import time
import numpy as np
import pandas as pd
import pickle
import sys
import argparse

# from joblib import Parallel, delayed
from communication import send, receive, wait_to_get_data_from_slaves
from functions import F, prox_r
from data_processer import get_data, get_test_data, save_data_for_slave, remove_tmp_files, read_config
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix
from signal import SIGPIPE, SIG_DFL

#prevent 'broken pipe' problem
signal.signal(SIGPIPE, SIG_DFL)

random.seed = 0

C = 6

penalty_size_by_data = {'rcv1': 3e-6, 'url': 1e-8, 'news': 1e-6}

def master_print(*args, **kw_args):
    print('Master:', end= ' ')
    print(*args, **kw_args)

class Master(object):
    """ Communications are done using select """
    
    def __init__(self, n_slaves, algo, data_name, cross_val, port=8888, backlog=50, path=''):
        self.M = n_slaves
        self.algo = algo
        self.cross_val = cross_val
        self.connected_slaves = 0
        self.port = port
        self.data_name = data_name
        self.path = path
        # Client map
        self.numerate_map = {}
        # Output socket list
        self.outputs = []
        # Assign ip
        self.assign_ip()
        # Start server
        address = (self.ip, port)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.ip, port))
        master_print('Listening to port', port, '...')
        self.server.listen(backlog)
        # Trap keyboard interrupts
        signal.signal(signal.SIGINT, self.sighandler)

    def read_data(self):
        master_print('Reading the data')
        self.A, self.b = get_data(self.data_name, self.path)
        self.b = csr_matrix(self.b).T
        self.n = self.b.shape[0]
        master_print('The data is read. There are', self.n, 'observations and', self.A.shape[1], 'features.')
        
    def assign_ip(self):
        #hack it to get its ip
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 8000))
        ip = s.getsockname()[0]
        s.close()
        master_print('The ip is', ip)
        self.ip = ip
        with open(self.path + 'ip', 'w') as f:
            f.write(str(ip))
        
    def sighandler(self, signum, frame):
        master_print('Shutting down server...')
        # Close existing slave sockets
        for o in self.outputs:
            o.close()
        self.server.close()
        
    def distribute_data(self):
            
        #split the data
        # sep_idx = [0] + sorted(random.sample(range(self.b.shape[0] - 1), self.M - 1)) + [self.n]
        sep_idx = [0] + [(self.n * i) // (self.M + C) for i in range(1 + C, self.M + C)] + [self.n]
        master_print('Distributing the data...')
        for i in range(self.M):
            start_idx = sep_idx[i]
            end_idx   = sep_idx[i + 1]
            master_print(end_idx - start_idx, 'points go to slave', i + 1)
            save_data_for_slave(self.A, self.b, start_idx, end_idx, i + 1, self.path)
            
    def get_x_hat(self, x_bar=None):
        if x_bar is None:
            x_bar = self.x
        x_hat = prox_r(x_bar, self.gamma, self.l1)
        return x_hat
            
    def get_objective_value(self, x_hat):
        return F(x_hat, self.A, self.b, self.l2, self.l1)
        
    def print_sparsity(self, x_hat):
        master_print('sparsity of x: %.3f' % (1 - x_hat.count_nonzero() / x_hat.shape[0]))
        
    def get_sparsity(self, x_hat=None):
        if x_hat is None:
            x_hat = self.get_x_hat()
        return 'sparsity of x: ' + "{:.3f}".format(1 - x_hat.count_nonzero() / x_hat.shape[0])

    def print_summary(self, it=None, start=None, end=None, prefix='Current', print_sparsity=True, final=False, max_delay=None):
        x_hat = self.x if self.algo in ['asynch_ave', 'daga'] else self.get_x_hat()
        to_print = prefix + ' loss: ' + str(self.get_objective_value(x_hat))
        if print_sparsity:
            to_print += ', ' + self.get_sparsity(x_hat)
        if max_delay is not None and self.algo != 'synch_gd':
            to_print += ', maximal delay: ' + str(max_delay)
        master_print(to_print, end='\r' if not final else '\n')
        if final:
            master_print('It took %d iterations' % it, 'and', end - start, 'sec')
            
    def initialize_algorithm(self, first_time=True):
        self.tol = 1e-10
        self.l1 = penalty_size_by_data[self.data_name]
        self.l2 = 1 / self.n
        self.values = []
        self.times = []
        self.iterates = []
        if first_time:
            self.L = 0.25 * np.max(self.A.multiply(self.A).sum(axis=1))
        if self.algo == 'asynch_ave':
            first_option = 16 * ( (1 + self.l2 / (48 * self.L) ) ** (1 / self.M) - 1 ) / self.l2
            second_option = ( (1 + self.l2 / (self.M * self.L) ) ** (1 / self.M) - 1 ) / self.l2
            self.gamma = 2 * max(first_option, second_option)
        else:
            self.gamma = 2 / (self.L + 2 * self.l2)
        self.x = csr_matrix(np.zeros(self.A.shape[1])).T
        self.alpha = csr_matrix(np.zeros(self.A.shape[1])).T
        start = time.time()
        initial_value = self.get_objective_value(self.get_x_hat())
        self.values.append(initial_value)
        self.times.append(0)
        if not first_time:
            return
        print('It takes', time.time() - start, 'second to compute current value')
        master_print('Initial value:', initial_value, 'smoothness:', self.L, 'stepsize:', self.gamma)
        
    def save_values_history(self):
        master_print('Computing and saving intermediate values...')
        if self.algo == 'asynch_gd':
            x_hats = [self.get_x_hat(iterate) for iterate in self.iterates[::]]
        else:
            x_hats = self.iterates
        self.values += [self.get_objective_value(x_hat) for x_hat in x_hats]
        pickle.dump(self.values, open(self.path + 'logs_' + self.data_name + '/values_' + self.algo + '.p', 'wb'))
        pickle.dump(self.times, open(self.path + 'logs_' + self.data_name + '/times_' + self.algo + '.p', 'wb'))

        if not self.cross_val:
            return
        self.A, self.b = get_test_data(self.data_name, self.path)
        self.b = csr_matrix(self.b).T
        self.n = self.b.shape[0]
        self.values = self.values[0] + [self.get_objective_value(x_hat) for x_hat in x_hats]
        pickle.dump(self.values, open(self.path + 'logs_' + self.data_name + '/values_' + self.algo + '_test.p', 'wb'))
        pickle.dump(self.delays, open(self.path + 'logs_' + self.data_name + '/delays_' + self.algo + '.p', 'wb'))
        
    def inform_slaves(self, message):
        master_print('Informing slaves...')
        slave_informed = 0
        while slave_informed < self.M:
            data_and_socket = wait_to_get_data_from_slaves(self.inputs, self.outputs, self.server)
            if data_and_socket is None:
                break
            data, s = data_and_socket
            send(s, message)
            if (message == 'Terminate'):
                s.close()
                self.inputs = [inp for inp in self.inputs if inp != s]
                self.outputs = [outp for outp in self.outputs if outp != s]
            slave_informed += 1
        
    def wait_until_all_slaves_connect(self):
        self.inputs = [self.server]
        self.outputs = []
        while self.connected_slaves < self.M:
            try:
                inputready, outputready, exceptready = select.select(self.inputs, self.outputs, [])
            except select.error as e:
                break
            except socket.error as e:
                break
            
            for s in inputready:  
                if s == self.server:
                    # handle the server socket
                    slave, address = self.server.accept()
                    master_print('Got connection %d from %s' % (slave.fileno(), address))
                    
                    # Compute slave name and send back
                    self.numerate_map[slave] = self.connected_slaves
                    self.connected_slaves += 1
                    send(slave, ['SLAVE: ' + str(address[0]), self.connected_slaves])
                    self.inputs.append(slave)
                    
                    self.outputs.append(slave)
                    
        for i in range(self.M):
            slave_can_read = True
            send(self.outputs[i], [slave_can_read, self.A.shape[1]])
        return
                
    def work_until_condition(self, max_iter, save_values, print_state):
        norms = np.ones(self.M)
        it = 0
        slaves_waiting = 0
        stop_condition = False
        start = time.time()
        delays = [0 for i in range(self.M)]
        max_delay = 0
        iter_step = max(1, max_iter // 1000)
        self.max_delays = []
        while not stop_condition:
            data_and_socket = wait_to_get_data_from_slaves(self.inputs, self.outputs, self.server, self.numerate_map, delays)
            if not data_and_socket:
                continue
            if self.algo != 'synch_gd':
                it += 1
            if not print_state:
                to_print = ', current max delay: ' + str(max_delay) if self.algo != 'synch_gd' else ''
                master_print('Iteration: ' + str(it) + to_print, end='\r' if it < max_iter else '\n')
            data, s = data_and_socket
            if self.algo in ['asynch_ave', 'daga', 'daga2']:
                delta_x, delta_grad = data
                self.alpha += delta_grad
            else:
                delta_x = data

            self.x += delta_x
            if self.algo in ['asynch_ave', 'daga']:
                self.x = self.get_x_hat()
                
            if self.algo != 'synch_gd' and save_values and it % iter_step == 0:
                self.iterates.append(self.x)
                self.times.append(time.time() - start)
            
            # Send updated values in response
            if self.algo != 'synch_gd':
                data = [self.x, self.alpha] if self.algo in ['asynch_ave', 'daga', 'daga2'] else self.x
                send(s, data)
            else:
                slaves_waiting += 1
            

            delays = [d + 1 for d in delays]
            delays[self.numerate_map[s]] = 0
            max_delay = max(delays)
            self.max_delays.append(max_delay)
            N = 10 * (self.M - (self.M - 1) * (self.algo == 'synch_gd'))
            if print_state and it % N == 0:
                self.print_summary(max_delay=max_delay)
                
            if self.algo == 'synch_gd' and slaves_waiting == self.M:
                it += self.M
                if save_values and it % iter_step == 0:
                    self.iterates.append(self.x)
                    self.times.append(time.time() - start)
                for i in range(self.M):
                    send(self.outputs[i], self.x)
                slaves_waiting = 0
            
            norms[self.numerate_map[s]] = norm(delta_x)
            stop_condition = max(norms) * self.M < self.tol or it >= max_iter
            if stop_condition:
                break
            
        if stop_condition:
            end = time.time()
            self.print_summary(it, start, end, prefix='Final', final=True)
            if save_values:
                self.save_values_history()
    
    def optimize(self, max_iter, save_values, print_state):
        max_iter = max_iter * self.M
        self.read_data()
        self.distribute_data()
        self.wait_until_all_slaves_connect()
        
        algos = ['asynch_ave', 'synch_gd', 'asynch_gd'] if self.algo == 'all' else [self.algo]
        for j, algo in enumerate(algos):
            self.algo = algo
            self.initialize_algorithm(j == 0)
            master_print('Sending parameters...')
            for i in range(self.M):
                send(self.outputs[i], [self.x, self.M, self.gamma, self.l2, self.l1, self.n, self.algo, self.data_name])
            master_print('Start optimizing using', self.algo)
            self.work_until_condition(max_iter, save_values, print_state)
            if j < len(algos) - 1:
                self.inform_slaves('Change algorithm')
            else:
                self.inform_slaves('Terminate')
            master_print('Finished optimizing using', self.algo)
                    
        for s in self.inputs:
            s.close()
        self.server.close()
        
        remove_tmp_files(self.M, self.path)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Run parametric server for distributed optimization')
    parser.add_argument('--n_slaves', action='store', type=int, help='Number of workers excluding the server')
    parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=200, help='Max number of iterations')
    parser.add_argument('-p', action='store_true', default=False, help='Print progress')
    parser.add_argument('--port', action='store', default = 8888, type=int, help='Server\'s port')
    parser.add_argument('-v', action='store_true', default=False, help='Save produced values to a file')
    parser.add_argument('--algo', action='store', dest='algo', choices=['asynch_gd', 'synch_gd', 'asynch_ave', 'daga', 'daga2', 'all'],
                        help='Algorithms: asynchronous/synchronous gradient descent, asynchronous average gradient, DAGA')
    parser.add_argument('--data', action='store', dest='data', default = 'rcv1', help='Name of the dataset that should be used')
    parser.add_argument('--cv', action='store_true', default=False, help='Use test set to evaluate performance')
    parser.add_argument('--path', action='store', default='', help='Path to the data')
    
    configuration = read_config()
    if configuration is not None:
        results = parser.parse_args(configuration)
    else:
        results = parser.parse_args()
    
    Master(results.n_slaves, results.algo, results.data, results.cv, port=results.port, path=results.path).optimize(results.max_it, results.v, results.p)
