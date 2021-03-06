#############################################################################
#! /usr/bin/env python

"""
Slave for distributed optimization

"""

import socket
import select
import sys
import time
import argparse
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.datasets import load_svmlight_file
from communication import send, receive
from functions import prox_r, f_grad
from data_processer import get_data

PRINT_SUFFIX = ''

def slave_print(*args, **kw_args):
    print('Slave' + PRINT_SUFFIX + ':', end= ' ')
    print(*args, **kw_args)

class Slave(object):

    def __init__(self, ip=None, port=8888, path=''):
        # Quit flag
        self.path = path
        if ip is None:
            with open(path + 'ip', 'r') as f:
                ip = f.read()
        self.flag = False
        self.port = int(port)
        self.x = csr_matrix(0)
        self.x_ave = csr_matrix(0)
        self.alpha = csr_matrix(0)
        self.alpha_ave = csr_matrix(0)
        self.n = 0
        self.n_i = 0
        self.algo = None
        self.i = None
        
        # Connect to server at port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((ip, self.port))
            slave_print ('Connected to server@%d' % self.port)
            # Send slave's name...
            data = receive(self.sock)
            # Contains slave address, set it
            addr = data[0].split('SLAVE: ')[1]
            self.i = data[1]
            global PRINT_SUFFIX
            PRINT_SUFFIX = ' ' + str(self.i)
        except socket.error as e:
            slave_print ('Could not connect to server @%d' % self.port)
            sys.exit(1)
            
    def wait_until_get_data(self):
        while True:
            try:
                inputready, outputready, exceptrdy = select.select([0, self.sock], [],[])
            except select.error as e:
                break
            except socket.error as e:
                break
            
            for i in inputready:
                data = receive(self.sock)
                if data is None:
                    slave_print('Something went wrong')
                    self.flag = True
                    break
                can_read, n_features = data
                if can_read == True:
                    slave_print('Reading my part of the data with', n_features, 'features...')
                    self.A, self.b = load_svmlight_file(self.path + 'slave' + str(self.i) + '_data', n_features=n_features, zero_based=True)
                    self.b = csr_matrix(self.b).T
                    self.n_i = self.b.shape[0]
                    return
    
    def wait_until_get_parameters(self):
        parameters_not_known = True
        while parameters_not_known:
            try:
                inputready, outputready, exceptrdy = select.select([0, self.sock], [],[])
            except select.error as e:
                break
            except socket.error as e:
                break
            
            for i in inputready:
                data = receive(self.sock)
                if data is None:
                    slave_print('Something went wrong')
                    self.flag = 1
                    parameters_not_known = False
                    break
                slave_print('Receiving parameters...')
                self.x_ave, self.M, self.gamma, self.l2, self.l1, self.n, self.algo, data_name = data
                slave_print('Received parameters: M = %d, l2 = %3f, l1 = %3f, n = %d, algo = %s, dataset is %s'
                            % (self.M, self.l2, self.l1, self.n, self.algo, data_name))
                self.x = self.x_ave
                parameters_not_known = False
                return

    def serve(self):
        self.wait_until_get_data()
        self.wait_until_get_parameters()
                    
        self.alpha = csr_matrix(np.zeros(self.x.shape[0])).T
        self.alpha_ave = csr_matrix(np.zeros(self.x.shape[0])).T

        slave_print('Start optimizing...')
        it = 0
        while not self.flag:
            it += 1
            if it == 1:
                start = time.time()
            
            if self.algo in ['asynch_ave', 'daga']:
                grad = f_grad(self.x, self.A, self.b, self.l2)
                delta_grad = (grad - self.alpha) * (self.n_i / self.n)
                if self.algo == 'asynch_ave':
                    delta_x = -self.gamma * (self.alpha_ave + delta_grad)
                else:
                    delta_x = -self.gamma * (self.alpha_ave + delta_grad * (self.n / self.n_i))

            elif self.algo == 'daga2':
                z = prox_r(self.x_ave, self.gamma, self.l1)
                grad = f_grad(z, self.A, self.b, self.l2)
                delta_grad = (grad - self.alpha) * (self.n_i / self.n)
                x_new = z - self.gamma * (self.alpha_ave + delta_grad * (self.n / self.n_i))
                delta_x = (x_new - self.x_ave)
                self.x = x_new
            elif self.algo == 'daga3':
                x_new = z - self.gamma * grad
                delta_x = (x_new - self.x) * self.n_i / self.n - self.gamma * self.alpha_ave
                self.x = x_new

            if self.algo in ['asynch_ave', 'daga', 'daga2']:
                self.alpha = grad
                data = [delta_x, delta_grad]
            elif self.algo in ['synch_gd', 'asynch_gd']:
                delta = csr_matrix(np.zeros(self.A.shape[1])).T
                p = 1
                for i in range(p):
                    z = prox_r(self.x_ave + delta, self.gamma, self.l1)
                    grad = f_grad(z, self.A, self.b, self.l2)
                    x_new = z - self.gamma * grad
                    delta += (self.n_i / self.n) * (x_new - self.x)
                    self.x = x_new
                data = delta
                
            if it == 1:
                dif = time.time() - start
                n_passes = p if self.algo == 'asynch_gd' else 1
                slave_print('It takes', dif, 'seconds to make', n_passes, 'pass(es) through the data')
                start = time.time()
            
            send(self.sock, data)

            if it == 1:
                end = time.time()
                slave_print('It takes', end - start, 'seconds to send an update')
                
            not_updated = True
            while not_updated:
                try:
                    inputready, outputready, exceptrdy = select.select([0, self.sock], [],[])
                    for inp in inputready:
                        if inp == self.sock:
                            data = receive(self.sock)
                            not_updated = False
                            if data is None:
                                slave_print('Shutting down...')
                                self.flag = True
                                break
                            elif type(data) is str and data == 'Change algorithm':
                                self.wait_until_get_parameters()
                                slave_print('Start optimizing...')
                                continue
                            elif type(data) is str and data == 'Terminate':
                                slave_print('Terminating.')
                                self.flag = True
                                break
                            if self.algo in ['asynch_ave', 'daga', 'daga2']:
                                self.x, self.alpha_ave = data
                                break
                            else:
                                self.x_ave = data
                                break

                except KeyboardInterrupt:
                    slave_print('Interrupted.')
                    self.sock.close()
                    break
            # crash = np.random.binomial(1, 0.01)
#             if crash:
#                 time.sleep(2)
        self.sock.close()

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serve as a worker for distritued optimization')
    parser.add_argument('--ip', action='store', help='ip address of the server', default=None)
    parser.add_argument('--port', action='store', dest='port', default = 8888, type=int, help='Server\'s port')
    parser.add_argument('--path', action='store', default='', help='Path to the data')
    results = parser.parse_args()
    slave = Slave(results.ip, results.port, results.path)
    slave.serve()