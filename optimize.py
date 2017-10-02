import subprocess
import argparse
import socket
import time
import os

from subprocess import run, check_output, Popen
from communication import get_ip

MAX_IT = 200

parser = argparse.ArgumentParser(description='Optimize convex objective by distributed proximal gradient descent')
parser.add_argument('n_slaves', action='store', type=int, help='Number of workers excluding the server')
parser.add_argument('-m', action='store_true', default=False, help='Run only the master')
parser.add_argument('-s', action='store_true', default=False, help='Run only slaves')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=MAX_IT, help='Max number of iterations')
parser.add_argument('-p', action='store_true', default=False, help='Print progress')
parser.add_argument('-v', action='store_true', default=False, help='Save produced values to a file')
parser.add_argument('--algo', action='store', dest='algo', choices=['asynch_gd', 'synch_gd', 'asynch_ave', 'daga', 'daga2', 'all'], 
                    help='Algorithms: asynchronous/synchronous gradient descent, asynchronous average gradient, DAGA', 
                    default='asynch_gd')
parser.add_argument('--ip', action='store', dest='ip', help='Max number of iterations')
parser.add_argument('--port', action='store', dest='port', default=8888, type=int, help='Server\'s port')
parser.add_argument('--data', action='store', dest='data', default='rcv1', help='Name of the dataset that should be used')
parser.add_argument('--cv', action='store_true', default=False, help='Use test set to evaluate performance')

results = parser.parse_args()

arguments = [str(results.n_slaves), '--algo', results.algo, '--port', str(results.port), '--data', results.data]

if results.v:
    arguments.append('-v')
if results.p:
    arguments.append('-p')
if results.max_it != MAX_IT:
    arguments.append('--max_it')
    arguments.append(str(results.max_it))
if results.cv:
    arguments.append('--cv')

if not results.s:
    print('Setting up server...')
    master = Popen(['python3', 'master.py'] + arguments)
    time.sleep(2)
    ip = get_ip()
else:
    ip = results.ip
    
if not results.m:
    print('Starting slaves...')
    for i in range(results.n_slaves):
        slave = Popen(['python3', 'slave.py', ip, '--port', str(results.port)])
    
if not results.s:
    master.wait()
    
if not results.m:
    slave.wait()