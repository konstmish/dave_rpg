import pickle
import socket
import struct
import select
import signal
import sys

from random import choice
from signal import SIGPIPE, SIG_DFL
signal.signal(SIGPIPE,SIG_DFL)

def send(channel, data):
    msg = pickle.dumps(data)
    msg = struct.pack('>I', len(msg)) + msg
    try:
        channel.send(msg)
    except OSError:
        print('Failed to send message')

def receive(channel):
    data = receive_by_size(channel, 4)
    if data is None:
        return None
    data_size = struct.unpack('>I', data)[0]
    data = receive_by_size(channel, data_size)
    return pickle.loads(data)
    
def receive_by_size(channel, size):
    data = b''
    while len(data) < size:
        try:
            packet = channel.recv(size - len(data))
            if not packet:
                return None
            data += packet
        except ConnectionResetError:
            return None
    return data
    
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 8000))
    ip = s.getsockname()[0]
    s.close()
    return str(ip)
    
def wait_to_get_data_from_slaves(inputs, outputs, server, numerate_map=None, delays=None):
    while True:
        try:
            inputready, outputready, exceptready = select.select(inputs, outputs, [])
        except select.error as e:
            print('Master:', e)
            break
        except socket.error as e:
            print('Master:', e)
            break
        
        if not inputready:
            continue
        
        if delays:
            max_delay = max([delays[numerate_map[inp]] for inp in inputready])
            priority = [inp for inp in inputready if numerate_map[inp] == 0] + [inp for inp in inputready if delays[numerate_map[inp]] == max_delay]
            s = priority[0]
        else:
            s = choice(inputready)
        
        if s == server:
            # ignore
            print('Master: Somebody tried to connect, I rejected it')

        elif s == sys.stdin:
            # handle standard input
            junk = sys.stdin.readline()
            print('Master: standard input should not be used.')
            return None
        else:
            # handle all other sockets
            try:
                data = receive(s)
                if data is not None:
                    return [data, s]
                else:
                    print('Master: %d hung up, terminating...' % s.fileno())
                    return None
                        
            except socket.error as e:
                # Remove
                inputs.remove(s)
                outputs.remove(s)
                print('Master: Socket error.')
                return None