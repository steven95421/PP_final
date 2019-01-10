#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:30:20 2018

@author: yang
"""
import sys
import numpy as np
from io import BytesIO
from numpysocket import NumpySocket
import cv2
import time
import threading
from queue import Queue
import socket
import multiprocessing.pool as mpool
sys.argv.append(1)
def thread_recieveNumpy(client_connection):
    length = None
    ultimate_buffer = b""
    while True:
        data = client_connection.recv(2048)
        ultimate_buffer += data
        if len(ultimate_buffer) == length:
            break
        while True:
            if length is None:
                if ':'.encode() not in ultimate_buffer:
                    break
                # remove the length bytes from the front of ultimate_buffer
                # leave any remaining bytes in the ultimate_buffer!
                length_str, ignored, ultimate_buffer = ultimate_buffer.partition(':'.encode())
                length = int(length_str)
            if len(ultimate_buffer) < length:
                break
            # split off the full message from the remaining bytes
            # leave any remaining bytes in the ultimate_buffer!
            ultimate_buffer = ultimate_buffer[length:]
            length = None
            break
    final_image = np.load(BytesIO(ultimate_buffer))['frame']
    print ('frame received')
    return final_image
address='140.113.69.226'
port=8014
num_of_server=int(sys.argv[1])
npSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
npSocket.bind((address,port))
npSocket.listen(num_of_server)
print('Listening...')


clients=[0 for i in range(num_of_server)]
for i in range(num_of_server):
    client, addr = npSocket.accept()
    print('Accepted Connection from: '+str(addr[0])+':'+str(addr[1]))
    data = client.recv(2048).decode()
    clients[int(data)]=client

q= Queue()
pool = mpool.ThreadPool()
def multiThreadSub(q):
    while(True):
        result=pool.map(thread_recieveNumpy, clients)
        for frame in result:
            q.put(frame)


    
# Read until video is completed
threading.Thread(target=multiThreadSub,args=(q,)).start()
time.sleep(1)
def play():
    while(True):
        frame=q.get()
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return
play()
print ("Closing")
cv2.destroyAllWindows()