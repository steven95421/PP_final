#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:30:24 2018

@author: yang
"""


import time  
time.sleep(2)  
sock.send('qq'.encode())
print (sock.recv(1024).decode())  
