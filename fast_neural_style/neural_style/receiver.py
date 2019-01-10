#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:30:20 2018

@author: yang
"""
from numpysocket import NumpySocket
import cv2
import time

npSocket = NumpySocket()
npSocket.startClient(8014)

# Read until video is completed
while(True):
    # Capture frame-by-frame
    try:
        frame = npSocket.recieveNumpy()
        cv2.imshow('Frame', frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
        npSocket.endClient()

npSocket.endClient()
print ("Closing")
cv2.destroyAllWindows()
