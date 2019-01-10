# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
import cv2
import torch
from mpi4py import MPI
import torchvision.transforms as tratransforms
frame=cv2.imread('/home/yang/下載/test.png')
trans=transforms.Compose(
    [
        transforms.ToTensor(),

    ])
tensor=trans(frame)
'''
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from PIL import Image
import struct
import re
import utils
from transformer_net import TransformerNet
from vgg import Vgg16
import cv2
import time
import pafy
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.size #number of processors
rank = comm.rank #calling process rank
from numpysocket import NumpySocket
host_ip = '140.113.69.226'  # change me
npSocket = NumpySocket()
npSocket.startServer(host_ip, 8014)

def stylize(frame):
    device = torch.device("cuda")

    content_image = frame
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load('../saved_models/udnie.pth')
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    return output.clamp(0,255)
url = "https://youtu.be/M0Z-CztQevc"
video = pafy.new(url)
best = video.getbestvideo('mp4')
playurl = best.url

cap = cv2.VideoCapture()
cap.open(playurl)

while(True):
    start=time.time()
    ret, frame = cap.read()
    #tensor=stylize(Image.fromarray(frame))
    #tensor=tensor[0].numpy().astype('uint8').transpose(1, 2, 0)

    #cv2.imshow('frame',np.frombuffer(tensor.tobytes(),dtype='uint8').reshape(1080,1920,3))
    #cv2.imshow('frame', tensor)
    npSocket.sendNumpy(frame)
    print(1/(time.time()-start))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

cap.release()


cv2.destroyAllWindows()



