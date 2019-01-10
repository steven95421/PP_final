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
npSocket.socket.sendall(str(rank).encode())
device = torch.device("cuda")
style_model = TransformerNet()
state_dict = torch.load('../saved_models/udnie.pth')
# remove saved deprecated running_* keys in InstanceNorm from the checkpoint






for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
style_model.load_state_dict(state_dict)
style_model.to(device)
def stylize(frame):
    

    content_image = frame
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    

    with torch.no_grad():

        output = style_model(content_image).cpu()
    return output.clamp(0,255)
url = "https://youtu.be/YweqHs7aq8A"
video = pafy.new(url)
best = video.getbestvideo('mp4')
playurl = best.url

#videofilepath = path.join(path.dirname(path.abspath(__file__)),'test.'+best.extension)
#videofilename = best.download(filepath = videofilepath)
#cap = cv2.VideoCapture(videofilepath)
cap = cv2.VideoCapture()
cap.open(playurl)
i=0
start=time.time()
while(True):
    ret, frame = cap.read()
    if(i%size==rank):
        #tensor=stylize(Image.fromarray(frame))
        #tensor=tensor[0].numpy().astype('uint8').transpose(1, 2, 0)
        #npSocket.sendNumpy(tensor)
        npSocket.sendNumpy(frame)
        print(rank,':',1/(time.time()-start))
        start=time.time()
        time.sleep(0.05)
    i+=1




