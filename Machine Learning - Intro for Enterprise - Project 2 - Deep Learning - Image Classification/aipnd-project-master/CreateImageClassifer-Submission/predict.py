#!/usr/bin/env python
# coding: utf-8

# Imports here
import torch
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np

import torchvision.models as models
from collections import OrderedDict

from torch import nn
from torch import optim

import time

from PIL import Image

import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, default="flowers/test/9/image_06413.jpg")
parser.add_argument("checkpoint", type=str, default="checkpoint.pth")
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--category_names", type=str, default="cat_to_name.json")
parser.add_argument("--gpu", action="store_true", default=False)
args = parser.parse_args()
#print(args.image_path)
#print(args.checkpoint)
#print(args.top_k)
#print(args.category_names)
#print(args.gpu)


with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(image_path):
    checkpoint = torch.load(image_path)
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized")
    
    model.class_to_idx = checkpoint['class_to_idx']

    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                     ('relu', nn.ReLU()),
                                     ('drop', nn.Dropout(0.5)),
                                     ('fc2', nn.Linear(5000, 102)),
                                     ('output', nn.LogSoftmax(dim=1))]))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

model = load_checkpoint(args.checkpoint)
#print(model)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2,0,1))
    
    return np_image

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if (torch.cuda.is_available() & gpu) else "cpu")
    #print(device)
    
    model.to(device)
    image = process_image(image_path)
    
    #check for gpu option
    if str(device) == "cpu":
        image = torch.from_numpy(image).type(torch.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    
    image = image.unsqueeze(0)
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_ps , top_indices = ps.topk(topk)
    top_ps = top_ps.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_ps, top_classes

probs, classes = predict(args.image_path, model, args.top_k, args.gpu)
flower_names = [cat_to_name[i] for i in classes]
#print(flower_names)
print("Flower Name(s): " + str(flower_names))
print("Class Probability(ies): " + str(probs))
