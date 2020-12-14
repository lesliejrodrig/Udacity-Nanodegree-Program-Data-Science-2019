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
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_directory", type=str, default="flowers")
parser.add_argument("--save_dir", type=str, default="checkpoint.pth")
parser.add_argument("--arch", type=str, default="vgg19")
parser.add_argument("--learning_rate", type=float, default="0.001")
parser.add_argument("--hidden_units", type=int, default="5000")
parser.add_argument("--epochs", type=int, default="5")
parser.add_argument("--gpu", action="store_true", default=False)
args = parser.parse_args()
#print(args.data_directory)
#print(args.save_dir)
#print(args.arch)
#print(args.learning_rate)
#print(args.hidden_units)
#print(args.epochs)
#print(args.gpu)

### BEGIN CODE COPIED FROM WORKSPACE UTILS PY CODE
import signal
from contextlib import contextmanager
import requests


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable

### END WORKSPACE UTILS PY CODE


data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_train = transforms.Compose([transforms.Resize(224),transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop((224,224)),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

data_transforms_validation = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop((224,224)),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

data_transforms_test = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop((224,224)),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)
image_datasets_validation = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)
image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
dataloaders_validation = torch.utils.data.DataLoader(image_datasets_validation, batch_size=64, shuffle=True)
dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=64, shuffle=True)

#Load Pre-Trained Network
if args.arch == "vgg19":
    model = models.vgg19(pretrained=True)
elif args.arch == "vgg16":
    model = models.vgg16(pretrained=True)
else:
    print("Architecture not recognized, can only recognize vgg16 or vgg19")
    sys.exit()
#print(model)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
#Build custom classifier here, replacing the pretrained model classifier
#Requires to start with 25088 since it needs to match pretrained model in_features, 5000 is simple param
model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, args.hidden_units)),
                                 ('relu', nn.ReLU()),
                                 ('drop', nn.Dropout(0.5)),
                                 ('fc2', nn.Linear(args.hidden_units, 102)),
                                 ('output', nn.LogSoftmax(dim=1))]))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

print("Begin training of the model")
start = time.time()

with active_session():
    epochs = args.epochs
    steps = 0
    print_every = 5

    device = torch.device("cuda" if (torch.cuda.is_available() & args.gpu) else "cpu")
    
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss=0
    
        for inputs, labels in iter(dataloaders_train):
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
            
                #Make sure gradients are turned off to reduce compute time
                with torch.no_grad():
                    for inputs, labels in iter(dataloaders_validation):
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        logps = model.forward(inputs)                    
                        test_loss += criterion(logps, labels).item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                    
                        equals = (labels.data == ps.max(dim=1)[1])
                        accuracy += equals.type(torch.FloatTensor).mean()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders_validation):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders_validation):.3f}")
                running_loss = 0
                model.train()
            
end = time.time()
print("Time taken to train: " + str(end - start))

# TODO: Do validation on the test set
#Switch to eval mode
model.eval()

model.to(device)

with torch.no_grad():
    accuracy = 0
    
    for inputs, labels in iter(dataloaders_test):
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        
        ps = torch.exp(logps)
        equals = (labels.data == ps.max(dim=1)[1])
        accuracy += equals.type(torch.FloatTensor).mean()

    print("Test Accuracy: {}".format(accuracy/len(dataloaders_test)))

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets_train.class_to_idx

checkpoint = {'arch': args.arch, 'class_to_idx': model.class_to_idx, 'model_state_dict':model.state_dict()}

torch.save(checkpoint,args.save_dir)
