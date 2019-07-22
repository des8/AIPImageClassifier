# Imports here
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import transforms, models, datasets

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import OrderedDict

import time
import json
import copy
import argparse

def load_data(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(45), 
                                                    transforms.RandomResizedCrop(224), 
                                                    transforms.RandomHorizontalFlip(), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.Resize(256), 
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(), 
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)}
    
    #get dataset sizes
    dataset_sizes = {'train': len(image_datasets['train']),
                     'valid': len(image_datasets['valid']),
                     'test': len(image_datasets['test'])}
    
    return image_datasets, dataloaders, dataset_sizes

# TODO: Build and train your network
def train_model(args, model, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    
    print(device)
    print(num_epochs)
    image_datasets, dataloaders, dataset_sizes = load_data(args)
    since = time.time()    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
        
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train() #training mode
            else:
                model.eval() #evaluation model

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) 
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def general(args):
    
    image_datasets, dataloaders, data_sizes = load_data(args)
    
    if args.arch =='vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch =='vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print("Only VGG arch available. Using default arch vgg13")
        model = models.vgg13(pretrained=True)        
    
    for param in model.parameters():
        param.requires_grad = False
        
   
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096, True)), 
                                            ('relu1', nn.ReLU()), 
                                            ('dropout1', nn.Dropout(p=0.5)), 
                                            ('hidden', nn.Linear(4096,512)), 
                                            ('fc2', nn.Linear(512, 102, True)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        model = model.to(device)
        if device == "cpu":
            print("gpu NOT available. Using cpu")
        
    criteria = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    epochs = args.epochs
    
    trained_model = train_model(args, model, criteria, optimizer, sched, epochs, device)
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    torch.save({'arch': args.arch, 'state_dict': model.state_dict(), 'optimizer_dict': optimizer.state_dict(), 'class_to_idx':model.class_to_idx},args.save_dir)
    
def main():
    parser = argparse.ArgumentParser(description='Imagine Classifier')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU')
    parser.add_argument('--arch', type=str, default='vgg13', help='CCN Network Architecture - Only VGG available', required=True)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden Units')
    parser.add_argument('--epochs', type=int, default=25, help='Number of Epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='Image Data Directory')
    parser.add_argument('--save_dir' , type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    general(args)

if __name__ == "__main__":
    main()