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

def load_model(args):
        
    check_point = torch.load(args.checkpoint)
    #set arch
    if check_point['arch'] =='vgg13':
        model = models.vgg13(pretrained=True)
    elif check_point['arch'] =='vgg16':
        model = models.vgg16(pretrained=True)
    elif check_point['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True) 
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = check_point['class_to_idx']
    
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('hidden', nn.Linear(4096, args.hidden_units)),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    # Put the classifier on the pretrained network
    model.classifier = classifier    
    model.load_state_dict(check_point['state_dict'])    
    
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        model = model.to(device)
        if device == "cpu":
            print("gpu NOT available. Using cpu")
            
    return model

#Process image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
    from PIL import Image    
    image = Image.open(image_path)
    
    if image.size[0] > image.size[1]:
        image.thumbnail((999999, 256))
    else:
        image.thumbnail((256, 999999))
                          
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
                      
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    image = (image - mean)/std
    
    image = image.transpose((2, 0, 1))
    
    return image

def predict(args, image_path, model, topk, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Process image
    img = process_image(image_path)    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    if args.gpu:
        if torch.device("cuda:0" if torch.cuda.is_available() else "cpu") == "cpu":
            model = model.cpu()
        else:
            model = model.cuda()
    model.eval()
    
    if args.gpu:
        if torch.device("cuda:0" if torch.cuda.is_available() else "cpu") == "cpu":            
            output = torch.exp(model.forward(Variable(model_input)))
            probs = torch.exp(output)
        else:
            output = torch.exp(model.forward(Variable(model_input.cuda())))
            probs = torch.exp(output.cpu())
    else:
        output = torch.exp(model.forward(Variable(model_input)))
        probs = torch.exp(output)
        
    # Top probs
    top_probs, top_classes = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_classes = top_classes.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_classes]
    top_flowers = [cat_to_name[idx_to_class[j]] for j in top_classes]
    return top_probs, top_class, top_flowers

def main():
    parser = argparse.ArgumentParser(description='Flower Name Predictor')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU')
    parser.add_argument('--image_path', type=str, help='Flower Image path', required=True)
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Category Names')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden Units')
    parser.add_argument('--top_k', type=int, default=5, help='Top K')
    parser.add_argument('--checkpoint' , type=str, default='checkpoint.pth', help='Checkpoint')
    args = parser.parse_args()

    import json
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_model(args)   
    top_probs, top_class, top_flowers = predict(args, args.image_path, model, args.top_k, cat_to_name)

    print('Predicted Class: ', top_class)
    print('Top Flowers: ', top_flowers)
    print('Probability: ', top_probs)
    
if __name__ == "__main__":
    main()
