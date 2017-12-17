from scipy.io import loadmat
from scipy.linalg import solve_sylvester
import os
import random
import math
import argparse
import torch
import time
import numpy as np
from sklearn.decomposition import PCA
from helpers import get_dataset_dict, save_object
import torch.nn.functional as f

# Example usage:
# python train_test.py -p data/animals_sae/ -r results/filename

##########################
#####   PARSING       ####
##########################

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dataset_path",
                    default='./data/apascal/',
                    help=("full path to the dataset"))
parser.add_argument("-l", "--lambda",
                    default=500000, type=float,
                    help=("Lamda parameter"))
parser.add_argument("-r", "--r_name",
                    required=True,
                    help=("name of the pkl object to save the results"))

##########################
#####   FUNCTIONS     ####
##########################

def print_size_info():
    print ('seen_class_ids: ', seen_class_ids.shape)
    print ('seen_attr_mat: ', seen_attr_mat.shape)
    print ('unseen_class_ids: ', unseen_class_ids.shape)
    print ('umseen_attr_mat: ', unseen_attr_mat.shape)

def refresh_file_pointers(dict_key,file_name,data_path):
    fid[dict_key].close()
    fid[dict_key] = open(os.path.join(data_path,file_name),'r')

##########################
#####     INIT        ####
##########################

### Command line arguments
args = vars(parser.parse_args())
dataset_path = args['dataset_path']
lambda_val = args['lambda']
obj_name = args['r_name']

### Init Global Variables [Constants]
seen_in_dict = 'seen_input'
seen_in_dat = 'seen_data_input.dat'
unseen_in_dict = 'unseen_input'
unseen_in_dat = 'unseen_data_input.dat'
seen_out_dict = 'seen_output'
seen_out_dat = 'seen_data_output.dat'
unseen_out_dict = 'unseen_output'
unseen_out_dat = 'unseen_data_output.dat'
seen_dataset_dict = 'dataset_seen_pointers'
unseen_dataset_dict = 'dataset_unseen_pointers'

### Load .mat files
attr_path = os.path.join(dataset_path, 'attr_data.mat')
data = loadmat(attr_path)

### Init Global Variables [Updated in code]
feature_size = None
fid = {}
results = {}
num_classes = None
description_size = None

##########################
####  DATA LOAD INIT  ####
##########################

## Seen Data Initilization
print('Seen Data Initilization: ')
start_time = time.time()
# Load seen attribute matrices
seen_class_ids = torch.LongTensor(data['seen_class_ids'])
seen_attr_mat = torch.FloatTensor(data['seen_attr_mat'])
# Load file readers for seen Data
fid[seen_in_dict] = open(os.path.join(dataset_path,seen_in_dat),'r')
fid[seen_out_dict] = open(os.path.join(dataset_path,seen_out_dat),'r')
# No. of classes and No. of semantic descriptors
num_classes = seen_attr_mat.shape[0]
description_size = seen_attr_mat.shape[1]
# Get the feature size
for feat_in in fid[seen_in_dict]:
    feature_size = list(map(float,feat_in.split(',')))
    feature_size = len(feature_size)
    break
refresh_file_pointers(seen_in_dict,seen_in_dat,dataset_path)
print('Time taken: ', time.time()-start_time, '\n')

## Unseen Data Initialization
print('Unseen Data Initialization: ')
start_time = time.time()
unseen_class_ids = torch.FloatTensor(data['unseen_class_ids'])
unseen_attr_mat = torch.FloatTensor(data['unseen_attr_mat'])
## Load file readers for unseen Data
fid[unseen_in_dict] = open(os.path.join(dataset_path,unseen_in_dat),'r')
fid[unseen_out_dict] = open(os.path.join(dataset_path,unseen_out_dat),'r')
print('Time taken: ', time.time()-start_time, '\n')

##############################
####  SEEN DATA LOADING   ####
##############################

## Create the empty training tensors
print('Creating the empty training tensors')
start_time = time.time()
dataset_classes = get_dataset_dict(dataset_path,fid[seen_out_dict],seen_dataset_dict)
train_size = sum([value for key, value in dataset_classes.items()])
refresh_file_pointers(seen_out_dict,seen_out_dat,dataset_path)
train_t = torch.zeros(feature_size,train_size)
semantic_t = torch.zeros(description_size,train_size)
print('Time taken: ', time.time()-start_time, '\n')

## Filling the empty training tensors
start_time = time.time()
print('Filling the empty training tensors')
train_index = 0
for feat_in, feat_out in zip(fid[seen_in_dict], fid[seen_out_dict]):
    feat_out = int(feat_out) - 1
    feat_in_split = list(map(float,feat_in.split(',')))
    train_t[:,train_index] = torch.FloatTensor(feat_in_split)
    semantic_t[:,train_index] = seen_attr_mat[feat_out,:]
    train_index += 1
print('Time taken: ', time.time()-start_time, '\n')

################################
####  UNSEEN DATA LOADING   ####
################################

## Create the empty testing tensors
print('Creating the empty testing tensors')
start_time = time.time()
test_dataset_classes = get_dataset_dict(dataset_path,fid[unseen_out_dict], unseen_dataset_dict)
test_size = sum([value for key, value in test_dataset_classes.items()])
refresh_file_pointers(unseen_out_dict,unseen_out_dat,dataset_path)
test_t = torch.zeros(feature_size,test_size)
test_true_classes = torch.zeros(test_size).long()
print('Time taken: ', time.time()-start_time, '\n')

## Filling the empty testing tensors
start_time = time.time()
print('Filling the empty testing tensors')
test_index = 0
for feat_in, feat_out in zip(fid[unseen_in_dict], fid[unseen_out_dict]):
    feat_out = int(feat_out) - 1
    feat_in_split = list(map(float,feat_in.split(',')))
    test_t[:,test_index] = torch.FloatTensor(feat_in_split)
    test_true_classes[test_index] = feat_out
    test_index += 1
print('Time taken: ', time.time()-start_time, '\n')

##########################
####     TRAINING     ####
##########################

## Obtaining inputs to Sylvester
start_time = time.time()
print('Obtaining inputs to Sylvester')
train_t = f.normalize(train_t,p=2,dim=1)
A = torch.mm(semantic_t,semantic_t.t())
B = lambda_val*torch.mm(train_t,train_t.t())
C = (1 + lambda_val)*torch.mm(semantic_t,train_t.t())
print('Time taken: ', time.time()-start_time, '\n')

# Solving the Sylvester
start_time = time.time()
print("Solving Sylvester")
W = solve_sylvester(A.numpy(), B.numpy(), C.numpy())
W = torch.FloatTensor(W)
W = f.normalize(W,p=2,dim=1)
print('Time taken : ', time.time()-start_time, '\n')

## Compute training error
print('Compute training error: ')
start_time = time.time()
train_semantic_pred = torch.mm(W, train_t)
_, train_pred_classes = torch.max(torch.mm(seen_attr_mat, train_semantic_pred), dim=0)
_, train_true_classes = torch.max(torch.mm(seen_attr_mat, semantic_t), dim=0)
train_accuracy = torch.sum(train_pred_classes == train_true_classes) / train_pred_classes.numel()
print('train_accuracy (%): ', train_accuracy)
print('Time taken : ', time.time()-start_time, '\n')

##########################
####     TESTING      ####
##########################

## Compute testing error
print('Compute testing error: ')
start_time = time.time()
# test_t = f.normalize(test_t,p=2,dim=1)

test_semantic_pred = torch.mm(W, test_t)
test_semantic_pred = f.normalize(test_semantic_pred,p=2,dim=0)

unseen_attr_mat = f.normalize(unseen_attr_mat,p=2,dim=0)
unseen_attr_mat = f.normalize(unseen_attr_mat,p=2,dim=1)

_, test_pred_classes = torch.max(torch.mm(unseen_attr_mat, test_semantic_pred), dim=0)
test_accuracy = torch.sum(test_pred_classes == test_true_classes) / test_pred_classes.numel()
print('test_accuracy: ', '%.04f'%test_accuracy)
print('test_random_accuracy : ', '%.04f'%(1.0/unseen_attr_mat.size()[0]))
print('Time taken : ', time.time()-start_time, '\n')

##########################
####   SAVING FILES   ####
##########################

## Save files in dictionary and into a pickle
print('Save files in dictionary and into a pickle: ')
start_time = time.time()
results['weights'] = W 
results['train_pred_classes'] = train_pred_classes
results['train_true_classes'] = train_true_classes
results['test_pred_classes'] = test_pred_classes
results['test_true_classes'] = test_true_classes
results['train_accuracy'] = train_accuracy
results['test_accuracy'] = test_accuracy
save_object(results, obj_name)
print('Time taken : ', time.time()-start_time, '\n')

##########################
####  CLOSING FILES   ####
##########################

fid[seen_in_dict].close()
fid[seen_out_dict].close()
fid[unseen_in_dict].close()
fid[unseen_out_dict].close()
