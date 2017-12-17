from scipy.io import loadmat
from scipy.linalg import solve_sylvester
import os
import random
import math
import argparse
import torch
import time
import numpy as np
from helpers import get_dataset_dict, save_object

# Example usage:
# python read_dataset.py -p ./data/apascal/ -l 500000 -f 10 -r 10f_apascal


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
parser.add_argument("-f", "--folds",
                    default=10,  type=int,
                    help=("Number of folds for cross validation"))
parser.add_argument("-r", "--r_name",
                    required=True,
                    help=("name of the pkl object to save the results"))

##########################
#####   FUNCTION      ####
##########################

def print_size_info():
    print ('seen_class_ids: ', seen_class_ids.shape)
    print ('seen_attr_mat: ', seen_attr_mat.shape)
    print ('unseen_class_ids: ', unseen_class_ids.shape)
    print ('umseen_attr_mat: ', unseen_attr_mat.shape)

def combine_splits(splits, split_idx):
    # split_idx in [0, len(splits)-1]
    train_splits = [splits[idx] for idx in range(len(splits)) if idx != split_idx]
    train_class_ids = [item for sublist in train_splits for item in sublist]
    valid_class_ids = splits[split_idx]
    return train_class_ids, valid_class_ids

def refresh_file_pointers(dict_key,file_name,data_path):
    fid[dict_key].close()
    fid[dict_key] = open(os.path.join(data_path,file_name),'r')

##########################
#####     INIT        ####
##########################
### Arguments
args = vars(parser.parse_args())
dataset_path = args['dataset_path']
lambda_val = args['lambda']
num_folds = args['folds']
obj_name = args['r_name']

### Init Variables
seen_in_dict = 'seen_input'
seen_in_dat = 'seen_data_input.dat'
unseen_in_dict = 'unseen_input'
unseen_in_dat = 'unseen_data_input.dat'
seen_out_dict = 'seen_output'
seen_out_dat = 'seen_data_output.dat'
unseen_out_dict = 'unseen_output'
unseen_out_dat = 'unseen_data_output.dat'
seen_dataset_dict = 'dataset_seen'
feature_size = 0

fid = {}
fid_count = {}
attr_path = os.path.join(dataset_path, 'attr_data.mat')
data = loadmat(attr_path)

##########################
####    DATA LOADING  ####
##########################
## Load file readers for seen Data
start_time = time.time()
seen_class_ids = torch.LongTensor(data['seen_class_ids'])
seen_attr_mat = torch.FloatTensor(data['seen_attr_mat'])
fid[seen_in_dict] = open(os.path.join(dataset_path,seen_in_dat),'r')
fid[seen_out_dict] = open(os.path.join(dataset_path,seen_out_dat),'r')
fid_count['seen'] = 0
print('Seen Data: ', time.time()-start_time)

## Load file readers for unseen Data
start_time = time.time()
unseen_class_ids = torch.FloatTensor(data['unseen_class_ids'])
unseen_attr_mat = torch.FloatTensor(data['unseen_attr_mat'])
fid[unseen_in_dict] = open(os.path.join(dataset_path,unseen_in_dat),'r')
fid[unseen_out_dict] = open(os.path.join(dataset_path,unseen_out_dat),'r')
fid_count['unseen'] = 0
print('Unseen Data: ', time.time()-start_time)



for feat_out in fid[seen_out_dict]:
    if (int(feat_out)-1) in train_class_ids:
        train_size += 1
    elif (int(feat_out)-1) in valid_class_ids:
        valid_size += 1
    else:
        print("Error: class not present in list")
        exit(1)
print("GOT HERE")

# Find train-validation splits
start_time = time.time()
num_classes = seen_attr_mat.shape[0]
class_indices = list(range(num_classes))
print(class_indices)
random.shuffle(class_indices)
num_classes_per_fold = int(math.floor(num_classes / num_folds ))
splits = [class_indices[fold_idx:fold_idx+num_classes_per_fold] for fold_idx in range(0,num_classes,num_classes_per_fold)]
if len(splits[-1]) < num_classes_per_fold:
    splits[-2] += (splits[-1])
    del splits[-1]
print('Find train-validation splits: ', time.time()-start_time)



## Load The splits in memory
## Get the feature size
for feat_in in fid[seen_in_dict]:
    feature_size = list(map(float,feat_in.split(',')))
    feature_size = len(feature_size)
    break
print('feature_size: ', feature_size)
refresh_file_pointers(seen_in_dict,seen_in_dat,dataset_path)

## Create the empty split tensors of the right size
#-----------------------
start_time = time.time()


dataset_classes = get_dataset_dict(dataset_path,fid[seen_out_dict],seen_dataset_dict)
print("dataset_classes: ", dataset_classes)
refresh_file_pointers(seen_out_dict,seen_out_dat,dataset_path)
splits_t = []
splits_semantic_t = []
description_size = seen_attr_mat.shape[1]
for split in splits:
    instance_num = sum([dataset_classes[label] for label in split])
    split_tensor = torch.zeros(feature_size,instance_num)
    split_semantic_tensor = torch.zeros(description_size,instance_num)
    splits_t.append(split_tensor)
print('Creating the empty split tensors: ', time.time()-start_time)

# Filling the empty tensors
#-----------------------
split_index = [0]*num_folds
start_time = time.time()
seen_train_index = 0
seen_valid_index = 0
for feat_in, feat_out in zip(fid[seen_in_dict], fid[seen_out_dict]):
    feat_out = int(feat_out) - 1
    feat_in_split = list(map(float,feat_in.split(',')))
    value_asigned = False
    for index in range(len(splits)):
        if int(feat_out) in splits[index]:
            splits_t[index][:,split_index[index]] = torch.FloatTensor(feat_in_split)
            train_semantic_t[:,split_index[index]] = seen_attr_mat[feat_out,:]
            split_index[index] += 1
            value_asigned = True
            break
    if not value_asigned:
        print("ERROR: Class not found in splits")
        exit(1)
print('Filling the empty tensors: ', time.time()-start_time)

results = {}
train_accuracy_list =[]
valid_accuracy_list =[]

exit()
##########################
#####     MAIN        ####
##########################
# Go over the folds for tunning
for index in range(num_folds):
# for index in range(1):
    # Combine the splits into training and validation
    train_class_ids, valid_class_ids = combine_splits(splits, index)

    # Finding training and validation attribute matrix
    # ----------------
    start_time = time.time()
    train_attr_mat = torch.zeros(len(train_class_ids), seen_attr_mat.size()[1])
    valid_attr_mat = torch.zeros(len(valid_class_ids), seen_attr_mat.size()[1])
    train_class_count = 0
    for class_id in train_class_ids:
        train_attr_mat[train_class_count,:] = seen_attr_mat[class_id,:]
        train_class_count += 1

    valid_class_count = 0
    for class_id in valid_class_ids:
        valid_attr_mat[valid_class_count,:] = seen_attr_mat[class_id,:]
        valid_class_count += 1
    print('Finding training and validation attribute matrix: ', time.time()-start_time)

    #Finding the dataset sizes
    #-------------------
    start_time = time.time()
    train_size = 0
    valid_size = 0
    description_size = seen_attr_mat.shape[1]
    feature_size = 0
    train_classes_size = len(train_class_ids)
    valid_classes_size = len(valid_class_ids)
    for feat_in in fid[seen_in_dict]:
        feature_size = list(map(float,feat_in.split(',')))
        feature_size = len(feature_size)
        break
    print('feature_size: ', feature_size)
    refresh_file_pointers(seen_in_dict,seen_in_dat,dataset_path)
    for feat_out in fid[seen_out_dict]:
        if (int(feat_out)-1) in train_class_ids:
            train_size += 1
        elif (int(feat_out)-1) in valid_class_ids:
            valid_size += 1
        else:
            print("Error: class not present in list")
            exit(1)
    print('splits: ', splits)
    print ('train_class_ids: ', train_class_ids)
    print ('valid_class_ids: ', valid_class_ids)
    print('train_size: ', train_size)
    print('valid_size: ', valid_size)
    print('Finding the dataset sizes: ', time.time()-start_time)

    refresh_file_pointers(seen_out_dict,seen_out_dat,dataset_path)

    # Create the empty tensors
    #---------------------
    start_time = time.time()
    train_t = torch.zeros(feature_size,train_size)
    valid_t = torch.zeros(feature_size,valid_size)
    print ('train_t size', train_t)
    train_semantic_t = torch.zeros(description_size,train_size)
    valid_semantic_t = torch.zeros(description_size,valid_size)
    print('description_size,train_size', description_size,train_size)
    print('train_semantic_t size: ', train_semantic_t)
    print('Create the empty tensors: ', time.time()-start_time)

    # Filling the empty tensors
    #-----------------------
    start_time = time.time()
    seen_train_index = 0
    seen_valid_index = 0
    for feat_in, feat_out in zip(fid[seen_in_dict], fid[seen_out_dict]):
        feat_out = int(feat_out) - 1 
        feat_in_split = list(map(float,feat_in.split(',')))
        if int(feat_out) in train_class_ids:
            # print("train t size:", train_t[:,seen_train_index].size())
            # print("train t receiving:", torch.FloatTensor(feat_in_split).size())
            train_t[:,seen_train_index] = torch.FloatTensor(feat_in_split)
            train_semantic_t[:,seen_train_index] = seen_attr_mat[feat_out,:]
            seen_train_index += 1
        elif int(feat_out) in valid_class_ids:
            valid_t[:,seen_valid_index] = torch.FloatTensor(feat_in_split)
            valid_semantic_t[:,seen_valid_index] = seen_attr_mat[feat_out,:]
            seen_valid_index += 1
        else:
            print("Error: class not present in list")
            exit(1)       
    print('Filling the empty tensors: ', time.time()-start_time)    

    # Computing A, B, C
    #-------------
    start_time = time.time()
    A = torch.mm(train_semantic_t,train_semantic_t.t())
    B = lambda_val*torch.mm(train_t,train_t.t())
    C = (1 + lambda_val)*torch.mm(train_semantic_t,train_t.t())
    print('Computing A, B, C: ', time.time()-start_time)

    # Solving the Sylvester
    #--------------
    start_time = time.time()
    print("Solving Sylvester")
    W = solve_sylvester(A.numpy(), B.numpy(), C.numpy())
    print('Time taken solving: ', time.time()-start_time)
    W = torch.FloatTensor(W)

    ## Compute training error
    start_time = time.time()
    train_semantic_pred = torch.mm(W, train_t)
    _, train_pred_classes = torch.max(torch.mm(train_attr_mat, train_semantic_pred), dim=0)
    _, train_true_classes = torch.max(torch.mm(train_attr_mat, train_semantic_t), dim=0)
    train_accuracy = torch.sum(train_pred_classes == train_true_classes) / train_pred_classes.numel()
    train_accuracy_list.append(train_accuracy)
    print('Compute training error: ', time.time()-start_time)

    ## Compute validation error
    start_time = time.time()
    valid_semantic_pred = torch.mm(W, valid_t)
    _, valid_pred_classes = torch.max(torch.mm(valid_attr_mat, valid_semantic_pred), dim=0)
    _, valid_true_classes = torch.max(torch.mm(valid_attr_mat, valid_semantic_t), dim=0)
    valid_accuracy = torch.sum(valid_pred_classes == valid_true_classes) / valid_pred_classes.numel()
    valid_accuracy_list.append(valid_accuracy)
    print('Compute validation error: ', time.time()-start_time)

    #Saving the files in a dictionary
    start_time = time.time()
    print('Training accuracy: ', train_accuracy_list[-1])
    print('Validation accuracy: ', valid_accuracy_list[-1])
    results['weights'] = W # Saving only the last set of weights.
    results['train_pred_classes'] = train_pred_classes
    results['train_true_classes'] = train_true_classes
    results['valid_pred_classes'] = valid_pred_classes
    results['valid_true_classes'] = valid_true_classes
    print('Saving the files in a dictionary: ', time.time()-start_time)

results['train_accuracy'] = train_accuracy_list
results['valid_accuracy'] = valid_accuracy_list

# test_t = torch.zeros(feature_size,train_size)

start_time = time.time()
save_object(results, obj_name)
print('Saving the pickle: ', time.time()-start_time)

exit()
    # val_t = torch.zeros(feature_size,valid_size)
    # val_train_semantic_t = torch.zeros(description_size,valid_size)


# M = torch.zeros(3, 2)
# training_in_t =
# training_out_t =
# validation_in_t =
# validation_out_t =

temp = []

for feat_in, feat_out in zip(fid[seen_in_dict], fid[seen_out_dict]):
    feat_in = list(map(float,feat_in.split(',')))
    print("Train Feat in",len(feat_in))
    break
    # print("Feat out",len(feat_out))
    # temp.append(feat_in)
    # exit()

count = 0
for feat_in, feat_out in zip(fid[seen_in_dict], fid[seen_out_dict]):
    count += 1


print("COUNT TRAIN: ",count)



for feat_in, feat_out in zip(fid[unseen_in_dict], fid[unseen_out_dict]):
    feat_in = list(map(float,feat_in.split(',')))
    print("Test Feat in",len(feat_in))
    break

count = 0
for feat_in, feat_out in zip(fid[unseen_in_dict], fid[unseen_out_dict]):
    count += 1

print("COUNT TEST: ",count)

# print len(temp), len(temp[0])

print_size_info()

fid[seen_in_dict].close()
fid[seen_out_dict].close()
fid[unseen_in_dict].close()
fid[unseen_out_dict].close()

## W = solve_sylvester(A, B, C)
