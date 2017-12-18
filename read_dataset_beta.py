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
parser.add_argument("-b", "--beta",
                    default=0.5, type=float,
                    help=("Beta parameter"))
parser.add_argument("-f", "--folds",
                    default=10,  type=int,
                    help=("Number of folds for cross validation"))
parser.add_argument("-t", "--tunning_steps",
                    default=2,  type=int,
                    help=("Number of tunning step loops"))
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
tunning_steps = args['tunning_steps']

### Init Variables
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
feature_size = 0

fid = {}
attr_path = os.path.join(dataset_path, 'attr_data.mat')
data = loadmat(attr_path)

## Seen Data
##########################
####    DATA LOADING  ####
##########################
## Load file readers for seen Data
start_time = time.time()
seen_class_ids = torch.LongTensor(data['seen_class_ids'])
seen_attr_mat = torch.FloatTensor(data['seen_attr_mat'])
fid[seen_in_dict] = open(os.path.join(dataset_path,seen_in_dat),'r')
fid[seen_out_dict] = open(os.path.join(dataset_path,seen_out_dat),'r')
print('Seen Data: ', time.time()-start_time)

## Load file readers for unseen Data
start_time = time.time()
unseen_class_ids = torch.FloatTensor(data['unseen_class_ids'])
unseen_attr_mat = torch.FloatTensor(data['unseen_attr_mat'])
fid[unseen_in_dict] = open(os.path.join(dataset_path,unseen_in_dat),'r')
fid[unseen_out_dict] = open(os.path.join(dataset_path,unseen_out_dat),'r')
print('Unseen Data: ', time.time()-start_time)


# Find train-validation splits
start_time = time.time()
num_classes = seen_attr_mat.shape[0]
class_indices = list(range(num_classes))
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
refresh_file_pointers(seen_out_dict,seen_out_dat,dataset_path)
splits_t = []
splits_semantic_t = []
description_size = seen_attr_mat.shape[1]
for split in splits:
    instance_num = sum([dataset_classes[label] for label in split])
    split_tensor = torch.zeros(feature_size,instance_num)
    split_semantic_tensor = torch.zeros(description_size,instance_num)
    splits_t.append(split_tensor)
    splits_semantic_t.append(split_semantic_tensor)
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
            splits_semantic_t[index][:,split_index[index]] = seen_attr_mat[feat_out,:]
            split_index[index] += 1
            value_asigned = True
            break
    if not value_asigned:
        print("ERROR: Class not found in splits")
        exit(1)
print('Filling the empty tensors: ', time.time()-start_time)
refresh_file_pointers(seen_in_dict,seen_in_dat,dataset_path)
refresh_file_pointers(seen_out_dict,seen_out_dat,dataset_path)

results = {}
full_train_accuracy_list =[]
full_valid_accuracy_list =[]

##########################
#####     MAIN        ####
##########################

# Tune the parameters
begin_val = 0.001
end_val = 1
exploration_step = 0.1
pivot = begin_val
best_pivot = pivot
best_valid_acc = 0.0
best_W = None
for tune_step in range(tunning_steps):
    # Go over the folds for tunning
    while(pivot <= end_val):
        train_accuracy_list =[]
        valid_accuracy_list =[]
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
            # print('Finding training and validation attribute matrix: ', time.time()-start_time)

            #Build the splits
            #-------------------
            start_time = time.time()
            # Get training and validation size:
            train_size = sum([dataset_classes[label] for label in train_class_ids])
            valid_size = sum([dataset_classes[label] for label in valid_class_ids])
            # Create the empty tensors
            train_t = torch.zeros(feature_size,train_size)
            valid_t = torch.zeros(feature_size,valid_size)
            train_semantic_t = torch.zeros(description_size,train_size)
            valid_semantic_t = torch.zeros(description_size,valid_size)
            # Build the training
            init_index = 0
            end_index = 0
            for fold_index in range(num_folds):
                if fold_index != index:
                    end_index = init_index+\
                            sum([dataset_classes[label] for label in splits[fold_index]])
                    train_t[:,range(init_index,end_index)] = splits_t[fold_index]
                    train_semantic_t[:,range(init_index,end_index)] = splits_semantic_t[fold_index]
                    init_index = end_index
            # Build the validation
            valid_t[:,:] = splits_t[index]
            valid_semantic_t[:,:] = splits_semantic_t[index]
            # print('Build the Fold: ', time.time()-start_time)

            # Computing A, B, C
            #-------------
            start_time = time.time()
            A = torch.mm(train_semantic_t,train_semantic_t.t())
            rows, cols = A.size()
            A = A + torch.ones(rows,cols)*pivot
            B = lambda_val*torch.mm(train_t,train_t.t())
            C = (1 + lambda_val)*torch.mm(train_semantic_t,train_t.t())
            # print('Computing A, B, C: ', time.time()-start_time)

            # Solving the Sylvester
            #--------------
            start_time = time.time()
            # print("Solving Sylvester")
            W = solve_sylvester(A.numpy(), B.numpy(), C.numpy())
            # print('Time taken solving: ', time.time()-start_time)
            W = torch.FloatTensor(W)
            W = f.normalize(W,p=2,dim=1)

            ## Compute training error
            train_semantic_pred = torch.mm(W, train_t)
            _, train_pred_classes = torch.max(torch.mm(train_attr_mat, train_semantic_pred), dim=0)
            _, train_true_classes = torch.max(torch.mm(train_attr_mat, train_semantic_t), dim=0)
            train_accuracy = torch.sum(train_pred_classes == train_true_classes) / train_pred_classes.numel()
            train_accuracy_list.append(train_accuracy)

            ## Compute validation error
            start_time = time.time()
            # valid_t = torch.FloatTensor(pca.transform(valid_t.numpy().transpose()).transpose())
            valid_semantic_pred = torch.mm(W, valid_t)
            _, valid_pred_classes = torch.max(torch.mm(valid_attr_mat, valid_semantic_pred), dim=0)
            _, valid_true_classes = torch.max(torch.mm(valid_attr_mat, valid_semantic_t), dim=0)
            valid_accuracy = torch.sum(valid_pred_classes == valid_true_classes) / valid_pred_classes.numel()
            valid_accuracy_list.append(valid_accuracy)

            #Saving the files in a dictionary
            # start_time = time.time()
            # print('Training accuracy: ', train_accuracy_list[-1])
            # print('Validation accuracy: ', valid_accuracy_list[-1])
            # results['weights'] = W # Saving only the last set of weights.
            # results['train_pred_classes'] = train_pred_classes
            # results['train_true_classes'] = train_true_classes
            # results['valid_pred_classes'] = valid_pred_classes
            # results['valid_true_classes'] = valid_true_classes
            # print('Saving the files in a dictionary: ', time.time()-start_time)
        current_acc = np.mean(valid_accuracy_list)
        if current_acc > best_valid_acc:
            best_W = W
            best_valid_acc = current_acc
            best_train_acc = np.median(train_accuracy_list)
            best_pivot = pivot
        print("Current Pivot: ", pivot)
        print("Current Current acc: ", best_valid_acc)
        full_train_accuracy_list = [pivot] + train_accuracy_list
        full_valid_accuracy_list = [pivot] + valid_accuracy_list
        pivot += exploration_step
    pivot = best_pivot
    begin_val = max(begin_val,pivot-exploration_step)
    end_val = min(end_val,pivot+exploration_step)
    exploration_step = exploration_step/10
print("BEST PIVOT:",best_pivot)
print("BEST PIVOT:",best_valid_acc)

results['best_lambda'] = best_pivot
results['lambda_train_acc'] = best_train_acc
results['lambda_valid_acc'] = best_valid_acc
results['weight'] = best_W
results['full_train_accuracy_list'] = full_train_accuracy_list
results['full_valid_accuracy_list'] = full_valid_accuracy_list
# test_t = torch.zeros(feature_size,train_size)

start_time = time.time()
save_object(results, obj_name)
print('Saving the pickle: ', time.time()-start_time)
