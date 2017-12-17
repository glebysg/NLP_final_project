from scipy.io import loadmat
from scipy.linalg import solve_sylvester
import os
import random
import math
import argparse
import torch
import pickle
import time
import numpy as np
from sklearn.decomposition import PCA

# Example usage:
# python read_dataset.py -p ./data/apascal/ -l 500000 -f 10 -r 10f_apascal

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

def save_object(obj, filename):
        with open(filename, 'wb') as output:
                    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

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

args = vars(parser.parse_args())
dataset_path = args['dataset_path']
lambda_val = args['lambda']
num_folds = args['folds']
obj_name = args['r_name']
#dataset_path = '.\\apascal'
# dataset_path = '.\\animals'
#dataset_path = '.\\sun'

fid = {}
fid_count = {}
attr_path = os.path.join(dataset_path, 'attr_data.mat')
data = loadmat(attr_path)

pca_feature_size = 1000

## Seen Data
start_time = time.time()
seen_class_ids = torch.LongTensor(data['seen_class_ids'])
seen_attr_mat = torch.FloatTensor(data['seen_attr_mat'])
fid['seen_input'] = open(os.path.join(dataset_path,'seen_data_input.dat'),'r')
fid['seen_output'] = open(os.path.join(dataset_path,'seen_data_output.dat'),'r')
fid_count['seen'] = 0
print('Seen Data: ', time.time()-start_time)

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

## Unseen Data
start_time = time.time()
unseen_class_ids = torch.FloatTensor(data['unseen_class_ids'])
unseen_attr_mat = torch.FloatTensor(data['unseen_attr_mat'])
fid['unseen_input'] = open(os.path.join(dataset_path,'unseen_data_input.dat'),'r')
fid['unseen_output'] = open(os.path.join(dataset_path,'unseen_data_output.dat'),'r')
fid_count['unseen'] = 0
print('Unseen Data: ', time.time()-start_time)
# print('seen_attr_mat: ', seen_attr_mat)
# print('unseen_attr_mat: ', unseen_attr_mat)

## 
# For each split in splits: create 
# split_attr
# for split in splits:

# exit()

## Count the dimentions of the training and validation folds
results = {}
train_accuracy_list =[]
valid_accuracy_list =[]

# for index in range(len(train_class_ids)):
for index in range(1):
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
    for feat_in in fid['seen_input']:
        feature_size = list(map(float,feat_in.split(',')))
        feature_size = len(feature_size)
        break
    print('feature_size: ', feature_size)
    refresh_file_pointers('seen_input','seen_data_input.dat',dataset_path)
    for feat_out in fid['seen_output']:
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

    refresh_file_pointers('seen_output','seen_data_output.dat',dataset_path)

    # Create the empty tensors
    #---------------------
    start_time = time.time()
    train_t = torch.zeros(feature_size,train_size)
    valid_t = torch.zeros(feature_size,valid_size)
    print ('train_t size', train_t.size())
    train_semantic_t = torch.zeros(description_size,train_size)
    valid_semantic_t = torch.zeros(description_size,valid_size)
    print('description_size,train_size', description_size,train_size)
    print('train_semantic_t size: ', train_semantic_t.size())
    print('Create the empty tensors: ', time.time()-start_time)

    # Filling the empty tensors
    #-----------------------
    start_time = time.time()
    seen_train_index = 0
    seen_valid_index = 0
    for feat_in, feat_out in zip(fid['seen_input'], fid['seen_output']):
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

    ## Applying PCA
    #--------------
    start_time = time.time()
    pca = PCA(n_components=int(pca_feature_size), svd_solver='arpack')
    pca.fit(train_t.numpy().transpose())
    train_t = torch.FloatTensor(pca.transform(train_t.numpy().transpose()).transpose())
    print('train_t size: ', train_t.size(), type(train_t))
    print('PCA Time: ', time.time()-start_time)

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
    valid_t = torch.FloatTensor(pca.transform(valid_t.numpy().transpose()).transpose())
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

for feat_in, feat_out in zip(fid['seen_input'], fid['seen_output']):
    feat_in = list(map(float,feat_in.split(',')))
    print("Train Feat in",len(feat_in))
    break
    # print("Feat out",len(feat_out))
    # temp.append(feat_in)
    # exit()

count = 0
for feat_in, feat_out in zip(fid['seen_input'], fid['seen_output']):
    count += 1


print("COUNT TRAIN: ",count)



for feat_in, feat_out in zip(fid['unseen_input'], fid['unseen_output']):
    feat_in = list(map(float,feat_in.split(',')))
    print("Test Feat in",len(feat_in))
    break

count = 0
for feat_in, feat_out in zip(fid['unseen_input'], fid['unseen_output']):
    count += 1

print("COUNT TEST: ",count)

# print len(temp), len(temp[0])

print_size_info()

fid['seen_input'].close()
fid['seen_output'].close()
fid['unseen_input'].close()
fid['unseen_output'].close()

## W = solve_sylvester(A, B, C)
