from scipy.io import loadmat
from scipy.linalg import solve_sylvester
import os
import random
import math
import argparse
import torch
import pickle

# Example usage:
# python read_dataset.py -p ./data/apascal/ -l 500000 -f 10

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dataset_path",
                    default='./data/apascal/',
                    help=("full path to the dataset"))
parser.add_argument("-l", "--lambda",
                    default=500000, type=float,
                    help=("full path to the dataset"))
parser.add_argument("-f", "--folds",
                    default=10,  type=int,
                    help=("full path to the dataset"))

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
    fid[dict_key] = open(os.path.join(data_path,file_name),'r')

def close_file_pointers(dict_key):
    fid[dict_key].close()

args = vars(parser.parse_args())
dataset_path = args['dataset_path']
lambda_val = args['lambda']
num_folds = args['folds']
#dataset_path = '.\\apascal'
# dataset_path = '.\\animals'
#dataset_path = '.\\sun'

fid = {}
fid_count = {}
attr_path = os.path.join(dataset_path, 'attr_data.mat')
data = loadmat(attr_path)

## Seen Data
seen_class_ids = data['seen_class_ids']
seen_class_ids.shape = (seen_class_ids.shape[0],)
seen_attr_mat = data['seen_attr_mat']
fid['seen_input'] = open(os.path.join(dataset_path,'seen_data_input.dat'),'r')
fid['seen_output'] = open(os.path.join(dataset_path,'seen_data_output.dat'),'r')
fid_count['seen'] = 0
# Find train-validation splits
num_classes = seen_attr_mat.shape[0]
class_indices = list(range(num_classes))
print(class_indices)
random.shuffle(class_indices)
num_classes_per_fold = int(math.floor(num_classes / num_folds ))
splits = [class_indices[fold_idx:fold_idx+num_classes_per_fold] for fold_idx in range(0,num_classes,num_classes_per_fold)]
if len(splits[-1]) < num_classes_per_fold:
    splits[-2] += (splits[-1])
    del splits[-1]


## Unseen Data
unseen_class_ids = data['unseen_class_ids']
unseen_class_ids.shape = (unseen_class_ids.shape[0],)
unseen_attr_mat = data['unseen_attr_mat']
fid['unseen_input'] = open(os.path.join(dataset_path,'unseen_data_input.dat'),'r')
fid['unseen_output'] = open(os.path.join(dataset_path,'unseen_data_output.dat'),'r')
fid_count['unseen'] = 0

## Count the dimentions of the training and validation folds
# for index in range(len(train_class_ids)):
for index in range(1):
    train_class_ids, valid_class_ids = combine_splits(splits, index)
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
    close_file_pointers('seen_input')
    refresh_file_pointers('seen_input','seen_data_input.dat',dataset_path)
    for feat_out in fid['seen_output']:
        if int(feat_out) in train_class_ids:
            train_size += 1
        if int(feat_out) in valid_class_ids:
            valid_size += 1
    close_file_pointers('seen_output')
    refresh_file_pointers('seen_output','seen_data_output.dat',dataset_path)
    # Create the tensors
    train_t = torch.zeros(feature_size,train_size)
    semantic_t = torch.zeros(description_size,train_size)
    print(description_size,train_size)
    print(semantic_t)
    w_t = torch.zeros(description_size,feature_size)
    seen_train_index = 0
    for feat_in, feat_out in zip(fid['seen_input'], fid['seen_output']):
        feat_out = int(feat_out)
        feat_in_split = list(map(float,feat_in.split(',')))
        if int(feat_out) in train_class_ids:
            # print("train t size:", train_t[:,seen_train_index].size())
            # print("train t receiving:", torch.FloatTensor(feat_in_split).size())
            train_t[:,seen_train_index] = torch.FloatTensor(feat_in_split)
            semantic_t[:,seen_train_index] = torch.FloatTensor(seen_attr_mat[feat_out,:])
        seen_train_index += 1

    A = torch.mm(semantic_t,semantic_t.t())
    B = lambda_val*torch.mm(train_t,train_t.t())
    C = (1 + lambda_val)*torch.mm(semantic_t,train_t.t())

    W = solve_sylvester(A.numpy(), B.numpy(), C.numpy())
    save_object(W,'first_w')
exit()
    # val_t = torch.zeros(feature_size,valid_size)
    # val_semantic_t = torch.zeros(description_size,valid_size)


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
