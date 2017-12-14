from scipy.io import loadmat
from scipy.linalg import solve_sylvester
import os
import random
import math

def print_size_info():
    print 'seen_class_ids: ', seen_class_ids.shape
    print 'seen_attr_mat: ', seen_attr_mat.shape
    print 'unseen_class_ids: ', unseen_class_ids.shape
    print 'umseen_attr_mat: ', unseen_attr_mat.shape

def combine_splits(splits, split_idx):
    # split_idx in [0, len(splits)-1]
    train_split = [split for split in splits[]]
    valid_split = splits[split_idx]

#dataset_path = '.\\apascal'
dataset_path = '.\\animals'
#dataset_path = '.\\sun'

num_folds = 10

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
class_indices = range(num_classes)
random.shuffle(class_indices)
num_classes_per_fold = int(math.floor(num_classes / num_folds ))
splits = [class_indices[fold_idx:fold_idx+num_classes_per_fold] for fold_idx in range(0,num_classes,num_classes_per_fold)]
if len(splits[-1]) < num_classes_per_fold:
    splits[-2] += (splits[-1])
    del splits[-1]
print splits

## Unseen Data
unseen_class_ids = data['unseen_class_ids']
unseen_class_ids.shape = (unseen_class_ids.shape[0],)
unseen_attr_mat = data['unseen_attr_mat']
fid['unseen_input'] = open(os.path.join(dataset_path,'unseen_data_input.dat'),'r')
fid['unseen_output'] = open(os.path.join(dataset_path,'unseen_data_output.dat'),'r')
fid_count['unseen'] = 0

# temp = []
# for feat_in, feat_out in zip(fid['seen_input'], fid['seen_output']):
#     feat_in = map(float,feat_in.split(','))
#     print len(feat_in)
#     temp.append(feat_in)

# print len(temp), len(temp[0])

print_size_info()

fid['seen_input'].close()
fid['seen_output'].close()
fid['unseen_input'].close()
fid['unseen_output'].close()

## W = solve_sylvester(A, B, C)
