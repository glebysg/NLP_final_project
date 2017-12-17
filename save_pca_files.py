from scipy.io import loadmat
import os
import argparse
import time
import numpy as np
from sklearn.decomposition import PCA
import torch
import random

# Example usage:
# python save_pca_files.py -p data/animals_sae -d 1000 -type pca -write False

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dataset_path",
                    default='./data/apascal/',
                    help=("full path to the dataset"))
parser.add_argument("-d", "--reduced_dim",
                    default=1000,
                    help=("Reduced Dimension after applying PCA"))
parser.add_argument("-type", "--type",
                    required=True,
                    help=("pca or rf"))
parser.add_argument("-write", "--write_flag",
                    required=True,
                    help=("pca or rf"))

def refresh_file_pointers(dict_key,file_name,data_path):
    fid[dict_key].close()
    fid[dict_key] = open(os.path.join(data_path,file_name),'r')

args = vars(parser.parse_args())
dataset_path = args['dataset_path']
reduced_dim = int(args['reduced_dim'])
method = args['type'].lower()
write_flag = args['write_flag'].lower()

fid = {}
fid['seen_input'] = open(os.path.join(dataset_path,'seen_data_input.dat'),'r')
fid['unseen_input'] = open(os.path.join(dataset_path,'unseen_data_input.dat'),'r')

data = loadmat(os.path.join(dataset_path, 'attr_data.mat'))
seen_attr_mat = torch.FloatTensor(data['seen_attr_mat'])
unseen_attr_mat = torch.FloatTensor(data['unseen_attr_mat'])

train_classes_size = seen_attr_mat.shape[0]
test_classes_size = unseen_attr_mat.shape[0]
description_size = seen_attr_mat.shape[1]

train_size = 0
test_size = 0
feature_size = 0
for feat_in in fid['seen_input']:
    feature_size = list(map(float,feat_in.split(',')))
    feature_size = len(feature_size)
    break
refresh_file_pointers('seen_input','seen_data_input.dat',dataset_path)
for feat_in in fid['seen_input']:
	train_size += 1
refresh_file_pointers('seen_input','seen_data_input.dat',dataset_path)
for feat_in in fid['unseen_input']:
	test_size += 1
refresh_file_pointers('unseen_input','unseen_data_input.dat',dataset_path)
print('feature_size: ', feature_size)
print('train_size: ', train_size)
print('test_size: ', test_size)
print('description_size: ', description_size)
print('')

# Create the empty tensors
#---------------------
print('Create the empty tensors')
start_time = time.time()
train_t = torch.zeros(feature_size,train_size)
test_t = torch.zeros(feature_size,test_size)
print('Time taken: ', time.time()-start_time)
print('')

# Filling the empty tensors
#-----------------------
print('Filling the empty tensors')
start_time = time.time()
train_index = 0
test_index = 0
for feat_in in fid['seen_input']: 
    feat_in_split = list(map(float,feat_in.split(',')))
    train_t[:,train_index] = torch.FloatTensor(feat_in_split)
    train_index += 1
for feat_in in fid['unseen_input']: 
    feat_in_split = list(map(float,feat_in.split(',')))
    test_t[:,test_index] = torch.FloatTensor(feat_in_split)
    test_index += 1
print('Time taken: ', time.time()-start_time)    
print('')

print('Before : train_t size: ', train_t.size())
print('Before : test_t size: ', test_t.size(), '\n')

if(method == 'pca'):
	## Applying PCA
	#--------------
	print('Applying PCA')
	start_time = time.time()
	pca = PCA(n_components=int(reduced_dim), svd_solver='arpack')
	pca.fit(train_t.numpy().transpose())
	train_t = torch.FloatTensor(pca.transform(train_t.numpy().transpose()).transpose())
	test_t = torch.FloatTensor(pca.transform(test_t.numpy().transpose()).transpose())
	print('Time taken: ', time.time()-start_time, '\n')
elif(method == 'rf'):
	print('Applying RF')
	start_time = time.time()
	perm = list(range(int(feature_size)))
	random.shuffle(perm)
	perm = perm[:reduced_dim]
	train_t = train_t[perm,:]
	test_t = test_t[perm,:]
	print('Time taken: ', time.time()-start_time, '\n')

print('After : train_t size: ', train_t.size())
print('After : test_t size: ', test_t.size(), '\n')

if write_flag == 'true':
	print('Writing to files')
	start_time = time.time()
	np.savetxt(os.path.join(dataset_path,method+'_seen_data_input.dat'), train_t.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(dataset_path,method+'_unseen_data_input.dat'), test_t.numpy().transpose(), delimiter=',')
	print('Time taken: ', time.time()-start_time, '\n')

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# pca = PCA(n_components=1, svd_solver='arpack')
# pca.fit(X)

# print('pca.explained_variance_ratio_: ', pca.explained_variance_ratio_)  
# print('pca.singular_values_: ', pca.singular_values_)  

# X_pca = pca.fit_transform(X)
# Xp = pca.inverse_transform(X_pca)

# print('X_pca: ', X_pca)
# print('Xp: ', Xp)

# U, S, V = pca._fit(X)
# print('U: ', U)
# print('S: ', S)
# print('V: ', V)