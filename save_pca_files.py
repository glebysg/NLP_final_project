from scipy.io import loadmat
import os
import argparse
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
                    help=("Number of folds for cross testation"))
parser.add_argument("-pca_dim", "--pca_dim",
                    required=True,
                    help=("Reduced Dimension after applying PCA"))

def refresh_file_pointers(dict_key,file_name,data_path):
    fid[dict_key].close()
    fid[dict_key] = open(os.path.join(data_path,file_name),'r')

args = vars(parser.parse_args())
dataset_path = args['dataset_path']
lambda_val = args['lambda']
num_folds = args['folds']
pca_dim = args['pca_dim']

fid = {}
fid['seen_input'] = open(os.path.join(dataset_path,'seen_data_input.dat'),'r')
fid['unseen_input'] = open(os.path.join(dataset_path,'unseen_data_input.dat'),'r')

data = loadmat(os.path.join(dataset_path, 'attr_data.mat'))
seen_class_ids = torch.LongTensor(data['seen_class_ids'])
seen_attr_mat = torch.FloatTensor(data['seen_attr_mat'])
unseen_class_ids = torch.FloatTensor(data['unseen_class_ids'])
unseen_attr_mat = torch.FloatTensor(data['unseen_attr_mat'])
description_size = seen_attr_mat.shape[1]

train_size = 0
test_size = 0
feature_size = 0
train_classes_size = len(train_class_ids)
test_classes_size = len(test_class_ids)
for feat_in in fid['seen_input']:
    feature_size = list(map(float,feat_in.split(',')))
    feature_size = len(feature_size)
    break
print('feature_size: ', feature_size)
refresh_file_pointers('seen_input','seen_data_input.dat',dataset_path)
for feat_out in fid['seen_output']:
    if (int(feat_out)-1) in train_class_ids:
        train_size += 1
    elif (int(feat_out)-1) in test_class_ids:
        test_size += 1
    else:
        print("Error: class not present in list")
        exit(1)


# Create the empty tensors
#---------------------
start_time = time.time()
train_t = torch.zeros(feature_size,train_size)
test_t = torch.zeros(feature_size,test_size)
print ('train_t size', train_t.size())
print('description_size,train_size', description_size,train_size)
print('train_semantic_t size: ', train_semantic_t.size())
print('Create the empty tensors: ', time.time()-start_time)

# Filling the empty tensors
#-----------------------
start_time = time.time()
seen_train_index = 0
test_index = 0
for feat_in in fid['seen_input']: 
    feat_in_split = list(map(float,feat_in.split(',')))
    train_t[:,seen_train_index] = torch.FloatTensor(feat_in_split)
    seen_train_index += 1
for feat_in in fid['unseen_input']: 
    feat_in_split = list(map(float,feat_in.split(',')))
    test_t[:,test_index] = torch.FloatTensor(feat_in_split)
    test_index += 1
print('Filling the empty tensors: ', time.time()-start_time)    

## Applying PCA
#--------------
start_time = time.time()
pca = PCA(n_components=int(pca_feature_size), svd_solver='arpack')
pca.fit(train_t.numpy().transpose())
train_t = torch.FloatTensor(pca.transform(train_t.numpy().transpose()).transpose())
print('train_t size: ', train_t.size(), type(train_t))
print('PCA Time: ', time.time()-start_time)


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)

print('pca.explained_variance_ratio_: ', pca.explained_variance_ratio_)  
print('pca.singular_values_: ', pca.singular_values_)  

X_pca = pca.fit_transform(X)
Xp = pca.inverse_transform(X_pca)

print('X_pca: ', X_pca)
print('Xp: ', Xp)

U, S, V = pca._fit(X)
print('U: ', U)
print('S: ', S)
print('V: ', V)